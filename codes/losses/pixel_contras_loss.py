import torch
import torch.nn.functional as F
from abc import ABC
from torch import nn

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=100,
                 ignore_index=-1,
                 device='cuda:0'):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

        self.device = device

    def _anchor_sampling(self, X, y_hat, y):
        """
        对每个区域的特征进行正负样本采样
        X: 特征, (batch_size, num_regions, feat_dim)
        y_hat: 预测标签, (batch_size, num_regions)
        y: 真正的标签, (batch_size, num_regions)
        """
        batch_size, num_regions, feat_dim = X.shape

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]

            # 获取区域内的类别
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)
            
        if total_classes == 0:
            return None, None

        # 每个类最多选择的样本数量
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        # 用于存储采样的正负样本
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                # 选择 hard 和 easy 样本
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    continue

                # 从 hard 和 easy 样本中随机采样
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                # 将采样的特征和标签存储
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, memory_size, feat_size = Q.shape

        x_ = torch.zeros((class_num * memory_size, feat_size)).float().to(self.device)
        y_ = torch.zeros((class_num * memory_size, 1)).float().to(self.device)

        sample_ptr = 0
        for c in range(class_num):
            if c == 0:
                continue
            this_q = Q[c, :memory_size, :]
            x_[sample_ptr:sample_ptr + memory_size, ...] = this_q
            y_[sample_ptr:sample_ptr + memory_size, ...] = c
            sample_ptr += memory_size
        return x_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        """
        区域级别的对比学习
        X_anchor: anchor 特征, (num_classes, n_view, feat_dim)
        y_anchor: anchor 对应的标签, (num_classes)
        queue: 存储的历史负样本队列
        """
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        # 展开 anchor 特征和标签
        y_anchor = y_anchor.contiguous().view(-1, 1)  # (anchor_num × n_view) × 1
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)  # (anchor_num × n_view) × feat_dim

        # 如果有负样本队列，合并正负样本
        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        # 计算对比矩阵
        mask = torch.eq(y_anchor, y_contrast.T).float().to(self.device)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        # 处理标签和预测值，确保其大小与特征图一致
        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')

        if predict is not None:
            predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')
        else:
            predict = labels.clone()

        labels = labels.long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        # 调整 labels 和 predict 的形状
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)  # (N, H, W, C)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # (N, HW, C)

        # 采样 anchor 和正负样本
        feats_, labels_ = self._anchor_sampling(feats, labels, predict)

        if feats_ is None or labels_ is None:
            return torch.tensor(0.0, device=feats.device)  # 如果采样无效，返回 0 损失

        # 计算对比损失
        loss = self._contrastive(feats_, labels_, queue=queue)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, tau=5, n_class=4, bg=False, norm=True):
        super(ContrastiveLoss, self).__init__()
        self._tau = tau
        # self._n_class = n_class
        # self._bg = bg
        self._norm = norm

    def forward(self, centroid_s, centroid_t, bg=False, split=False):  # centroid_s (4, 32)
        norm_t = torch.norm(centroid_t, p=2, dim=1, keepdim=True)  # (4, 1)
        if self._norm:
            norm_s = torch.norm(centroid_s, p=2, dim=1, keepdim=True)  # (4, 1) compute the L2 norm of each centroid
            centroid_s = centroid_s / (norm_s + 1e-7)
            centroid_t = centroid_t / (norm_t + 1e-7)
        # a matrix with shape (#class, 2 * #class) storing the exponential values between two centroids
        # centroid_matrix = torch.zeros(n_class, 2 * n_class)
        # n_class = centroid_s.size()[0]
        # loss = 0
        # for i in range(0 if bg else 1, n_class):
        #     exp_sum = 0
        #     exp_self = 0
        #     for j in range(n_class):
        #         if i == j:
        #             exp_self = exp_func(centroid_t[i], centroid_s[j], tau=self._tau) + \
        #                        exp_func(centroid_t[i], centroid_t[j], tau=self._tau)
        #             exp_sum = exp_sum + exp_self
        #         else:
        #             exp_sum = exp_sum + exp_func(centroid_t[i], centroid_s[j], tau=self._tau) + \
        #                       exp_func(centroid_t[i], centroid_t[j], tau=self._tau)
        #     logit = -torch.log(exp_self / (exp_sum + 1e-7))
        #     loss = loss + logit
        exp_mm = torch.exp(torch.mm(centroid_t, centroid_s.transpose(0, 1)))
        exp_mm_t = torch.exp(torch.mm(centroid_t, centroid_t.transpose(0, 1)))
        diag_idx = torch.arange(0 if bg else 1, 4, dtype=torch.long)
        denom = exp_mm[0 if bg else 1:].sum(dim=1) + exp_mm_t[0 if bg else 1:].sum(dim=1)
        if split:
            nom1, nom2 = exp_mm[diag_idx, diag_idx], exp_mm_t[diag_idx, diag_idx]
            logit = 0.5 * (-torch.log(nom1 / (denom + 1e-7)) - torch.log(nom2 / (denom + 1e-7)))
        else:
            nom = exp_mm[diag_idx, diag_idx] + exp_mm_t[diag_idx, diag_idx]
            logit = -torch.log(nom / (denom + 1e-7))
        loss = logit.sum()
        return loss