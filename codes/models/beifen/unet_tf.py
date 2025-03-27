import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.swin_decoder import SwinTransDecoder
from codes.models._base import BaseModel2D
from codes.utils.init import kaiming_normal_init_weight
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
import pdb
#将输入特征映射到一个固定维度的嵌入空间
class EmbeddingHead(nn.Module):
    def __init__(self, dim_in, embed_dim=256, embed='convmlp'):
        super(EmbeddingHead, self).__init__()

         #dim_in：输入特征的通道数。
         #embed_dim：嵌入特征的维度，默认为 256。
         #embed：选择嵌入方法，默认为 'convmlp'
        if embed == 'linear':
            self.embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)   #使用一个1x1卷积层将输入特征直接映射到嵌入维度。这个过程相当于一个线性变换。
        elif embed == 'convmlp':
            self.embed = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),              #先使用1x1卷积层，保持输入通道不变。
                nn.BatchNorm2d(dim_in),                                #对卷积后的特征进行归一化
                nn.ReLU(),                                             #激活
                nn.Conv2d(dim_in, embed_dim, kernel_size=1)            #再次执行 1x1 卷积，将输入特征映射到嵌入维度
            )

    def forward(self, x):
        return F.normalize(self.embed(x), p=2, dim=1)


class UNetTF(BaseModel2D):

    def __init__(self,
                 encoder_name="resnet50",                       #编码器名
                 encoder_depth=5,                               #编码器的深度
                 encoder_weights="imagenet",                    #编码器的预训练权重
                 decoder_use_batchnorm=True,                    #是否在解码器上使用批归一化
                 decoder_channels=(256, 128, 64, 32, 16),       #解码器的每个阶段的通道数，表示解码器中每一层输出特征图的通道数
                 decoder_attention_type=None,                   #解码器中使用的注意力机制类型
                 in_channels=3,                                 #输入图像的通道数为3
                 classes=2,                                     #输出的类别数，表示网络需要预测的类别数
                 activation=None,                               #输出层的激活函数类型
                 embed_dim=96,                                  #嵌入特征的维度
                 norm_layer=nn.LayerNorm,                       #使用的归一化层类型
                 img_size=224,                                  #输入图像尺寸
                 patch_size=4,                                  #用于将图像切分成小块（patch）的尺寸
                 depths=[2, 2, 2, 2],                           #每个阶段的 transformer 层的数量
                 num_heads=[3, 6, 12, 24],                      #多头注意力机制中每个阶段的头数
                 window_size=7,                                 #窗口注意力机制中的窗口大小
                 qkv_bias=True,                                 #是否在查询、键、值投影时使用偏置
                 qk_scale=None,                                 #用于缩放查询和键的可选比例因子
                 drop_rate=0.,
                 attn_drop_rate=0.,                             #一般和注意力层的 dropout 概率有关
                 use_checkpoint=False,                          #是否在训练时使用检查点以节省内存
                 ape=True,                                      #是否使用绝对位置编码
                 cls=True,                                      #是否使用分类 token
                 contrast_embed=False,                          #是否使用对比学习嵌入
                 contrast_embed_dim=256,                        #对比学习嵌入的维度
                 contrast_embed_index=-3,                       #用于选择对比学习嵌入的位置，默认值 -3
                 mlp_ratio=4.,                                  #多层感知机（MLP）的宽度与嵌入维度的比率
                 drop_path_rate=0.1,                            #路径 dropout 的比例
                 final_upsample="expand_first",                 #最后一次上采样的方法，可能用于恢复图像的原始分辨率
                 patches_resolution=[56, 56]                    #切分的小块的分辨率
                 ): 
        super().__init__()
        self.cls = cls
        self.contrast_embed_index = contrast_embed_index

        #Encoder Resnet50
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels
        #CNN Decoder
        self.cnn_decoder = UnetDecoder(
            encoder_channels=encoder_channels,                 #encoder_channels：编码器输出的通道数；decoder_channels：解码器中每一层的通道数
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,                            #解码器的层数，与编码器深度相匹配
            use_batchnorm=decoder_use_batchnorm,               #是否在解码器中使用批归一化,默认为True
            center=True if encoder_name.startswith("vgg") else False,    #如果编码器是 VGG 网络，解码器可能需要一个中心块
            attention_type=decoder_attention_type,             #选择是否使用注意力机制，默认为None
        )
        # 分割头
        self.seg_head = SegmentationHead(                      
            in_channels=decoder_channels[-1],                  #输入通道数，来自解码器的最后一层
            out_channels=classes,                              #输出通道数，即需要分割的类别数
            activation=activation,                             #激活函数类型
            kernel_size=3,                                     #卷积核大小
        )
        #Swin Transformer 解码器
        self.swin_decoder = SwinTransDecoder(classes, embed_dim, norm_layer, img_size, patch_size, depths, num_heads,
                                             window_size, qkv_bias, qk_scale, drop_rate, attn_drop_rate, use_checkpoint,
                                             ape, mlp_ratio, drop_path_rate, final_upsample, patches_resolution,
                                             encoder_channels)
        

        #Classification Head分类头
        self.cls_head = ClassificationHead(in_channels=encoder_channels[-1], classes=4) if cls else None
        # 嵌入头
        self.embed_head = EmbeddingHead(dim_in=encoder_channels[contrast_embed_index],
                                        embed_dim=contrast_embed_dim) if contrast_embed else None
        self._init_weights()

    def _init_weights(self):
        kaiming_normal_init_weight(self.cnn_decoder)
        kaiming_normal_init_weight(self.seg_head)
        if self.cls_head is not None:
            kaiming_normal_init_weight(self.cls_head)
        if self.embed_head is not None:
            kaiming_normal_init_weight(self.embed_head.embed)

    def forward(self, x, device):
        features = self.encoder(x)                        #从输入x中提取特征

        pdb.set_trace()
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(len(features))
        # print(features[0].shape) [16, 3, 224, 224]
        # print(features[1].shape) [16, 64, 112, 112]
        # print(features[2].shape) [16, 256, 56, 56]
        # print(features[3].shape) [16, 512, 28, 28]
        # print(features[4].shape) [16, 1024, 14, 14]
        # print(features[5].shape)
        # pdb.set_trace()

        seg = self.seg_head(self.cnn_decoder(*features))  #使用 CNN 解码器处理提取的特征，并通过分割头生成分割结果 seg
        seg_tf = self.swin_decoder(features, device)      #使用 Swin Transformer 解码器生成另一个分割结果 seg_tf

        embedding = self.embed_head(features[self.contrast_embed_index]) if self.embed_head else None   #如果embed_head存在提取特定层的特征并通过嵌入头生成嵌入向量 embedding
        cls = self.cls_head(features[-1]) if self.cls_head else None                                    #使用分类头生成分类结果
       
        pdb.set_trace()

        return {'seg': seg, 'seg_tf': seg_tf, 'cls': cls, 'embed': embedding}                           #返回值

    def inference(self, x, **kwargs):
        features = self.encoder(x)                        #提取x的特征
        seg = self.seg_head(self.cnn_decoder(*features))  #通过CNN解码器和分割头生成的结果
        preds = torch.argmax(seg, dim=1, keepdim=True).to(torch.float)      #通过 torch.argmax 计算出每个像素的分割类别（取概率最大的类别）。然后将输出转换为浮点数格式。
        return preds                                       #分割结果的预测张量
    #特征提取推理：这个方法主要用于特征提取，可以获取编码器输出的特征以及嵌入表示。
    def inference_features(self, x, **kwargs):
        features = self.encoder(x)
        embedding = self.embed_head(features[self.contrast_embed_index]) if self.embed_head else None    #通过嵌入头生成嵌入向量embedding
        return {'feats': features, 'embed': embedding}           #输出：一个包含两个键的字典，包括特征和嵌入表示
    # Swin Transformer 推理
    def inference_tf(self, x, device, **kwargs):
        features = self.encoder(x)
        seg_tf = self.swin_decoder(features, device)
        preds = torch.argmax(seg_tf, dim=1, keepdim=True).to(torch.float)
        return preds
