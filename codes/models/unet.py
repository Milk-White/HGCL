from ._base import BaseModel2D
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.unet.decoder import UnetDecoder
import torch.nn as nn   
import pdb


###########################
class FeatureSwap(nn.Module):
    def __init__(self, swap_ratio=0.5):
        super(FeatureSwap, self).__init__()
        self.swap_ratio = swap_ratio

    def forward(self, x1, x2):
        batch_size, channels, height, width = x1.size()
        swap_channels = int(channels * self.swap_ratio)  # 确保swap_channels正常计算
        
        if swap_channels == 0:
            # 如果通道数太少，可能导致没有交换
            return x1, x2
        
        # 执行交换操作
        x1_swap = x1[:, :swap_channels, :, :]
        x2_swap = x2[:, :swap_channels, :, :]
        
        x1[:, :swap_channels, :, :] = x2_swap
        x2[:, :swap_channels, :, :] = x1_swap
        






class Unet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",                          #编码器的名称
        encoder_depth: int = 5,                                  #编码器深度
        encoder_weights: Optional[str] = "imagenet",             #编码器权重
        decoder_use_batchnorm: bool = True,                      #解码器是否使用正则化
        decoder_channels: List[int] = (256, 128, 64, 32, 16),    #解码器的通道数
        decoder_attention_type: Optional[str] = None,            #解码器解码器使用的注意力类型
        in_channels: int = 3,                                    #图像输入通道
        classes: int = 1,                                        #最终输出类别数量
        activation: Optional[Union[str, callable]] = None,       #输出层的激活函数
        aux_params: Optional[dict] = None,                       #可选的辅助参数，用于构建分类头。
    ):
        super().__init__()
        #编码器
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        #解码器
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        #分割头
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        #分类头 
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)   #例如u-resnet34
        self.initialize()
    ########################################
        self.feature_swap = FeatureSwap(swap_ratio=0.5)  # 初始化特征交换模块




    def forward_features(self, x):
        features = self.encoder(x)

         # 在解码过程中进行特征交换
        for i in range(len(features) - 1):
            features[i], features[i+1] = self.feature_swap(features[i], features[i+1])

            x = self.decoder(*features)
            
            pdb.set_trace()
        return self.segmentation_head(x)
        

        return features


class UNet(BaseModel2D):

    def __init__(self,
                 encoder_name="resnet34",
                 encoder_depth=5,
                 encoder_weights="imagenet",
                 decoder_use_batchnorm: bool = True,              #解码器是否使用正则化
                 decoder_channels=(256, 128, 64, 32, 16),         #解码器的通道数
                 decoder_attention_type=None,
                 in_channels=3,
                 classes=1,
                 activation=None,
                 aux_params=None):
        super().__init__()

        self.segmentor = Unet(encoder_name=encoder_name,
                              encoder_depth=encoder_depth,
                              encoder_weights=encoder_weights,
                              decoder_use_batchnorm=decoder_use_batchnorm,
                              decoder_channels=decoder_channels,
                              decoder_attention_type=decoder_attention_type,
                              in_channels=in_channels,
                              classes=classes,
                              activation=activation,
                              aux_params=aux_params)
        self.num_classes = classes

    def forward(self, x):
        return {'seg': self.segmentor(x)}

    def inference_features(self, x, **kwargs):                    #输入X
        feats = self.segmentor.forward_features(x)                #使用self.segmentor的forward_features方法向前传播；feats 是从 forward_features 方法得到的特征
        return {'feats': feats}                                   #方法最后返回一个字典，其中包含一个键 'feats' 对应的值是 feats
