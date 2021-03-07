#%%
from functools import partial
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.densenet import densenet201
from timm.models.inception_v4 import inception_v4
from timm.models.resnest import resnest50d_4s2x40d, resnest50d_1s4x24d
from timm.models.senet import legacy_seresnet101, legacy_seresnext50_32x4d
from timm.models.efficientnet import tf_efficientnet_b0_ns
                          
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from Utils import do_mixup, interpolate, pad_framewise_output, init_layer, init_bn

encoder_params = {
    "resnest50d" : {
        "features" : 2048,
        "init_op"  : partial(timm.models.resnest50d, pretrained=True, in_chans=1)
    },
    "inception_v4" : {
        "features" : 1536,
        "init_op" : partial(timm.models.inception_v4, pretrained=True, in_chans=1)
    },
    "densenet201" : {
        "features": 1920,
        "init_op": partial(timm.models.densenet201, pretrained=True, in_chans=1)
    },
    "resnest50d_4s2x40d" : {
        "features" : 2048,
        "init_op" : partial(timm.models.resnest50d_4s2x40d, pretrained=True, in_chans=1)
    },
    "resnest50d_1s4x24d" : {
        "features" : 2048,
        "init_op" : partial(timm.models.resnest50d_1s4x24d, pretrained=True, in_chans=1)
    }}
class AttentionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear", temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.conv_attention = nn.Conv1d(in_channels=in_features,
                                        out_channels=out_features,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True)
        self.conv_classes = nn.Conv1d(in_channels=in_features,
                                      out_channels=out_features,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True)

        self.batch_norm_attention = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv_attention)
        init_layer(self.conv_classes)
        init_bn(self.batch_norm_attention)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        #norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        norm_att = torch.softmax(torch.tanh(self.conv_attention(x)), dim=-1)
        classes = self.nonlinear_transform(self.conv_classes(x))
        x = torch.sum(norm_att * classes, dim=2)
        return x, norm_att, classes

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class AudioClassifier(nn.Module):
    def __init__(self, encoder, sample_rate, window_size, hop_size, 
                 mel_bins, fmin, fmax, classes_num):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        #self.interpolate_ratio = 29 # Downsampled ratio
        self.interpolate_ratio = 29

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
                                                 win_length=window_size, window=window, 
                                                 center=center, pad_mode=pad_mode, 
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, 
                                                 ref=ref, amin=amin, top_db=top_db, 
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
                                               freq_drop_width=8, freq_stripes_num=2)
        
        self.batch_norm = nn.BatchNorm2d(mel_bins)
        self.encoder = encoder_params[encoder]["init_op"]()
        
        #self.encoder.last_linear = Linear(encoder_params[encoder]['features'], 2048, bias=True)
        #self.encoder.classifier = Linear(2048, encoder_params[encoder]['features'], bias=True)
        #self.fc = Linear(encoder_params[encoder]['features'], 2048, bias=True)
        
        #self.encoder.fc = nn.Linear(2048, 2048)
        self.dropout = Dropout(0.3)
        self.att_head = AttentionHead(1000, classes_num, activation='sigmoid')
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_weight()

    
    def init_weight(self):
        init_bn(self.batch_norm)
        init_layer(self.encoder.fc)
        self.att_head.init_weights()


    def forward(self, input, spec_aug=False, mixup_lambda=None):
        #print(input.type())                                                     # Input : (16, 144000)
        x = self.spectrogram_extractor(input.float())                            # Output : (batch_size, 1, time_steps, n_fft + 1) : (16, 1, 696, 513)
        x = self.logmel_extractor(x)                                             # Output : (batch_size, 1, time_steps, mel_bins)     : (16, 1 , 696, 128)
        frames_num = x.shape[2]
        
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = x.transpose(1, 3)
        x = self.batch_norm(x)
        x = x.transpose(1, 3)
                                                                                  # (16, 1, 2087, 128)
        x = self.encoder.forward_features(x)                                     # output : (batch_size, n_features, 66, 4)
        # Aggregate in time axis
        x = torch.mean(x, dim=3)                                                 # (16, 2048, 22) : (batch_size, n_features, _)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2                                                              # (16, 2048, 22)

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)                                                    # (batch_size, 22, n_features)
        #x = self.encoder.classifier(x)                                                    # (16, 22, 2048) : (batch_size, time, n_features)
        x = F.relu_(self.encoder.fc(x))                                          # (16, 22, 2048)
        x = x.transpose(1, 2)                                                    # (16, 2048, 22)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_head(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        #print("clipwise_output.size : {}".format(clipwise_output.size()))        #(16, 24)     : (batch_sizes, n_features)
        #print("norm_att.size : {}".format(norm_att.size()))                      #(16, 24, 22) : (batch_sizes, n_features, time)
        #print("segmentwise_output.size : {}".format(segmentwise_output.size()))  #(16, 24, 22) : (batch_sizes, n_features, time)
        
        #Upscale back to original size
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)                    # (16,696, 24) : (batch_sizes x time x num_classes)

        framewise_output = pad_framewise_output(framewise_output, frames_num)     # (16,696, 24) : (batch_sizes x time x num_classes)
        output_dict = {
           'framewise_output': framewise_output,
            'clipwise_output': clipwise_output
        }

        return output_dict

#def get_model(is_mean_teacher=False)

'''


Reference 
SED Model
- https://www.kaggle.com/gopidurgaprasad/rfcx-sed-model-stater

Audio Detection
- https://www.kaggle.com/gopidurgaprasad/rfcs-audio-detection-pytorch-stater
'''

# %%
