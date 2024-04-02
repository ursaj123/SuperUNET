import numpy as np
import os
import torch 
import torch.nn as nn

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        # this is the skip connection part
        residual = x # shape (B, in_C, H, W)
        residual = self.skip_conv(residual) # shape (B, out_C, H, W)

        # this is the main part
        out = self.conv1(x) # shape (B, out_C, H, W)
        out = self.bn1(out) # shape (B, out_C, H, W)
        out = self.relu(out) # shape (B, out_C, H, W)
        out = self.conv2(out)  # shape (B, out_C, H, W)
        out += residual # shape (B, out_C, H, W)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class squeeze_and_excitation(nn.Module):
    def __init__(self, in_channels, r=2):
        super(squeeze_and_excitation, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//r, in_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # shape is (B, in_C, H, W)
        orig = x.clone() # shape is (B, in_C, H, W)
        out = self.global_pool(x) # shape is (B, in_C, 1, 1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out) # shape is (B, in_C//r)
        out = self.relu(out) # shape is (B, in_C//r)
        out = self.fc2(out) # shape is (B, in_C)
        out = self.sigmoid(out) # shape is (B, in_C)
        out = out.view(out.size(0), out.size(1), 1, 1) # shape is (B, in_C, 1, 1)
        out = orig * out # shape is (B, in_C, H, W)
        return out


class fusion_upsampling(nn.Module):
    def __init__(self, enc_C, dec_C, r=2):
        super(fusion_upsampling, self).__init__()
        # assert enc_C == dec_C//2, "enc_C should be equal to dec_C//2"
        # se modules for encoder and decoder
        self.enc_se = squeeze_and_excitation(enc_C, r)
        self.dec_se = squeeze_and_excitation(dec_C, r)

        # upsampling part
        self.conv_tr = nn.ConvTranspose2d(dec_C, enc_C, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(enc_C)
        self.relu = nn.ReLU()


    def forward(self, enc_map, dec_map):
        # enc_map shape is (B, enc_C, 2*H, 2*W)
        # dec_map shape is (B, 2*enc_C, H, W) (dec_C = 2*enc_C)
        # se part
        enc_map = self.enc_se(enc_map) # shape is (B, enc_C, 2*H, 2*W)
        dec_map = self.dec_se(dec_map) # shape is (B, 2*enc_C, H, W)

        # upsampling part
        dec_map = self.conv_tr(dec_map) # shape is (B, enc_C, 2*H, 2*W)
        dec_map = self.bn(dec_map) # shape is (B, enc_C, 2*H, 2*W)
        dec_map = self.relu(dec_map) # shape is (B, enc_C, 2*H, 2*W)

        # fusion part
        out = torch.concat((enc_map, dec_map), dim=1) # shape is (B, 2*enc_C, 2*H, 2*W)
        return out
    


class dynamic_receptive_field_module(nn.Module):
    def __init__(self, in_C):
        super(dynamic_receptive_field_module, self).__init__()
        self.conv1x1 = nn.Conv2d(in_C, in_C, kernel_size=1, stride=1, padding='same')
        self.conv3x3 = nn.Conv2d(in_C, in_C, kernel_size=3, stride=1, padding='same')
        self.conv5x5 = nn.Conv2d(in_C, in_C, kernel_size=5, stride=1, padding='same')

        self.conv1x1_dil = nn.Conv2d(in_C, in_C, kernel_size=1, stride=1, padding='same', dilation=1)
        self.conv3x3_dil = nn.Conv2d(in_C, in_C, kernel_size=3, stride=1, padding='same', dilation=3)
        self.conv5x5_dil = nn.Conv2d(in_C, in_C, kernel_size=5, stride=1, padding='same', dilation=5)
        

        # self.final_conv = nn.Conv2d(3*in_C, in_C, kernel_size=3, stride=1, padding='same')
        # self.bn = nn.BatchNorm2d(in_C)

        self.last_conv1 = nn.Conv2d(3*in_C, in_C, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(in_C)
        self.relu = nn.ReLU()

        self.last_conv2 = nn.Conv2d(2*in_C, in_C, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(in_C)


    def forward(self, x):
        # x shape is (B, in_C, H, W)
        orig = x.clone()
        # 1x1 conv part
        out_1 = self.conv1x1(x) # shape is (B, in_C, H, W)
        out_1 = self.conv1x1_dil(out_1) # shape is (B, in_C, H, W)
        # 3x3 conv part
        out_3 = self.conv3x3(x)
        out_3 = self.conv3x3_dil(out_3)
        # 5x5 conv part
        out_5 = self.conv5x5(x)
        out_5 = self.conv5x5_dil(out_5)
        # concat all the outputs
        out = torch.cat((out_1, out_3, out_5), dim=1) # shape is (B, 3*in_C, H, W)
        out = self.last_conv1(out) # shape is (B, in_C, H, W)
        out = self.bn1(out) # shape is (B, in_C, H, W)
        out = self.relu(out) # shape is (B, in_C, H, W)

        # skip connection
        out = torch.cat((orig, out), dim=1) # shape is (B, 2*in_C, H, W)

        # last convolution to get the half number of channels
        out = self.last_conv2(out) # shape is (B, in_C, H, W)
        out = self.bn2(out) # shape is (B, in_C, H, W)
        out = self.relu(out) # shape is (B, in_C, H, W)




        # # skip connection
        # out = self.final_conv(out) # shape is (B, in_C, H, W)
        # out = self.bn(out) # shape is (B, in_C, H, W)
        # out = self.relu(out) # shape is (B, in_C, H, W)

        # # skip connection
        # # out = torch.cat((orig, out), dim=1) # shape is (B, 2*in_C, H, W)
        # out+= orig # shape is (B, in_C, H, W)

        return out
    
class downsample(nn.Module):
    def __init__(self, channels = [3, 8, 16, 32, 64, 128]):
        super(downsample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.channels = channels
        self.blocks = nn.ModuleList([])
        for i in range(5):
            self.blocks.append(resblock(self.channels[i], self.channels[i+1]))
        self.skips = []
        pass

    def forward(self, x):
        # x shape is (B, 3, H, W)
        for i in range(5):
            x = self.blocks[i](x) # (B, channels[i+1], H, W)
            self.skips.append(x) # (B, channels[i+1], H, W)
            if i < 4:
                x = self.pool(x) # (B, channels[i+1], H/2, W/2)
        return self.skips
    

class upsample(nn.Module):
    def __init__(self, channels = [128, 64, 32, 16, 8, 3]):
        super(upsample, self).__init__()
        self.channels = channels
        self.frm_blocks = nn.ModuleList([])
        self.res_blocks = nn.ModuleList([])
        self.drfm_blocks = nn.ModuleList([])
        for i in range(4):
            self.frm_blocks.append(fusion_upsampling(self.channels[i+1], self.channels[i]))
            self.res_blocks.append(resblock(self.channels[i], self.channels[i+1]))
            self.drfm_blocks.append(dynamic_receptive_field_module(self.channels[i+1]))
        


    def forward(self, skips):
        dec_map = skips[-1] # (B, 128, 3, 3)
        for i in range(4):
            enc_map = skips[-i-2] # (B, channels[i+1], 2**i*H, 2**i*W)
            dec_map = self.frm_blocks[i](enc_map, dec_map) # (B, channels[i], 2**i*H, 2**i*W)
            dec_map = self.res_blocks[i](dec_map) # (B, channels[i+1], 2**i*H, 2**i*W)
            dec_map = self.drfm_blocks[i](dec_map) # (B, channels[i+1], 2**i*H, 2**i*W)

    
        return dec_map


class SuperUNET(nn.Module):
    def __init__(self, num_classes = 2):
        super(SuperUNET, self).__init__()
        self.down = downsample()
        self.up = upsample()
        self.conv1x1 = nn.Conv2d(8, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # x shape is (B, 3, H, W)
        skips = self.down(x) # all the skip connections
        out = self.up(skips)
        out = self.conv1x1(out)
        return out



