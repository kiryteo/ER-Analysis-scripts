import torch
import torch.nn as nn
import torchvision.models


class conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(conv, self).__init__()
        if(activation):
            self.layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
          )
        else:
          self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1)
             )

    def forward(self, x):
        x = self.layer(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.layer = nn.Sequential(
            conv(in_ch, out_ch),
            nn.MaxPool2d(2)
            )

    def forward(self, x):
        x = self.layer(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = conv(in_ch, out_ch)

    def forward(self, x):
        y = self.up(x)
        y = self.conv(y)
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input_conv = conv(1, 16)
        self.d1 = down(16, 64)
        self.d2 = down(64, 256)
        self.d3 = down(256, 512)

        self.u1 = up(1024, 512)
        self.u2 = up(512, 256)
        self.u3 = up(256, 64)
        self.u4 = up(64, 16)
        self.output_conv = conv(16, 1, False)

    def forward(self, input, pgt):
        y_ip = self.input_conv(input)
        y1_ip = self.d1(y_ip)
        y2_ip = self.d2(y1_ip)
        y3_ip = self.d3(y2_ip)

        y_pgt = self.input_conv(pgt)
        y1_pgt = self.d1(y_pgt)
        y2_pgt = self.d2(y1_pgt)
        y3_pgt = self.d3(y2_pgt)

        z = torch.cat([y3_ip, y3_pgt], dim=1)
        # print(z.shape)

        y4 = self.u1(z)
        y5 = self.u2(y4)
        y6 = self.u3(y5)
        y7 = self.u4(y6)

        output = self.output_conv(y7)

        return output

from torchsummary import summary
net = Net().cuda()


print(summary(net, [(1,128,128), (1, 128, 128)]))















# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )
#
#
# class ResNetUNet(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()
#
#         self.base_model = torchvision.models.resnet18(pretrained=True)
#         self.base_layers = list(self.base_model.children())
#
#         self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convrelu(64, 64, 1, 0)
#         self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convrelu(128, 128, 1, 0)
#         self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convrelu(256, 256, 1, 0)
#         self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convrelu(512, 512, 1, 0)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#         self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
#         self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
#         self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
#         self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
#
#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
#
#         self.conv_last = nn.Conv2d(64, n_class, 1)
#
#     def forward(self, input):
#         x_original = self.conv_original_size0(input)
#         x_original = self.conv_original_size1(x_original)
#
#         layer0 = self.layer0(input)
#         layer1 = self.layer1(layer0)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)
#
#         layer4 = self.layer4_1x1(layer4)
#         x = self.upsample(layer4)
#         layer3 = self.layer3_1x1(layer3)
#         x = torch.cat([x, layer3], dim=1)
#         x = self.conv_up3(x)
#
#         x = self.upsample(x)
#         layer2 = self.layer2_1x1(layer2)
#         x = torch.cat([x, layer2], dim=1)
#         x = self.conv_up2(x)
#
#         x = self.upsample(x)
#         layer1 = self.layer1_1x1(layer1)
#         x = torch.cat([x, layer1], dim=1)
#         x = self.conv_up1(x)
#
#         x = self.upsample(x)
#         layer0 = self.layer0_1x1(layer0)
#         x = torch.cat([x, layer0], dim=1)
#         x = self.conv_up0(x)
#
#         x = self.upsample(x)
#         x = torch.cat([x, x_original], dim=1)
#         x = self.conv_original_size2(x)
#
#         out = self.conv_last(x)
#
#         return out