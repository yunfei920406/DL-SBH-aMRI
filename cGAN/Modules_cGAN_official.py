import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class ConditionalGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, embed_dim=16,num_classes = 7):
        super(ConditionalGenerator, self).__init__()

        # Embedding layer to map class labels to a feature vector
        self.embedding = nn.Embedding(num_classes, embed_dim)


        self.down1 = UNetDown(in_channels+embed_dim, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, target_label):
        # U-Net generator with skip connections from encoder to decoder

        cond = self.embedding(target_label)  # [B, embed_dim]
        cond = cond.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))  # [B, embed_dim, 256, 256]

        # 拼接条件信息与输入图像
        x_cond = torch.cat([x, cond], dim=1)  # [B, C + embed_dim, 256, 256]

        # Downsampling path
        d1 = self.down1(x_cond)

        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


import torch
import torch.nn as nn

class Discriminator(nn.Module):
    '''Discriminator with PatchGAN and Category Conditioning'''

    def __init__(self, conv_dim, layer_num, num_classes = 6):
        super(Discriminator, self).__init__()

        # Embedding layer for category labels (assuming one-hot encoded labels)
        self.embedding = nn.Embedding(num_classes, conv_dim)  # 将类别标签转换为嵌入向量

        layers = []

        # input layer: concatenating image and condition (category label)
        layers.append(nn.Conv2d(1 + conv_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim

        # hidden layers
        for i in range(1, layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(current_dim * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2

        self.model = nn.Sequential(*layers)

        # output layer
        self.conv_src = nn.Sequential(
            nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output the probability that the image is real
        )

    def forward(self, x, c):
        # Convert categorical label c into an embedding
        c_embedded = self.embedding(c).unsqueeze(2).unsqueeze(3)  # 将类别标签转换为嵌入向量并调整为与图像一致的形状

        # Concatenate the image x and the category embedding along the channel dimension
        x = torch.cat([x, c_embedded.expand(-1, -1, x.size(2), x.size(3))], dim=1)


        # Pass through the model
        x = self.model(x)

        # Output layer: determine if the image is real or fake (patch GAN)
        out_src = self.conv_src(x)
        return out_src
