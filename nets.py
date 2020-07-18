import torch
import torch.nn as nn


class OrdinaryEncoder(nn.Module):
    """VGGライクなエンコーダ

    Parameters:
        input_nc(int) : 入力画像のチャンネル
        ngf(int) : コンボリューションチャンネルの倍数
        use_dropout(bool) : ドロップアウトをするかどうか
    """

    def __init__(self, input_nc, ngf=64):
        super(OrdinaryEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.Conv2d(ngf * 8, 3, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.last(out)

        return out


class DeconvDecoder(nn.Module):
    '''一般的なDeconvメインのジェネレータ
     Parameters:
        input_nc(int) : 入力画像のチャンネル
        ngf(int) : コンボリューションチャンネルの倍数
        use_dropout(bool) : ドロップアウトをするかどうか
    '''

    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DeconvDecoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(input_nc, ngf * 8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(ngf, 3, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out
