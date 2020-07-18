import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt

from nets import OrdinaryEncoder, DeconvDecoder
from EndoscopicDataset_utils import EndoscopicDataset, make_datapath_list, DataTransform


class AutoEncoder:
    def __init__(self):
        self.encoder = OrdinaryEncoder(input_nc=3)
        self.decoder = DeconvDecoder(input_nc=1)

    def train(self, dataset_rootpath, num_epoch=100, lr=1e-5, mini_batch_size=8, color_mean=(0.485, 0.456, 0.406),
              color_std=(0.229, 0.224, 0.225), save_iter_freq=100, Is_continue=False):

        # GPUが使えるかを確認
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス : ", device)

        # 損失関数,最適化手法の定義
        criterion = nn.MSELoss()
        beta1, beta2 = 0.0, 0.9
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta1, beta2))
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(beta1, beta2))

        # 重みをロード，ネットワークをGPUに
        if Is_continue:
            self.encoder.load_state_dict(torch.load(r'D:\Deep_Learning\AutoEncoder\model\encoder_model.pth'))
            self.decoder.load_state_dict(torch.load(r'D:\Deep_Learning\AutoEncoder\model\decoder_model.pth'))
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder.train()
        self.decoder.train()

        # ネットワークが固定で高速化
        torch.backends.cudnn.benchmark = True

        # データローダの作成
        train_img_list, val_img_list = make_datapath_list(dataset_rootpath)
        train_dataset = EndoscopicDataset(train_img_list, phase="train",
                                          transform=DataTransform(color_mean=color_mean, color_std=color_std))
        val_dataset = EndoscopicDataset(val_img_list, phase="val",
                                        transform=DataTransform(color_mean=color_mean, color_std=color_std))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
        val_daraloader = torch.utils.data.DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=True)

        # epochのループ
        iteration = 0
        for epoch in range(num_epoch):

            # 開始時刻を保存
            t_epoch_start = time.time()
            epoch_loss = 0.0

            print('-------')
            print('Epoch {} / {}'.format(epoch, num_epoch))
            print('-------')
            print('(train)')

            # データローダからminibatchずつ取り出すループ
            for esophagus_img in train_dataloader:

                # ------
                # 1. AutoEncoderの学習
                # ------

                # ミニバッチサイズが1だとバッチノーマライゼーションでエラーになる
                if esophagus_img.size()[0] == 1:
                    continue

                # 画像の再構成
                esophagus_img = esophagus_img.to(device)
                compressed_features = self.encoder(esophagus_img)
                reconstructed_esophagus_img = self.decoder(compressed_features)

                # 損失値の計算，バックプロパゲーション
                loss = criterion(reconstructed_esophagus_img, esophagus_img)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                # -----
                # 2. 記録
                # -----
                epoch_loss += loss.item()

                if (iteration % save_iter_freq) == 0:
                    torch.save(self.encoder.state_dict(),
                               r'D:\Deep_Learning\AutoEncoder\model\encoder_model.pth')
                    torch.save(self.decoder.state_dict(),
                               r'D:\Deep_Learning\AutoEncoder\model\decoder_model.pth')
                    input_for_save = esophagus_img[0].cpu().numpy().transpose((1, 2, 0))
                    input_for_save = np.clip(input_for_save, 0.0, 1.0)
                    plt.imsave(r'D:\Deep_Learning\AutoEncoder\save_img\{}_input.jpg'.format(iteration), input_for_save)
                    output_for_save = reconstructed_esophagus_img[0].cpu().detach().numpy().transpose((1, 2, 0))
                    output_for_save = np.clip(output_for_save, 0.0, 1.0)
                    plt.imsave(r'D:\Deep_Learning\AutoEncoder\save_img\{}_output.jpg'.format(iteration), output_for_save)

                iteration += 1

            t_epoch_finish = time.time()
            print('------')
            print('epoch {}'.format(epoch))
            print('loss : {}  || timer {:.4f} sec'.format(epoch_loss / mini_batch_size, time.time() - t_epoch_start))

            t_epoch_start = time.time()
