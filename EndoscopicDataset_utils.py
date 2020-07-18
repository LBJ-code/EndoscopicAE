import os.path as osp
from PIL import Image
from glob import glob
from data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Normalize_Tensor

import torch.utils.data as data


def make_datapath_list(rootpath):
    '''
    学習，検証の画像データへのファイルパスリストを作成する．
    :param rootpath:

    :return train_img_list, test_img_list:
    '''

    subset_list = ['seq_{}'.format(i) for i in range(7)]
    train_img_list = list()
    for subset in subset_list:
        train_img_list += glob(osp.join(rootpath, subset, '*.png'))

    test_img_list = list()
    test_img_list += glob(osp.join(rootpath, 'test', '*.png'))

    return train_img_list, test_img_list


class DataTransform:
    '''
    画像の前処理クラス．訓練時と検証時で挙動が異なる．
    訓練時にはデータオギュメンテーションする．

    Attributes
    ----------
    color_mean : (R, G, B)
        各色チャンネルの平均値
    color_std : (R, G, B)
        各色チャンネルの標準偏差
    '''

    def __init__(self, color_mean, color_std):
        self.data_transform = {
            'train' : Compose([
                RandomMirror(), # 反転
                Normalize_Tensor(color_mean, color_std) # 色情報の標準化とテンソル化
            ]),
            'val' : Compose([
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img):
        '''
        :param phase: 'train' or 'val'
        :return: トランスフォーム後のテンソル
        '''

        return self.data_transform[phase](img)


class EndoscopicDataset(data.Dataset):
    """
    食道のデータセットを作成するクラス．Pytorchのdatasetクラスを継承．

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    phase : 'train' or 'test
        学習か訓練かを設定する
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, phase, transform):
        self.img_list = img_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        ''' 前処理した画像のTensor形式のデータを取得'''
        img = self.pull_item(index)
        return img

    def pull_item(self, index):
        '''画像のTensor形式のデータを取得する'''

        #1. 画像読込
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path) # [高さ][幅][色RGB]

        #2. 前処理を実施
        img = self.transform(self.phase, img)

        return img

'''デバック用に使いまわし可
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
train_img_list, _ = make_datapath_list(r"D:\Deep_Learning\MonoDepth2\esophagus\imgs")
train_dataset = EndoscopicDataset(train_img_list, "train",
                                  transform=DataTransform(color_mean, color_std))

import numpy as np
import matplotlib.pyplot as plt

index = 0
imges = train_dataset.__getitem__(index)
img_val = imges
img_val = img_val.numpy().transpose((1, 2, 0))
img_val = np.clip(img_val, 0.0, 1.0)
plt.imsave('dataset.jpg', img_val)
plt.imshow(img_val)
plt.show()
'''