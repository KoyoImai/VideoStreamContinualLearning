import torch.utils.data as data
import numpy as np
import torchvision


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.'
    Code copied from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class Cifar100BaseDatset(data.Dataset):

    def __init__(self, root, transforms, train, download, seed, senario):

        data.Dataset.__init__(self)

        # 初期化
        self.root = root
        self.train = train
        self.download = download
        self.num_batches_seen = 0

        # データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)

        # 変更前のラベルを獲得
        self.ooriginal_labels = self.dataset.targets

        # ラベルを変更
        if senario == "iid":
            print("no change labels")
        elif senario == "":
            assert False
        else:
            self.dataset.targets = sparse2coarse(self.dataset.targets)

        # データ拡張
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms

    ## データセットのサイズを返す
    def __len__(self):
        return len(self.dataset)
    
    ## 確認したバッチ数（イタレーション数）を更新して返す
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen