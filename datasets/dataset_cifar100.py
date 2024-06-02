import torch.utils.data as data
import torchvision
import numpy as np

from datasets.dataset_base_cifar100 import Cifar100BaseDatset





class Cifar100Dataset(Cifar100BaseDatset):
    
    def __init__(self, root, transforms, train, download, seed, senario):

        super().__init__(root, transforms, train, download, seed, senario)

    
    def __getitem__(self, index):

        # データとラベルをself.datasetから取り出す
        image, label = self.dataset[index]
        image2 = image.copy()

        # データの情報を保管するための辞書
        meta = {}

        # データに加えるデータ拡張を設定
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        
        # データにデータ拡張を加える
        if transform is not None:
            im1 = transform(image)
            im2 = transform(image2)
        
        # データに関する情報を辞書に格納
        meta['transid'] = i
        meta['index'] = index
        meta['label'] = label

        out = {
            'input1': im1,
            'input2': im2,
            'meta': meta,
        }

        return out



