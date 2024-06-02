import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import numpy as np


from datasets.dataset_base_cifar100 import Cifar100BaseDatset




class Cifar100Dataset(Cifar100BaseDatset):
    
    def __init__(self, root, transforms, train, download, seed, senario):

        super().__init__(root, transforms, train, download, seed, senario)


    def __getitem__(self, index):

        image, label = self.dataset[index]

        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        
        if transform is not None:
            im = transform(image)

        meta['transid'] = i
        meta['index'] = index
        meta['label'] = label

        out = {
            'input': im,
            'meta': meta,
        }

        return out
    


