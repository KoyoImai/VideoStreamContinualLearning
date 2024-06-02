import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps
import numpy as np



class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
        

class ContrastiveLearningViewGenerator(object):

    def __init__(self, num_patch=20, dataset=None):

        # クロップ数の設定
        self.num_patch = num_patch

        # データセットによってデータサイズを決定
        if dataset == 'cifar10' or dataset == 'cifar100':
            self.data_insize = 32
        elif dataset == "imagenet100":
            self.data_insize = 224
        else:
            assert False

    
    def __call__(self, x):

        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.data_insize, scale=(0.25, 0.25), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            normalize
        ])

        # データ拡張を実行
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]

        return augmented_x


