import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


def getmodel(arch):

    if arch == "resnet18-cifar":
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        return backbone, 512
    elif arch == 'resnet18-imagenet':
        backbone = resnet18()
        backbone.fc = nn.Identity()
        return backbone, 512
    else:
        assert False




class encoder_resnet18(nn.Module):
    def __init__(self, z_dim=1024, hidden_dim=4096, norm_p=2, arch=''):
        
        super().__init__()

        # エンコーダ（特徴抽出器部分）の作成
        backbone, feature_dim = getmodel(arch)
        self.backbone = backbone

        # 
        self.norm_p = norm_p

        # MLP1の作成
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                         nn.BatchNorm1d(hidden_dim),
                                         nn.ReLU()
                                        )
        
        # MLP2の作成
        self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.Linear(hidden_dim, z_dim)
                                        )
        
    def forward(self, x, is_test=False, linear=False, knn=False):

        # エンコーダにデータを入力し出力を獲得
        feature1 = self.backbone(x)

        # MLP1にエンコーダの出力を入力して出力を獲得
        feature2 = self.pre_feature(feature1)

        # MLP2にMLP1の出力を入力して出力を獲得
        z = F.normalize(self.projection(feature2), p=self.norm_p)

        if is_test:
            assert False
        else:
            return z, feature1, feature2
        


