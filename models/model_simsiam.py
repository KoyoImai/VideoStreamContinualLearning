import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50


def fix_pred_lr(model=None):

    optim_params = [{'params': model.module.backbone.parameters(), 'fix_lr': False},
                    {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    
    return optim_params


def getmodel(arch):

    if arch == "resnet18-cifar":

        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        
        return backbone, 512

    elif arch == 'resnet18-imagenet':

        backbone = resnet18(zero_init_residual=True)
        backbone.fc = nn.Identity()

        return backbone, 512



class encoder_resnet18(nn.Module):

    def __init__(self, dim=2048, pred_dim=512, arch=None):

        super(encoder_resnet18, self).__init__()

        ## encoder（特徴抽出器）の作成
        backbone, feature_dim = getmodel(arch)
        self.backbone = backbone

        # projectorの作成
        self.projector = nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False),
                                       nn.BatchNorm1d(feature_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(feature_dim, feature_dim, bias=False),
                                       nn.BatchNorm1d(feature_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(feature_dim, dim, bias=True),
                                       nn.BatchNorm1d(dim, affine=False)
        )
        #self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # predictorの作成
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(pred_dim, dim))
        

    def forward(self, x1, x2=None):

        # encoderの出力
        feature1 = self.backbone(x1)

        # projectorの出力
        z1 = self.projector(feature1)
        
        # predictorの出力
        p1 = self.predictor(z1)

        if x2 is None:
            return p1, z1.detach(), feature1
        

        # encoderの出力
        feature2 = self.backbone(x2)

        # projectionの出力
        z2 = self.projector(feature2)

        # predictorの出力
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach(),
        



# SimSiamのpytorch公式実装
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer