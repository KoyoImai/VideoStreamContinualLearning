## 動画データを用いてデータストリームによって自己教師あり学習を行うためのコード
## 目的は，自己教師ありオンライン継続学習によって，クラス分類や物体検出など多様なタスクで高精度なモデルを学習すること


import random
import wandb
import os
import numpy as np
import torch
import torch.nn as nn
import sys



from utils import parse_option
from optimizers.optimizer import get_optimizer




def main():

    
    """  コマンドライン引数の処理  """
    args = parse_option()
    print("args : ", args)
    

    """  wandbの開始  """
    #wandb.init(project=f"{args.log_name}")


    """  ディレクトリの作成  """
    # 学習過程のパラメータ保存用のディレクトリ
    logdir = os.path.join(args.main_dir, args.log_dir, args.log_name)
    os.makedirs(logdir, exist_ok=True)


    """  シード値の固定  """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    """  モデルの定義 & データ拡張の定義  """
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.arch == 'resnet18':
            if args.ssl_method == 'empssl':
                
                # empssl用のモデルを作成
                from models.model_empssl import encoder_resnet18
                model = encoder_resnet18(arch='resnet18-cifar')
                
                # empssl用のデータ拡張を定義
                from augmentation.empssl_aug import ContrastiveLearningViewGenerator
                augmentation = ContrastiveLearningViewGenerator(num_patch=args.num_patch, dataset=args.dataset)

            elif args.ssl_method == 'simsiam':
                
                # simsiam用のモデルを作成
                from models.model_simsiam import encoder_resnet18, SimSiam
                model = encoder_resnet18(arch='resnet18-cifar')

                # SimSiam公式実装
                #from torchvision.models import resnet18
                #model = SimSiam(resnet18)

                # simsiam用のデータ拡張を定義
                from augmentation.simsiam_aug import return_augmentation
                augmentation = return_augmentation()
            
            elif args.ssl_method == 'moco':

                assert False

        else:
            assert False
    else:
        assert False

    # GPUが使用可能であればmodelをGPUに配置
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    #print("model : ", model)

    
    """  データセットの作成  """
    ## CIFAR10の場合
    if args.dataset == "cifar10":
        trainfname = os.path.join(args.data_dir, 'CIFAR10')
        if args.train_samples_per_cls is None:
            train_samples_per_cls = [5000]
        else:
            train_samples_per_cls = args.train_samples_per_cls
        data_insize = 32
        
    ## CIFAR100の場合
    elif args.dataset == 'cifar100':
        trainfname = os.path.join(args.data_dir, 'CIFAR100')
        
        # empsslの場合（マルチクロップの場合）
        if args.ssl_method == 'empssl':
            from datasets.dataset_empssl_cifar100 import Cifar100Dataset
            train_dataset = Cifar100Dataset(
                trainfname,
                transforms=augmentation,
                train=True,
                download=True,
                seed=args.seed,
                senario=args.senario
            )
        # empssl以外の場合（2クロップの場合）
        else:
            from datasets.dataset_cifar100 import Cifar100Dataset
            train_dataset = Cifar100Dataset(
                trainfname,
                transforms=augmentation,
                train=True,
                download=True,
                seed=args.seed,
                senario=args.senario
            )
    
    ## 上記以外場合
    else:
        assert False

    """  train_loaderの作成  """
    ## この中でまとめてbatch_samplerとsamplerを作成
    if args.ssl_method == 'empssl':
        from trainloaders.trainloader_empssl import get_trainloader
        train_loader = get_trainloader(train_dataset, args)
    elif args.ssl_method == 'simsiam':
        from trainloaders.trainloader_empssl import get_trainloader
        train_loader = get_trainloader(train_dataset, args)


    #print(train_loader)
    #print("len(train_dataset) : ", len(train_dataset))
    #x = train_dataset.advance_batches_seen()
    #y = train_dataset.advance_batches_seen()
    #print("x : ", x)
    #print("y : ", y)




    """  optimizerの定義  """
    
    ## 最適化するパラメータ
    # SimSiamなど学習率の固定が必要な場合の処理
    if args.ssl_method == "simsiam":

        # predictorの学習率を固定する場合
        #if args.fix_pred_lr:
        if True:
            from models.model_simsiam import fix_pred_lr 
            optim_params = fix_pred_lr(model)
        
        # predictorの学習率を固定しない場合
        else:
            optim_params = model.parameters()
    else:
        optim_params = model.parameters()
    
    ## 最適化手法の決定
    optimizer = get_optimizer(optim_params, args)


    """  損失関数の定義  """
    if args.ssl_method == "simsiam":
        criterion = nn.CosineSimilarity(dim=1)
        if torch.cuda.is_available():
            criterion = criterion.cuda()
    elif args.ssl_method == "empssl":
        from losses.loss_empssl import Similarity_Loss, TotalCodingRate
        contrastive_loss = Similarity_Loss()
        criterion = TotalCodingRate(eps=args.empssl_eps)

        if torch.cuda.is_available():
            contrastive_loss = contrastive_loss.cuda()
            criterion = criterion.cuda()


    """  学習を実行  """
    for epoch in range(0, args.epoch):
        print("Train Epoch {}".format(epoch))
        sys.stdout.flush()

        # simsiamの場合
        if args.ssl_method == "simsiam":
            from train.train_simsiam import train
            train(train_loader=train_loader,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch,
                  args=args)
        
        # empsslの場合
        elif args.ssl_method == "empssl":
            from train.train_empssl import train
            train(train_loader=train_loader,
                  model=model,
                  criterion=criterion,
                  contrastive_loss=contrastive_loss,
                  optimizer=optimizer,
                  epoch=epoch,
                  args=args)




    """  modelの動作確認  """
    #data = torch.randn([62, 3, 32, 32]).cuda()
    #z, feat1, feat2 = model(data)
    #print("z.shape : ", z.shape)









if __name__ == '__main__':
    main()