import torch

from optimizers.optimizer_empssl import LARSWrapper


def get_optimizer(optim_params, args):

    # simsiamの場合
    if args.ssl_method == "simsiam":
        optimizer = torch.optim.SGD(optim_params,
                                    args.optim_lr,
                                    momentum=args.optim_momentum,
                                    weight_decay=args.optim_weight_decay)
    
    # empsslの場合
    elif args.ssl_method == "empssl":
        optimizer = torch.optim.SGD(optim_params,
                                    args.optim_lr,
                                    momentum=args.optim_momentum,
                                    weight_decay=args.optim_weight_decay)
        optimizer = LARSWrapper(optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True)

    
    return optimizer