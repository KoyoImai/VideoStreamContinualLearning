import torch

from trainloaders.samplers.iid_sampler_cifar import IidSampler
from trainloaders.samplers.seq_sampler_cifar import SeqSampler

from trainloaders.batch_samplers.random_buffer import RandomBufferBatchSampler
from trainloaders.batch_samplers.random_buffer import MinRedBufferDatastreamBatchSampler
from trainloaders.batch_samplers.minred_buffer import MinRedBufferBatchSampler  # buffer onlyで学習を行うMinRed
from trainloaders.batch_samplers.minred_buffer import MinRedBufferDatastreamBatchSampler



### trainloaderを作成して返す
def get_trainloader(dataset, args):

    print(dataset)

    if args.train_samples_per_cls is None:
        train_samples_per_cls = [2500]
    else:
        train_samples_per_cls = args.train_samples_per_cls
        

    """  Samplerの作成  """
    # iidデータセット（iidデータストリーム）の場合
    if args.senario == "iid":

        # CIFAR100 or CIFAR10の場合
        if args.dataset == "cifar100" or args.dataset == "cifar10":
            ## ここ少し微妙
            #train_sampler = IidSampler(dataset, batch_size=args.batch_size, seed=args.seed)
            train_sampler = None
        # imagenet100の場合
        elif args.dataset == 'imagenet100':
            assert False
        else:
            assert False
    
    # seqデータストリームの場合
    elif args.senario == "seq":

        # CIFAR100 or CIFAR10の場合
        if args.dataset == 'cifar100' or args.dataset == 'cifar10':
            train_sampler = SeqSampler(dataset, batch_size=args.batch_size,
                                       blend_ratio=args.blend_ratio,
                                       n_concurrent_classes=args.n_concurrent_classes,
                                       train_samples_per_cls=train_samples_per_cls)
        
        # imagenet100の場合
        elif args.dataset == 'imagenet100':
            assert False

        # krishnacamの場合
        elif args.dataset == 'krishnacam':
            assert False
        
        # 上記以外の場合はエラー
        else:
            assert False

    # 上記以外の場合はエラー
    else:
        assert False



    """  BatchSamplerの作成  """
    if args.buffer_type == "none":
        batch_sampler = None
    
    # ランダムバッファ
    elif args.buffer_type == "random":
        if args.buffer_only:
            batch_sampler = RandomBufferBatchSampler(buffer_size = args.buffer_size,
                                                    repeat=args.num_updates, dataset=args.dataset,
                                                    sampler=train_sampler, batch_size=args.batch_size)
        else:
            batch_sampler = MinRedBufferDatastreamBatchSampler(buffer_size = args.buffer_size,
                                                               repeat = args.num_updates, dataset = args.dataset,
                                                               sampler = train_sampler, batch_size = args.batch_size)
    
    # MinRedバッファ
    elif args.buffer_type == "minred":
        if args.buffer_only:
            batch_sampler = MinRedBufferBatchSampler(buffer_size = args.buffer_size,
                                                     repeat=args.num_updates, dataset=args.dataset,
                                                     sampler=train_sampler, batch_size=args.batch_size)
        else:
            batch_sampler = MinRedBufferDatastreamBatchSampler(buffer_size = args.buffer_size,
                                                               repeat=args.num_updates, dataset=args.dataset,
                                                               sampler=train_sampler, batch_size=args.batch_size)

    
    # それ以外の場合
    else:
        assert False


    """  TrainLoaderの作成  """
    #　バッファを使わない場合
    if batch_sampler is None:

        # バッファ不使用かつiidの場合
        if train_sampler is None:
            train_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=10,
                pin_memory=True
            )
        
        # バッファ不使用かつseqの場合
        elif train_sampler is not None:
            train_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
                prefetch_factor=1
            )
    
    # バッファを使用する場合
    elif batch_sampler is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
            prefetch_factor=1
        )

    return train_loader

