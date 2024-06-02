from torch.utils.data import RandomSampler, Sampler
import torch
import numpy as np
import random



class IidSampler(Sampler):

    def __init__(self, dataset, batch_size, seed):

        # データセットの総データ数
        self.num_samples = len(dataset)

        # バッチサイズ
        self.batch_size = batch_size

        # 確認したバッチ数
        self.num_batches_seen = 0

        # datasetのラベルをnumpy配列に変換し確保
        if torch.is_tensor(dataset.dataset.targets):
            self.labels = dataset.dataset.targets.detach().cpu().numpy()
        else:
            self.labels = np.array(dataset.dataset.targets)

    
    def __iter__(self):

        idx = list(range(self.num_samples))
        random.shuffle(idx)

        assert False
