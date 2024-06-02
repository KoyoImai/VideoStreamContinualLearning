from torch.utils.data import Sampler
import torch
import torch.nn.functional as F
from collections import deque


def tensorize_buffer(buffer):
    buffer_tensor = {}
    for k in buffer[0]:
        tens_list = [s[k] for s in buffer]
        if all(t is None for t in tens_list):
            continue
        dummy = [t for t in tens_list if t is not None][0] * 0.
        tens_list = [t if t is not None else dummy for t in tens_list]
        try:
            if isinstance(tens_list[0], torch.Tensor):
                tens = torch.stack(tens_list)
            elif isinstance(tens_list[0], (int, bool, float)):
                tens = torch.tensor(tens_list)
            else:
                tens = torch.tensor(tens_list)
            buffer_tensor[k] = tens
        except Exception as e:
            print(tens_list)
            print(e)
    return buffer_tensor


class BaseBufferBatchSampler(Sampler):

    def __init__(self, buffer_size, repeat, dataset, sampler, batch_size):

        self.sampler = sampler
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.repeat = repeat
        self.all_indices = list(self.sampler)   # 全データ（全フレーム）数

        self.gamma = 0.5

        # バッファの初期化
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False


    ## バッファにデータを追加
    def add_to_buffer(self, n):
        
        if self.db_head >= len(self.all_indices):
            return True
        
        # バッファにインデックス（データ）を保存
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,
                'lifespan': 0,
                'loss': None,
                'feature': None,
                'lifespan': 0,
                'label': None,
                'num_seen': 0,
                'seen': False,
            }]
        self.db_head += len(indices_to_add)

        # lifespanを加算
        for b in self.buffer:
            b['lifespan'] += 1
        
        return False, indices_to_add
    

    def sample_k(self, q, k):

        pass


    def make_batch(self):

        pass
    

    # バッファ内に保存しているデータの情報を更新
    def update_sample_stats(self, sample_info, ssl_method):

        # バッファ内に保持しているデータの，datasetにおけるインデックス：self.bufferでのインデックス
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}
        
        # 学習に使用したデータのdatasetにおけるインデックス
        sample_index = sample_info['meta']['index'].detach().cpu()
        
        # 学習に使用したデータの特徴量
        if ssl_method == "empssl":
            z = sample_info['feature'][:].detach()
        elif ssl_method == "simsiam":
            z1, z2 = sample_info['feature'][:, 0].detach(), sample_info['feature'][:, 1].detach()
            z = z1 + z2
        
        # 特徴量を正規化
        sample_features = F.normalize(z, p=2, dim=-1)

        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg
        
        for i in range(len(sample_index)):
            
            # 学習に使用したデータのdatasetでのインデックスを一つずつ獲得
            db_idx = sample_index[i].item()

            # 学習に使用したデータがバッファに保存されている場合に実行
            if db_idx in db2buff:
                
                # 学習に使用したデータのself.bufferでのインデックスを獲得
                b = self.buffer[db2buff[db_idx]]

                if not b["seen"]:
                    b['feature'] = sample_features[i]
                else:
                    b['feature'] = F.normalize(polyak_avg(b['feature'], sample_features[i], self.gamma), p=2, dim=-1)

                b['seen'] = True
        

        # バッファ内に保存してあるデータのうち，学習に使用したデータの情報だけをsamplesに格納
        samples = [
            self.buffer[db2buff[idx]] for idx in sample_index.tolist() if idx in db2buff
        ]

        if not samples:
            return {}
        else:
            return tensorize_buffer(samples)

    
    def __iter__(self):


        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)

        # modelが見ていないバッチを再送信
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0, -1):
            yield self.batch_history[-i]    
        
        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)
        while self.num_batches_yielded < len(self):

            # バッファにデータを追加
            done, indices_to_add = self.add_to_buffer(self.batch_size)
            self.indices = indices_to_add

            # バッファ内にデータが一定量たまるまでバッファにデータを格納し続ける
            if not done and len(self.buffer) < self.buffer_size:
                continue

            # バッファからデータを削除
            self.resize_buffer(self.buffer_size)

            # 下記の二つを使用して学習の進捗を把握可能 
            #print("self.db_head : ", self.db_head)
            #print("len(self.all_indices) : ", len(self.all_indices))


            # 規定回数（self.repeat）分だけデータを取り出す
            for j in range(self.repeat):

                # ミニバッチ作成（ミニバッチの作り方によってBatchSamplerの種類を変更）
                batch_idx = self.make_batch()

                self.num_batches_yielded += 1

                self.batch_history += [batch_idx]

                yield batch_idx
            
        self.init_from_ckpt = False


    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size
    

    def advance_batches_seen(self):
        return self.db_head, self.all_indices
