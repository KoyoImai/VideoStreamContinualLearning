from torch.utils.data import Sampler
from collections import deque
import torch
import random
import torch.nn.functional as F

from trainloaders.batch_samplers.base_buffer import BaseBufferBatchSampler



# データストリームのデータは一度すべてバッファに保存し，バッファから取り出したデータでのみ学習を行う
class MinRedBufferBatchSampler(BaseBufferBatchSampler):

    def __init__(self, buffer_size, repeat, dataset, sampler, batch_size):

        super().__init__(buffer_size, repeat, dataset, sampler, batch_size)


    def resize_buffer(self, n):


        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return
        
        # 類似度が高いデータのインデックスを返す
        def max_coverage_reductioin(x, n2rm):
            
            # 各データの特徴量の類似度を計算
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2

            # 同じデータ同士の類似度をマスクする
            sim.fill_diagonal_(-10.)

            # 削除データのインデックスを保存するリストの初期化
            idx2rm = []

            for i in range(n2rm):
                neig_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neig_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)

            # idx2rmは，bufferにおける削除するデータのインデックスを格納している
            return idx2rm
        
        # self.bufferからインデックスとバッファの内容を取り出す
        buffer = [(b, i) for i, b in enumerate(self.buffer) if b['seen']]

        # 削除対象のデータ数が少ない場合，lifespanの値を元に削除データを決定
        if len(buffer) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.buffer]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()
        
        else:

            # バッファに保存しているデータの特徴量を取り出して結合
            feats = torch.stack([b['feature'] for b, i in buffer], 0)
            
            # 冗長なデータのbufferにおけるインデックスを獲得
            idx2rm = max_coverage_reductioin(feats, n2rm)

            # 削除する冗長なデータのself.bufferにおけるインデックスを獲得
            idx2rm = [buffer[i][1] for i in idx2rm]


        # idx2rm（削除するデータのbufferでのインデックス）を整頓
        idx2rm = set(idx2rm)

        # バッファからデータを削除
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in idx2rm]


    def sample_k(self, q, k):
        if k <= len(q):
            return random.sample(q, k=k)
        else:
            return random.choices(q, k=k)


    def make_batch(self):

        # バッファからバッチサイズ分だけデータを取り出す
        batch = self.sample_k(self.buffer, self.batch_size)

        # datasetのインデックスを取り出す
        batch_idx = [b['idx'] for b in batch]

        return batch_idx
        

    


## バッファ内のデータとデータストリームのデータを組み合わせて学習を実行
## sample_kと__iter__を改良して，リプレイバッファのデータとデータストリームのデータで学習を可能にする
class MinRedBufferDatastreamBatchSampler(MinRedBufferBatchSampler):

    def __init__(self, buffer_size, repeat, dataset, sampler, batch_size):

        super().__init__(buffer_size, repeat, dataset, sampler, batch_size)

    
    def sample_k(self, q, k):
        # バッファ内データの数が,2*self.batch_size未満の場合,入力バッチのみで学習を行う
        if len(q) < 2*k:
            return random.sample(q, k=0)
        # バッファ内データがバッチサイズ以上なら，重複なしでランダムにバッチとして取り出す
        elif k <= len(q):
            return random.sample(q[:-k], k=k)
        else:
            return random.choices(q[:-k], k=k)


    def make_batch(self):

        # データストリームから入力されたデータを取り出す
        batch_idx_incom = [idx for idx in self.indices_to_add]

        # バッファからバッチサイズ分だけデータを取り出す
        batch = self.sample_k(self.buffer, self.batch_size)

        # リプレイバッファから取り出したデータのdatasetでのインデックスを取り出す
        batch_idx_mem = [b['idx'] for b in batch]

        # データストリームから入力されたデータとリプレイバッファから取り出したデータを連結してミニバッチを作成
        batch_idx = batch_idx_incom + batch_idx_mem

        return batch_idx



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

            # バッファ内にデータが一定量たまるまでバッファにデータを格納し続ける
            #if not done and len(self.buffer) < self.buffer_size:
            #    continue

            # 下記の二つを使用して学習の進捗を把握可能 
            #print("self.db_head : ", self.db_head)
            #print("len(self.all_indices) : ", len(self.all_indices))

            # 入力バッチの確認
            self.indices_to_add = indices_to_add

            # 規定回数（self.repeat）分だけデータを取り出す
            for j in range(self.repeat):

                # ミニバッチの作成
                batch_idx = self.make_batch()

                self.num_batches_yielded += 1

                self.batch_history += [batch_idx]

                yield batch_idx
            
            # バッファからデータを削除
            self.resize_buffer(self.buffer_size)
            
        self.init_from_ckpt = False

