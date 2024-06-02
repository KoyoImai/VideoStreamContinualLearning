import torch
import torch.nn.functional as F



def train(train_loader,
          model,
          criterion,
          contrastive_loss,
          optimizer,
          epoch,
          args):
    
    
    ## modelをtrainモードに変更
    model.train()

    ## train_loaderからデータを取り出して学習
    for data in train_loader:

        # 確認済みのバッチ数をカウント
        batch_i = train_loader.dataset.advance_batches_seen()

        # 学習用のデータを用意
        images = data['input']
        images = torch.cat(images, dim=0)
        #print("images.shape : ", images.shape)   # torch.Size([num_patch*batch_soize, 3, data_size, data_size])

        # gpuを使用可能なら学習用データをgpuに配置
        if torch.cuda.is_available():
            images = images.cuda()

        # modelにデータを入力して出力を獲得
        z_proj, feature1, feature2 = model(images)

        # z_projをargs.num_patch分だけ分割
        z_list = z_proj.chunk(args.num_patch, dim=0)

        # 各データの全パッチから平均特徴量を計算
        z_avg = chunk_avg(z_proj, args.num_patch)

        # 損失計算
        loss_contrast, _ = contrastive_loss(z_list, z_avg)
        loss_TCR = cal_TCR(z_proj, criterion, args.num_patch)

        loss = args.patch_sim * loss_contrast + args.empssl_tcr * loss_TCR

        # 学習進捗の確認
        db_head, all_indices = train_loader.batch_sampler.advance_batches_seen()

        print("progress : {}/{}[{}/{}], loss_contrast : {:.4f}, loss_TCR : {:.4f}, loss : {:.4f}".format(db_head,
                                                                                                         len(all_indices),
                                                                                                         (batch_i%args.num_updates) if (batch_i%args.num_updates)!=0 else args.num_updates,
                                                                                                         args.num_updates,
                                                                                                         loss_contrast,
                                                                                                         loss_TCR,
                                                                                                         loss))

        # バッファ内データの情報を更新
        with torch.no_grad():
            data['feature'] = z_avg.detach()
            if not args.buffer_type.startswith('none'):
                
                # バッファ内のデータの情報を更新
                stats = train_loader.batch_sampler.update_sample_stats(data, args.ssl_method)

        
        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


def cal_TCR(z, criterion, num_patches):
    
    z_list = z.chunk(num_patches, dim=0)
    #print("z_list[0].shape : ", z_list[0].shape)   # torch.Size([200, 1024])
    
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss / num_patches
    return loss


def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)