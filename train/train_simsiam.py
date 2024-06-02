import torch



def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          args):
    
    ## modelをtrainモードに変更
    model.train()

    
    ## train_loaderからデータを取り出して学習
    for data in train_loader:
        # 確認済みのバッチ数をカウント
        batch_i = train_loader.dataset.advance_batches_seen()

        # 学習用データの用意
        images = [data['input1'], data['input2']]

        # gpuが使用可能であれば学習用データをgpuに配置
        if torch.cuda.is_available():
             images = [imgs.cuda(non_blocking=True) for imgs in images]
        
        # modelに学習用データを入力した際の出力を獲得
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])

        # 損失を計算
        loss_per_sample = -(criterion(p1, z2.detach()) + criterion(p2, z1.detach())) * 0.5
        loss = loss_per_sample.mean()

        # バッファ内データの特徴量などの情報を更新
        with torch.no_grad():
            data['feature'] = torch.stack((z1, z2), 1).detach()
            if not args.buffer_type.startswith('none'):
                stats = train_loader.batch_sampler.update_sample_stats(data, args.ssl_method)

    
        # 学習進捗の確認
        db_head, all_indices = train_loader.batch_sampler.advance_batches_seen()
        if batch_i % 20 == 0:
            print("progress : {}/{}[{}/{}], loss : {:.4f}".format(db_head, 
                                                                  len(all_indices),
                                                                  (batch_i%args.num_updates) if (batch_i%args.num_updates)!=0 else args.num_updates,
                                                                  args.num_updates,
                                                                  loss))
        
        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




