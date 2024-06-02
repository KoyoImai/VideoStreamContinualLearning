import argparse

def parse_option():

    parser = argparse.ArgumentParser()

    ## 学習の名前，学習過程・結果の保存場所
    parser.add_argument('--log_name', type=str, default="practice")                                   # 学習の名前
    parser.add_argument('--data_dir', type=str, default='/home/kouyou')
    parser.add_argument('--main_dir', type=str, default="/home/kouyou/VideoStreamContinualLearning")  # main.pyが存在するディレクトリ
    parser.add_argument('--log_dir', type=str, default='Logs')                                        # 学習の記録を保存するディレクトリ

    ## seed値
    parser.add_argument('--seed', type=int, default=0)

    ## データ周り
    parser.add_argument('--dataset', type=str, default='')

    ## モデル周り
    parser.add_argument('--arch', type=str, default="")
    parser.add_argument('--ssl_method', type=str, default="")

    ## 学習パラメータ
    parser.add_argument('--num_patch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--epoch', type=int, default=1)

    ## 損失パラメータ
    parser.add_argument('--empssl_eps', type=float, default=0.2)
    parser.add_argument('--patch_sim', type=int, default=200)
    parser.add_argument('--empssl_tcr', type=float, default=1.0)
    
    # 最適化パラメータ
    parser.add_argument('--optim_lr', type=float, default=0.01)
    parser.add_argument('--optim_momentum', type=float, default=0.9)
    parser.add_argument('--optim_weight_decay', type=float, default=0.0001)

    # データストリームの設定
    parser.add_argument('--senario', type=str, default="")
    parser.add_argument('--train_samples_per_cls', type=int, default=None)
    parser.add_argument('--blend_ratio', type=float, default=0)
    parser.add_argument('--n_concurrent_classes', type=int, default=1)

    # バッファ
    parser.add_argument('--buffer_only', action='store_true')
    parser.add_argument('--buffer_type', type=str, default="")
    parser.add_argument('--buffer_size', type=int, default=4096)
    parser.add_argument('--num_updates', type=int, default=3)

    # その他
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fix_pred_lr', action='store_true')

    opt = parser.parse_args()
    return opt