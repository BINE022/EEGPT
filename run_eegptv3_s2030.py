"""
复现 Checkpoint 1:
    best-EEGPTV3_CAE_d8_e256_444_A000100_q0th0t0f0a1c0i0o2siam1s2030-ep184-vs0.8981.ckpt

配置来源: logs_EEGPTV3/EEGPTV3_CAE_csv/d8_e256_444_A000100_q0th0t0f0a1c0i0o2siam1s2030/hparams.yaml
训练方式: 从头训练 (无 ckpt_path)

文件名版本串 A000100 的真实含义 (见 pretrain_template_bf16.py init()):
    A{enc_all}{enc_chan}{enc_time}{dec_all}{dec_chan}{dec_time}
    = A{0}{0}{0}{1}{0}{0}  -> dec_use_all_attn=True, 其余 5 个 attn 标志均为 False

用法:
    python run_eegptv3_s2030.py
    # 或从命令行覆盖 6 个 attn 标志 (默认值即复现值, 一般无需修改):
    python run_eegptv3_s2030.py --ea1 0 --ea2 0 --ea3 0 --da1 1 --da2 0 --da3 0
"""
import pretrain_template_bf16 as pretrain_template

####################################################################
# ARGS —— 所有值严格来自 hparams.yaml
class ARGS:
    # -- (从头训练)
    ckpt_path = None

    seed = 2030
    version = None

    KEY = 'EEGPTV3'
    SKEY = 'CAE'

    max_epochs      = 200
    steps_per_epoch = None     # 由 main() 按 len(train_loader)/len(devices) 自动计算
    max_lr          = 5e-4     # hparams: 0.0005
    batch_size      = 160      # hparams: 160  (=32*5)
    early_stop      = 5

    log_path  = None           # 由 init() 自动生成为 ./logs_EEGPTV3/
    log_name  = None
    save_path = None
    save_name = None
    use_IdentityAE = None      # 由 init() 按 SKEY 设置

    devices        = [0, 1, 2, 3, 4]   # hparams: 5 卡
    weight_loss_a  = 1.0

    # -- dataset
    num_workers    = 20
    percentile_q   = -1
    use_tanh       = False
    use_tiny_data  = False
    use_fake_data  = False

    # -- model
    patch_size  = 50
    output_num  = 2
    embed_dim   = 256
    embed_num   = (4, 4, 4)
    depth       = 8
    num_heads   = 16
    mNCxy       = (7, 13, 7, 13)        # hparams: [7,13,7,13]
    use_avg_ref = True

    # -- attention (A000100: 仅 dec_use_all_attn=True)
    enc_use_all_attn      = False
    enc_use_indp_chan_sum = False
    enc_use_indp_time_sum = False
    dec_use_all_attn      = True        # <-- A000100 的第 4 位
    dec_use_indp_chan_sum = False
    dec_use_indp_time_sum = False

    use_time_indp_attn  = True
    use_siamese_forward = True

    # -- predictor
    predictor_embed_dim     = 128
    predictor_embed_dim_inp = 256
    predictor_depth         = 8         # hparams: 8
    use_predictor           = True
    use_combine_z           = True

    # -- phase
    train_phase = True
    valid_phase = False
    use_compile = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce EEGPTV3 CAE checkpoint (seed=2030)')
    # 6 个 attn 覆盖位; default 设为复现值, 不传参即原样复现
    parser.add_argument('--ea1', type=int, default=0, help='enc_use_all_attn')
    parser.add_argument('--ea2', type=int, default=0, help='enc_use_indp_chan_sum')
    parser.add_argument('--ea3', type=int, default=0, help='enc_use_indp_time_sum')
    parser.add_argument('--da1', type=int, default=1, help='dec_use_all_attn (repro=1)')
    parser.add_argument('--da2', type=int, default=0, help='dec_use_indp_chan_sum')
    parser.add_argument('--da3', type=int, default=0, help='dec_use_indp_time_sum')
    args = parser.parse_args()

    ARGS.enc_use_all_attn      = (args.ea1 == 1)
    ARGS.enc_use_indp_chan_sum = (args.ea2 == 1)
    ARGS.enc_use_indp_time_sum = (args.ea3 == 1)
    ARGS.dec_use_all_attn      = (args.da1 == 1)
    ARGS.dec_use_indp_chan_sum = (args.da2 == 1)
    ARGS.dec_use_indp_time_sum = (args.da3 == 1)

    pretrain_template.ARGS = ARGS
    pretrain_template.init()
    pretrain_template.main()
