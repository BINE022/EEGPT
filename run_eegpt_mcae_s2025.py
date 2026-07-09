"""
复现 Checkpoint 2 的【阶段 1】(从头训练):
    last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-ep199-vs1.2880.ckpt

配置来源: logs_EEGPT_mcae/EEGPT_mcae_CAE_csv/d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025/hparams.yaml
训练方式: 从头训练 (无 ckpt_path)

注意: 目标 best-...s2026... 是从本阶段产出的 last.ckpt 微调而来, 必须先跑此脚本。
      产出文件名形如 last-EEGPT_mcae_CAE_..._s2025-ep199-vs<X>.ckpt (vs 值取决于训练结果)。

用法:
    python run_eegpt_mcae_s2025.py
"""
import pretrain_template_bf16 as pretrain_template

####################################################################
# ARGS —— 所有值严格来自 hparams.yaml (seed=2025)
class ARGS:
    # -- 阶段 1: 从头训练
    ckpt_path = None

    seed = 2025
    version = None

    KEY = 'EEGPT_mcae'
    SKEY = 'CAE'

    max_epochs      = 200
    steps_per_epoch = None
    max_lr          = 5e-4        # hparams: 0.0005
    batch_size      = 320         # hparams: 320
    early_stop      = 5

    log_path  = None              # init() 生成为 ./logs_EEGPT_mcae/
    log_name  = None
    save_path = None
    save_name = None
    use_IdentityAE = None

    devices        = [0, 1, 2, 3, 4]
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
    mNCxy       = (7, 13, 7, 13)
    use_avg_ref = True

    # -- attention (A000000: 6 位全 0)
    enc_use_all_attn      = False
    enc_use_indp_chan_sum = False
    enc_use_indp_time_sum = False
    dec_use_all_attn      = False
    dec_use_indp_chan_sum = False
    dec_use_indp_time_sum = False

    use_time_indp_attn  = True
    use_siamese_forward = True

    # -- predictor  (阶段 1 关键差异: predictor_depth=4)
    predictor_embed_dim     = 128
    predictor_embed_dim_inp = 256
    predictor_depth         = 4          # hparams(s2025): 4
    use_predictor           = True
    use_combine_z           = True

    # -- phase
    train_phase = True
    valid_phase = False
    use_compile = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce EEGPT_mcae CAE stage-1 (seed=2025, from scratch)')
    parser.add_argument('--ea1', type=int, default=0, help='enc_use_all_attn')
    parser.add_argument('--ea2', type=int, default=0, help='enc_use_indp_chan_sum')
    parser.add_argument('--ea3', type=int, default=0, help='enc_use_indp_time_sum')
    parser.add_argument('--da1', type=int, default=0, help='dec_use_all_attn')
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
