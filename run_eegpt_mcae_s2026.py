"""
复现 Checkpoint 2 的【阶段 2】(从阶段 1 的 last.ckpt 微调) -> 最终目标:
    best-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2026-ep193-vs1.3019.ckpt

配置来源: logs_EEGPT_mcae/EEGPT_mcae_CAE_csv/d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2026/hparams.yaml
训练方式: 加载阶段 1 (seed=2025) 的 last.ckpt 继续训练

前置条件:
    1. 已运行 run_eegpt_mcae_s2025.py 完成 200 epoch
    2. 在 ./logs_EEGPT_mcae/checkpoints/ 下存在形如
       last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-ep199-vs<X>.ckpt
       的文件 (原仓库 vs 值=1.2880; 复现时该值会随训练波动)

阶段 2 与阶段 1 的唯一差异:
    - seed:        2025 -> 2026
    - predictor_depth: 4   -> 8
    - ckpt_path:   None  -> 阶段 1 的 last.ckpt

用法:
    # 默认按原仓库相对路径查找阶段 1 的 last.ckpt (用 glob 自动匹配, 避免vs值写死):
    python run_eegpt_mcae_s2026.py

    # 或显式指定阶段 1 的 ckpt 路径:
    python run_eegpt_mcae_s2026.py \
        --ckpt_path logs_EEGPT_mcae/checkpoints/last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-ep199-vs1.2880.ckpt
"""
import glob
import os
import pretrain_template_bf16 as pretrain_template

####################################################################
# 默认 ckpt_path: 自动匹配阶段 1 的 last.ckpt
_DEFAULT_CKPT_GLOB = 'logs_EEGPT_mcae/checkpoints/last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-ep199-*.ckpt'
_matches = sorted(glob.glob(_DEFAULT_CKPT_GLOB))
_DEFAULT_CKPT = _matches[-1] if _matches else None  # 取最新一个; 无则留 None 由 CLI 传入

####################################################################
# ARGS —— 所有值严格来自 hparams.yaml (seed=2026)
class ARGS:
    # -- 阶段 2: 从阶段 1 的 last.ckpt 微调
    ckpt_path = _DEFAULT_CKPT

    seed = 2026
    version = None

    KEY = 'EEGPT_mcae'
    SKEY = 'CAE'

    max_epochs      = 200
    steps_per_epoch = None
    max_lr          = 5e-4
    batch_size      = 320
    early_stop      = 5

    log_path  = None
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

    # -- predictor  (阶段 2 关键差异: predictor_depth=8)
    predictor_embed_dim     = 128
    predictor_embed_dim_inp = 256
    predictor_depth         = 8          # hparams(s2026): 8
    use_predictor           = True
    use_combine_z           = True

    # -- phase
    train_phase = True
    valid_phase = False
    use_compile = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce EEGPT_mcae CAE stage-2 (seed=2026, finetune from s2025)')
    parser.add_argument('--ckpt_path', type=str, default=_DEFAULT_CKPT,
                        help='阶段 1 产出的 last.ckpt 路径 (默认自动匹配)')
    parser.add_argument('--ea1', type=int, default=0, help='enc_use_all_attn')
    parser.add_argument('--ea2', type=int, default=0, help='enc_use_indp_chan_sum')
    parser.add_argument('--ea3', type=int, default=0, help='enc_use_indp_time_sum')
    parser.add_argument('--da1', type=int, default=0, help='dec_use_all_attn')
    parser.add_argument('--da2', type=int, default=0, help='dec_use_indp_chan_sum')
    parser.add_argument('--da3', type=int, default=0, help='dec_use_indp_time_sum')
    args = parser.parse_args()

    if args.ckpt_path is None:
        raise SystemExit(
            "[ERROR] 未找到阶段 1 的 last.ckpt。请先运行 run_eegpt_mcae_s2025.py, "
            "或用 --ckpt_path 显式指定路径。\n"
            f"  期望 glob: {_DEFAULT_CKPT_GLOB}"
        )
    ARGS.ckpt_path = args.ckpt_path

    ARGS.enc_use_all_attn      = (args.ea1 == 1)
    ARGS.enc_use_indp_chan_sum = (args.ea2 == 1)
    ARGS.enc_use_indp_time_sum = (args.ea3 == 1)
    ARGS.dec_use_all_attn      = (args.da1 == 1)
    ARGS.dec_use_indp_chan_sum = (args.da2 == 1)
    ARGS.dec_use_indp_time_sum = (args.da3 == 1)

    pretrain_template.ARGS = ARGS
    pretrain_template.init()
    pretrain_template.main()
