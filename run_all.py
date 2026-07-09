#!/usr/bin/env python3
"""
EEGPT 一键预训练脚本 (One-click pretraining orchestrator)

复现两个目标 checkpoint:
    1. best-EEGPTV3_CAE_d8_e256_444_A000100_q0th0t0f0a1c0i0o2siam1s2030-ep{E}-vs{V}.ckpt
    2. best-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2026-ep{E}-vs{V}.ckpt
       (依赖阶段1: last-EEGPT_mcae_..._s2025-ep199-vs{V}.ckpt)

三个训练阶段 (按依赖顺序):
    v3    -> run_eegptv3_s2030.py    (Ckpt1, 从头训练, ~200 epoch)
    mcae1 -> run_eegpt_mcae_s2025.py (Ckpt2 阶段1, 从头训练, ~200 epoch)
    mcae2 -> run_eegpt_mcae_s2026.py (Ckpt2 阶段2, 从 mcae1 的 last.ckpt 微调, ~200 epoch)

用法:
    # 完整复现 (3 阶段顺序执行; 需 5 GPU + 数据集就绪):
    python run_all.py

    # 仅复现 Ckpt1:
    python run_all.py --only v3

    # 仅复现 Ckpt2 阶段2 (需先完成 mcae1):
    python run_all.py --only mcae2

    # Dry-run: fake 数据 + 1 epoch + 单卡, 验证代码可跑通 (~30 秒/阶段, 无需数据):
    python run_all.py --dry-run

    # Dry-run 仅 v3:
    python run_all.py --dry-run --only v3

    # 跳过数据/GPU 前置检查 (调试用):
    python run_all.py --skip-prereqs

环境/数据准备详见 README.md 的 "环境搭建" 与 "数据准备" 章节。
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Stage registry: (id, entry_module, description, needs_prior_stage_id)
STAGES = [
    ('v3',    'run_eegptv3_s2030',
     'Ckpt1: EEGPTV3 s2030 (A000100, predictor_depth=8, from scratch)', None),
    ('mcae1', 'run_eegpt_mcae_s2025',
     'Ckpt2 stage1: EEGPT_mcae s2025 (A000000, predictor_depth=4, from scratch)', None),
    ('mcae2', 'run_eegpt_mcae_s2026',
     'Ckpt2 stage2: EEGPT_mcae s2026 (A000000, predictor_depth=8, finetune from s2025 last.ckpt)', 'mcae1'),
]

# Output checkpoint path patterns (used for prereq + summary)
CKPT_PATTERNS = {
    'v3':    'logs_EEGPTV3/checkpoints/best-EEGPTV3_CAE_d8_e256_444_A000100_q0th0t0f0a1c0i0o2siam1s2030-*.ckpt',
    'mcae1': 'logs_EEGPT_mcae/checkpoints/last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-*.ckpt',
    'mcae2': 'logs_EEGPT_mcae/checkpoints/best-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2026-*.ckpt',
}

# Required dataset index files (see README "数据准备")
DATA_PATHS = [
    ('TUHEEG_Processed', '/disks/HDD3/dataset/TUHEEG_Processed/all_samples_info.csv'),
    ('Data_processed',   '/disks/HDD3/dataset/Data_processed/Data_processed/all_samples_info.csv'),
]


def print_banner(stage_id, desc, dry_run):
    mode = 'DRY-RUN (fake data | 1 epoch | 1 GPU | ~30s)' if dry_run \
           else 'FULL (real data | 200 epochs | 5 GPUs | hours-days)'
    bar = '=' * 76
    print('\n' + bar)
    print(f'  STAGE : {stage_id}')
    print(f'  GOAL  : {desc}')
    print(f'  MODE  : {mode}')
    print(bar + '\n', flush=True)


def check_prereqs():
    """Verify data paths + GPU count before a real run. Returns list of warnings."""
    issues = []
    for name, p in DATA_PATHS:
        if not os.path.exists(p):
            issues.append(f'数据索引缺失: {name} -> {p}')
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL).strip()
        n_gpu = len([l for l in out.splitlines() if l.strip()])
        if n_gpu < 5:
            issues.append(f'GPU 数不足: 检测到 {n_gpu} 张, 推荐 5 张 '
                          f'(devices=[0,1,2,3,4]); 若坚持用 {n_gpu} 卡, '
                          f'需同步调整 batch_size / max_lr (见 README §6)')
    except Exception:
        issues.append('无法执行 nvidia-smi, 请确认 CUDA 驱动已安装')
    return issues


def check_prior_stage_done(stage_id):
    """For dependent stages (mcae2), verify the prior stage's ckpt exists."""
    for sid, _, _, prior in STAGES:
        if sid != stage_id or prior is None:
            continue
        matches = glob.glob(CKPT_PATTERNS[prior])
        if not matches:
            print(f'[ERROR] 阶段 {stage_id} 依赖 {prior} 的输出, 未找到:')
            print(f'        期望模式: {CKPT_PATTERNS[prior]}')
            print(f'        请先运行: python run_all.py --only {prior}')
            return False
        print(f'[ok] 依赖 {prior} 的 ckpt 已就绪: {matches[0]}')
    return True


def run_stage(stage_id, module_name, desc, dry_run):
    print_banner(stage_id, desc, dry_run)
    cmd = [sys.executable, '-u', os.path.join(HERE, '_run_stage.py'), module_name]
    if dry_run:
        cmd.append('--dry-run')
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=HERE)
    dt = time.time() - t0
    tag = 'OK' if rc == 0 else f'FAIL(rc={rc})'
    print(f'\n[stage] {stage_id}: {tag}  用时 {dt:.1f}s')
    return rc == 0


def main():
    ap = argparse.ArgumentParser(
        description='EEGPT 一键预训练 (复现两个 checkpoint)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='详见 README.md。Dry-run 示例: python run_all.py --dry-run')
    ap.add_argument('--only', choices=['v3', 'mcae1', 'mcae2'],
                    help='只运行指定阶段 (默认按顺序运行全部 3 阶段)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Dry-run: fake 数据 + 1 epoch + 单卡, 验证代码可跑通 (~30s/阶段)')
    ap.add_argument('--skip-prereqs', action='store_true',
                    help='跳过数据/GPU 前置检查 (仅用于调试)')
    ap.add_argument('-y', '--yes', action='store_true',
                    help='对前置检查警告自动回答 yes')
    args = ap.parse_args()

    # Stage selection
    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
    else:
        stages = STAGES

    # Prereq checks (skipped in dry-run)
    if not args.dry_run and not args.skip_prereqs:
        issues = check_prereqs()
        if issues:
            print('[WARN] 前置检查发现问题:')
            for i in issues:
                print(f'  - {i}')
            if not args.yes:
                resp = input('\n继续运行? [y/N] ').strip().lower()
                if resp != 'y':
                    print('[abort] 用户取消。可用 --dry-run 跳过所有检查, 或 --skip-prereqs 跳过仅此检查。')
                    sys.exit(1)
        # Also verify prior-stage dependencies
        for sid, _, _, _ in stages:
            if not check_prior_stage_done(sid):
                sys.exit(1)

    # Plan summary
    print(f'\n[plan] 将运行 {len(stages)} 个阶段: {[s[0] for s in stages]}')
    if args.dry_run:
        print('[plan] DRY-RUN: 不读取真实数据, 产出 checkpoint 位于 ./logs_dry_run/ (不可用于推理)')

    # Execute
    results = {}
    for sid, mod, desc, _ in stages:
        ok = run_stage(sid, mod, desc, args.dry_run)
        results[sid] = ok
        if not ok:
            print(f'\n[abort] 阶段 {sid} 失败, 停止后续阶段')
            break

    # Summary
    print('\n' + '=' * 76)
    print('  SUMMARY')
    print('=' * 76)
    for sid, _, desc, _ in stages:
        if sid not in results:
            status = 'SKIPPED'
        elif results[sid]:
            status = 'OK'
        else:
            status = 'FAILED'
        print(f'  [{sid:6s}] {status:8s}  {desc}')

    # Locate produced checkpoints (real run only)
    if not args.dry_run:
        print()
        for sid, _, _, _ in stages:
            if not results.get(sid):
                continue
            matches = glob.glob(CKPT_PATTERNS[sid])
            if matches:
                print(f'  {sid} -> {matches[0]}')
        if all(results.get(s[0]) for s in stages):
            print('\n[done] 全部阶段完成。')
        else:
            print('\n[done] 部分阶段失败, 见上方 SUMMARY。')
            sys.exit(1)
    else:
        if all(results.get(s[0]) for s in stages):
            print('\n[done] Dry-run 全部通过 — 代码与脚本验证 OK。')
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()
