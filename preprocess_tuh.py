#!/usr/bin/env python3
"""
TUH EEG v2.0.1 预处理脚本 (重建自 pretrain_data_process.ipynb + 磁盘格式反推)

流水线 (与原 notebook 一致):
    1. mne.io.read_raw_edf(fname, preload=True)
    2. 通道选择: 保留 name.replace('EEG ','').split('-')[0].upper() in used_channels 的通道
       (顺序按 EDF 原始顺序)
    3. raw.resample(250)              # 降采样/重采样到 250 Hz
    4. raw.filter(0.5, 40, method='iir')   # MNE 默认 Butterworth 4阶 + filtfilt (零相位)
    5. eeg_data = raw.get_data() * 1e6     # V -> µV
    6. np.float32(eeg_data)

磁盘格式补充步骤 (notebook 之后):
    7. 按 1000 样本切块: 第 i 块 = data[:, i*1000:(i+1)*1000], 末块可能不足 1000
    8. 每块转 float16 存为 {save_root}/{subject}/{session}/{ids}_part_{i}.npy
    9. CSV 行: ids,subject,data_path={subject}/{session},Dataset,channels,label=None,
              trials_len=T,session,sfreq=250

已在 sample aaaaadhj/s001 (ids=0) 上验证: per-channel corr=1.0, mean|diff|=0.0043 (纯 float16 量化噪声).

用法:
    # 完整处理所有 EDF:
    python preprocess_tuh.py --raw-root /mnt/.../v2.0.1/edf --save-root /disks/HDD3/dataset/TUHEEG_Processed

    # 只处理前 N 个 (测试):
    python preprocess_tuh.py --raw-root ... --save-root ... --limit 5

    # 验证模式: 处理并与已有 processed 目录逐块比对:
    python preprocess_tuh.py --raw-root ... --save-root /tmp/test_out --verify /disks/HDD3/dataset/TUHEEG_Processed --limit 5
"""
import argparse
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd
import mne

mne.set_log_level('warning')

# ---- used_channels (verbatim from pretrain_data_process.ipynb cell-5) ----
USED_CHANNELS = ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
    'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ',
    'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6',
    'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
    'FP1H', 'FP2H', 'AF1', 'AF2', 'AF5', 'AF6', 'F9', 'F10', 'FT9', 'FT10', 'TP9', 'TP10',
    'P9H', 'P10H', 'F1H', 'F2H', 'F5H', 'F6H', 'F7H', 'F8H', 'FC1H', 'FC2H', 'FC5H', 'FC6H',
    'FT1', 'FT2', 'C1H', 'C2H', 'C5H', 'C6H', 'T1', 'T2', 'TP1', 'TP2', 'CP1H', 'CP2H',
    'CP5H', 'CP6H', 'TP3', 'TP4', 'P1H', 'P2H', 'P3H', 'P4H', 'PO1', 'PO2', 'PO5', 'PO6',
    'O9', 'O10', 'FT7H', 'FT8H', 'TP7H', 'TP8H', 'PO9', 'PO10', 'IZ2', 'OZ2', 'PZ2', 'CPZ2',
    'TPP9H', 'TPP10H', 'AFF1', 'AFF2', 'FFC5H', 'FFC3H', 'FFC4H', 'FFC6H', 'FCC5H', 'FCC3H',
    'FCC4H', 'FCC6H', 'CCP5H', 'CCP3H', 'CCP4H', 'CCP6H', 'CPP5H', 'CPP3H', 'CPP4H', 'CPP6H',
    'PPO1', 'PPO2', 'I1', 'I2', 'AFP3H', 'AFP4H', 'AFF5H', 'AFF6H', 'FFT7H', 'FFC1H', 'FFC2H',
    'FFT8H', 'FTT9H', 'FTT7H', 'FCC1H', 'FCC2H', 'FTT8H', 'FTT10H', 'TTP7H', 'CCP1H', 'CCP2H',
    'TTP8H', 'TPP7H', 'CPP1H', 'CPP2H', 'TPP8H', 'PPO9H', 'PPO5H', 'PPO6H', 'PPO10H', 'POO9H',
    'POO3H', 'POO4H', 'POO10H', 'OI1H', 'OI2H', 'T3', 'T4', 'T9', 'T10', 'AFP1', 'AFP2',
    'AFF1H', 'AFF2H', 'PPO1H', 'POO1', 'POO2', 'PPO2H', 'NONE', 'T5', 'T6']

PART_LEN = 1000          # 每块样本数 (与 dataset_pretrain.py split_len 一致)
TARGET_SFREQ = 250       # 目标采样率
CSV_COLUMNS = ['ids', 'subject', 'data_path', 'Dataset', 'channels',
               'label', 'trials_len', 'session', 'sfreq']


def process_one(fname):
    """对单个 EDF 执行完整预处理, 返回 (data_float32[n_chan,T], select_channels[list[str]])."""
    raw = mne.io.read_raw_edf(fname, preload=True)
    orgin_channels = raw.ch_names

    drop_channels, select_channels = [], []
    for ch_name in orgin_channels:
        name = ch_name.replace('EEG ', '').split('-')[0].upper()
        if name not in USED_CHANNELS:
            drop_channels.append(ch_name)
        else:
            select_channels.append(name)

    if drop_channels:
        raw.drop_channels(drop_channels)

    raw.resample(TARGET_SFREQ)
    raw.filter(0.5, 40, method='iir')

    eeg_data = raw.get_data() * 1e6
    eeg_data = np.float32(eeg_data)
    return eeg_data, select_channels


def save_chunked(eeg_data, save_dir, ids):
    """按 PART_LEN 切块, 每块 float16 存为 {ids}_part_{i}.npy. 返回 part 数."""
    os.makedirs(save_dir, exist_ok=True)
    T = eeg_data.shape[1]
    n_parts = 0
    for i in range(0, T, PART_LEN):
        chunk = eeg_data[:, i:i + PART_LEN].astype(np.float16)
        np.save(os.path.join(save_dir, f'{ids}_part_{i // PART_LEN}.npy'), chunk)
        n_parts += 1
    return n_parts


def parse_subject_session(fname):
    """从 EDF 路径解析 subject_id 和 session_id (与 notebook 一致).

    路径形如 .../edf/022/aaaaadhj/s001_2004/02_tcp_le/aaaaadhj_s001_t000.edf
    -> subject_id = 'aaaaadhj' (倒数第4层), session_id = 's001' (倒数第3层 split('_')[0])
    """
    trial_info = fname.split('/')
    subject_id = trial_info[-4]
    session_id = trial_info[-3].split('_')[0]
    return subject_id, session_id


def _load_ref_ids_map(ref_root):
    """读 ref_root/all_samples_info.csv, 建 (subject, session) -> [ids,...] 映射 (列表, 因一个
    session 可能有多个 EDF recording t000/t001/...)."""
    csv = os.path.join(ref_root, 'all_samples_info.csv')
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    m = {}
    for r in df.itertuples():
        m.setdefault((r.subject, r.session), []).append(int(r.ids))
    return m


def _load_ref_parts(ref_dir, disk_ids):
    parts = sorted([f for f in os.listdir(ref_dir) if f.startswith(f"{disk_ids}_part_")],
                   key=lambda x: int(x.split('_')[2].split('.')[0]))
    if not parts:
        return None, None
    ref = np.concatenate(
        [np.load(os.path.join(ref_dir, p)).astype(np.float32) for p in parts], axis=1)
    return ref, parts


def verify_against(eeg_data, ids, ref_root, subject_id, session_id, ref_ids_map=None):
    """将刚处理的数据与 ref_root 下已有 parts 比对.

    一个 (subject, session) 在磁盘上可能对应多个 ids (多个 EDF recording). 本函数对每个候选
    disk_ids 计算相关系数, 报告最佳匹配 (corr 最高) 的统计. 这样不依赖本脚本与原处理的 ids 顺序一致.
    """
    ref_dir = os.path.join(ref_root, subject_id, session_id)
    if not os.path.isdir(ref_dir):
        print(f"    [verify] 参考目录不存在: {ref_dir}")
        return

    # 候选 disk ids: 优先用 CSV; 退化为扫目录
    cand = []
    if ref_ids_map is not None:
        cand = ref_ids_map.get((subject_id, session_id), [])
    if not cand:
        cand = sorted({int(f.split('_')[0]) for f in os.listdir(ref_dir)
                       if f.startswith('') and '_part_' in f})

    best = None  # (corr_mean, disk_ids, ref, parts)
    for cid in cand:
        ref, parts = _load_ref_parts(ref_dir, cid)
        if ref is None:
            continue
        n = min(eeg_data.shape[1], ref.shape[1])
        nc = min(eeg_data.shape[0], ref.shape[0])
        d, p = eeg_data[:nc, :n], ref[:nc, :n]
        corrs = [float(np.corrcoef(d[i], p[i])[0, 1]) for i in range(nc)]
        cmean = float(np.mean(corrs))
        if best is None or cmean > best[0]:
            best = (cmean, cid, ref, parts, d, p)

    if best is None:
        print(f"    [verify] 候选 ids {cand} 均无 part 文件")
        return
    cmean, cid, ref, parts, d, p = best
    diff = np.abs(d - p)
    nc = d.shape[0]
    corrs = [float(np.corrcoef(d[i], p[i])[0, 1]) for i in range(nc)]
    print(f"    [verify] best_disk_ids={cid} (over {len(cand)} cands) parts={len(parts)} "
          f"ref_shape={ref.shape} chan_corr[min/mean/max]={min(corrs):.4f}/{np.mean(corrs):.4f}/{max(corrs):.4f} "
          f"|diff| [mean/median/max]={diff.mean():.4f}/{np.median(diff):.4f}/{diff.max():.4f}")


def main():
    ap = argparse.ArgumentParser(description='TUH EEG v2.0.1 preprocessing (reproduces TUHEEG_Processed)')
    ap.add_argument('--raw-root', required=True,
                    help='TUH v2.0.1 edf 根目录 (含 000/, 001/, ...)')
    ap.add_argument('--save-root', required=True, help='输出根目录')
    ap.add_argument('--cache', default='eeg_path_list.npy',
                    help='EDF 路径列表缓存 (与 notebook 一致)')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个 (0=全部)')
    ap.add_argument('--start', type=int, default=0, help='从第 N 个开始 (跳过前 N 个)')
    ap.add_argument('--csv-name', default='all_samples_info.csv')
    ap.add_argument('--verify', default=None,
                    help='验证模式: 给出已有 processed 根目录, 处理后逐样本比对')
    ap.add_argument('--append', action='store_true',
                    help='追加模式: 不重写 CSV, 读取已有 CSV 继续 ids 编号')
    args = ap.parse_args()

    # ---- 收集 EDF 列表 ----
    if os.path.exists(args.cache):
        eeg_files = list(np.load(args.cache, allow_pickle=True))
        print(f"[cache] 从 {args.cache} 载入 {len(eeg_files)} 个 EDF 路径")
    else:
        eeg_files = sorted(glob.glob(os.path.join(args.raw_root, '**/*.edf'), recursive=True))
        np.save(args.cache, np.array(eeg_files))
        print(f"[scan] 在 {args.raw_root} 下找到 {len(eeg_files)} 个 EDF, 缓存到 {args.cache}")

    # ---- ids 起点与已有行 ----
    rows = []
    ids = 0
    out_csv = os.path.join(args.save_root, args.csv_name)
    if args.append and os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        rows = old.to_dict('records')
        ids = int(old['ids'].max()) + 1
        print(f"[append] 已有 CSV {len(rows)} 行, 新 ids 从 {ids} 起")

    # ---- 切片 ----
    files = eeg_files[args.start:]
    if args.limit > 0:
        files = files[:args.limit]
    print(f"[run] 处理 {len(files)} 个 EDF (start={args.start}, limit={args.limit or 'ALL'})")

    os.makedirs(args.save_root, exist_ok=True)
    error_files = []
    ref_ids_map = _load_ref_ids_map(args.verify) if args.verify else None
    for idx, fname in enumerate(files):
        try:
            eeg_data, select_channels = process_one(fname)
            subject_id, session_id = parse_subject_session(fname)
            save_dir = os.path.join(args.save_root, subject_id, session_id)
            n_parts = save_chunked(eeg_data, save_dir, ids)

            rows.append({
                'ids': ids, 'subject': subject_id,
                'data_path': os.path.join(subject_id, session_id),
                'Dataset': 'TUH_EEG', 'channels': select_channels,
                'label': None, 'trials_len': int(eeg_data.shape[-1]),
                'session': session_id, 'sfreq': TARGET_SFREQ,
            })
            extra = f" parts={n_parts}" if not args.verify else ""
            print(f"  [{idx+1}/{len(files)}] ids={ids} {subject_id}/{session_id} "
                  f"shape={eeg_data.shape}{extra}")
            if args.verify:
                verify_against(eeg_data, ids, args.verify, subject_id, session_id, ref_ids_map)
            ids += 1
        except Exception as e:
            error_files.append((fname, repr(e)))
            print(f"  [{idx+1}/{len(files)}] ERROR {fname}: {e!r}")
            traceback.print_exc()
            continue

    # ---- 写 CSV ----
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(out_csv, index=False)
    print(f"[done] 写入 {out_csv} ({len(df)} 行). error_files={len(error_files)}")
    if error_files:
        np.save(os.path.join(args.save_root, 'error_files.npy'), error_files)


if __name__ == '__main__':
    main()
