#!/usr/bin/env python3
"""
MOABB -> Data_processed 预处理脚本 (流水线与 TUH 保持一致)

目的
    从 MOABB 库读取 MI 数据集的原始数据, 经与 TUH 一致的预处理流水线,
    生成 /disks/SSD/data/transformed2/Data_processed 同构的目录, 供
    dataset_pretrain.py 中 dataset 2 加载使用.

预处理流水线 (与 preprocess_tuh.py 完全一致, 见 dataset/pretrain_data_process.ipynb)
    1. dataset.get_data(...) -> mne Raw (preload)
    2. raw.pick_types(eeg=True)              # 丢弃 EOG/ECG/EMG/stim 等非 EEG 通道
    3. raw.resample(250)                     # 降采样/重采样到 250 Hz
    4. raw.filter(0.5, 40, method='iir')     # MNE 默认 Butterworth 4 阶 + filtfilt (零相位)
    5. for each annotation (trial):
           i0 = round(onset * sf)
           i1 = round((onset + duration) * sf) + INCLUSIVE_OFFSET
           trial = raw.get_data(start=i0, stop=i1) * 1e6   # V -> uV
           trial.astype(np.float32)
           -> {save_root}/Data_processed/{DS}/sub_{X}/{label}/{global_ids}.npy

磁盘格式 (与原 Data_processed 一致)
    目录:   {save_root}/Data_processed/{Dataset}/sub_{X}/{label}/{ids}.npy
    CSV 列: ids,subject,data_path,Dataset,channels,label,trials_len(s),session,sfreq
        - trials_len(s) 为 **秒** (float), 与 Data_processed 一致; 注意与
          TUH CSV 的 trials_len (采样点数) 区分.
        - subject 为 **每个数据集内** 从 0 重新编号 (sub_0, sub_1, ...).
          原 Data_processed 使用了一个跨数据集的全局编号 (sub_0..sub_319...),
          该编号顺序无法从现有产物反推, 故新版本采用 per-dataset 顺序编号.
          dataset_pretrain.py 仅把 subject 当作分组键, 不依赖其具体数值.
        - data_path 形如 "Data_processed/{DS}/sub_{X}/{label}/{ids}.npy",
          与 data_dir_2 (/disks/SSD/data/transformed2) 拼接即可 np.load.

可用数据集 (15 个, 与原 Data_processed 一致; Dreyer2023 在 moabb<=1.2.0 中缺失)
    AlexMI              -> moabb.datasets.AlexMI              (code=AlexandreMotorImagery)
    BNCI2014_001/002/004
    BNCI2015_001/004
    Cho2017
    Lee2019_MI
    Ofner2017
    PhysionetMI
    Schirrmeister2017
    Stieger2021
    Weibo2014
    Zhou2016

关于"与原 Data_processed 数值一致性"的重要说明
    验证 (BNCI2014_001 sub_8 trial0) 显示: 通道选择 (pick_types eeg=True),
    标签序列, 每 subject 每 session 的 trial 数完全一致, 但逐通道波形相关
    系数仅 ~0.03 — 与是否加滤波 / 窗口边界 / demean 无关. 这说明原 Data_processed
    的产生方式并非简单的 MOABB get_data() + TUH 流水线 (可能直接读 .gdf/.mat,
    或使用了不同版本的源文件). 由于原脚本丢失, 本脚本不保证数值比特一致,
    只保证: (a) 流水线操作与 TUH 完全一致; (b) 目录/CSV 结构与 dataset_pretrain.py
    兼容. 重新预训练时使用本产物即可.

用法
    # 处理所有支持的数据集 (跳过 Dreyer2023):
    python preprocess_moabb.py --save-root /disks/SSD/data/transformed2_new

    # 仅处理前 2 个数据集, 每个数据集前 3 个 subject (快速测试):
    python preprocess_moabb.py --save-root /tmp/test_out \\
        --datasets AlexMI,BNCI2014_001 --limit-subjects 3

    # 验证模式: 处理后逐 trial 与已有 Data_processed 做相关系数比对:
    python preprocess_moabb.py --save-root /tmp/test_out \\
        --datasets BNCI2014_001 --limit-subjects 2 \\
        --verify /disks/SSD/data/transformed2
"""
import argparse
import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import mne

mne.set_log_level('warning')
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 全局常量 (与 preprocess_tuh.py / dataset_pretrain.py 对齐)
# ---------------------------------------------------------------------------
TARGET_SFREQ = 250
LOWCUT, HIGHCUT = 0.5, 40
CSV_COLUMNS = ['ids', 'subject', 'data_path', 'Dataset', 'channels',
               'label', 'trials_len(s)', 'session', 'sfreq']
# 窗口右端是否包含 +1 个采样点. 原 Data_processed 中 BNCI2014_001 trial 为 1001
# 样本 (4s*250Hz=1000, 多 1 个), 故默认为 1. 设为 0 则得到精确 1000.
DEFAULT_INCLUSIVE_OFFSET = 1
DEFAULT_DTYPE = np.float32

# ---------------------------------------------------------------------------
# MOABB 数据集注册表
#   key   = 磁盘 Dataset 列里的字符串 (也是输出目录名)
#   value = (MOABB 类名, 实例化 kwargs)
# Dreyer2023 在 moabb 1.2.0 中尚未提供, 缺省跳过.
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    'AlexMI':           ('AlexMI',            {}),
    'BNCI2014_001':     ('BNCI2014_001',      {}),
    'BNCI2014_002':     ('BNCI2014_002',      {}),
    'BNCI2014_004':     ('BNCI2014_004',      {}),
    'BNCI2015_001':     ('BNCI2015_001',      {}),
    'BNCI2015_004':     ('BNCI2015_004',      {}),
    'Cho2017':          ('Cho2017',           {}),
    'Lee2019_MI':       ('Lee2019_MI',        {}),
    'Ofner2017':        ('Ofner2017',         {}),
    'PhysionetMI':      ('PhysionetMI',       {}),
    'Schirrmeister2017':('Schirrmeister2017', {}),
    'Stieger2021':      ('Stieger2021',       {}),
    'Weibo2014':        ('Weibo2014',         {}),
    'Zhou2016':         ('Zhou2016',          {}),
    # 'Dreyer2023':     ('Dreyer2023',        {}),   # 需 moabb>=1.3.0
}


# ---------------------------------------------------------------------------
# 流水线核心
# ---------------------------------------------------------------------------
def process_subject_raw(raw, inclusive_offset=DEFAULT_INCLUSIVE_OFFSET,
                        dtype=DEFAULT_DTYPE):
    """对单个 mne Raw 对象执行 TUH 一致流水线, 返回 [(label, trial_array, t_len_s), ...].

    流水线: pick_types(eeg=True) -> resample(250) -> filter(0.5,40,iir)
    然后逐 annotation 取 [onset, onset+duration] 窗口, *1e6 转 uV, 转指定 dtype.

    跳过 description 为纯数字事件码 (如 '1','2','999') 或 duration<=0 的标注.
    """
    raw = raw.copy().pick_types(eeg=True)
    raw.resample(TARGET_SFREQ)
    raw.filter(LOWCUT, HIGHCUT, method='iir')

    sf = raw.info['sfreq']
    trials = []
    for ann in raw.annotations:
        desc = str(ann['description'])
        # 跳过纯事件码 (PhysionetMI 等的原始 T0/T1/T2 已由 MOABB 转为描述性字符串,
        # 但部分数据集仍可能保留数字事件码)
        if desc.isdigit():
            continue
        dur = float(ann['duration'])
        if dur <= 0:
            continue
        onset = float(ann['onset'])
        i0 = int(round(onset * sf))
        i1 = int(round((onset + dur) * sf)) + inclusive_offset
        if i1 <= i0:
            continue
        chunk = raw.get_data(start=i0, stop=i1) * 1e6
        trials.append((desc, chunk.astype(dtype), dur))
    return trials


# ---------------------------------------------------------------------------
# 输出 / CSV
# ---------------------------------------------------------------------------
def save_trial(arr, save_root, ds_name, sub_idx, label, global_ids):
    """保存单 trial: {save_root}/Data_processed/{ds}/sub_{X}/{label}/{ids}.npy. 返回相对 data_path."""
    save_dir = os.path.join(save_root, 'Data_processed', ds_name,
                            f'sub_{sub_idx}', str(label))
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{global_ids}.npy'
    np.save(os.path.join(save_dir, fname), arr)
    return os.path.join('Data_processed', ds_name, f'sub_{sub_idx}',
                        str(label), fname).replace('\\', '/')


def verify_trial(arr, verify_root, ds_name, sub_idx_in_disk, label,
                 disk_csv_sub):
    """尝试把刚生成的 trial 与已有 Data_processed 内同 ds+label+subject 的 trials
    对齐 (按 trial 顺序), 报告每 trial 的逐通道相关系数统计. 不强求比特一致."""
    if verify_root is None or disk_csv_sub is None or len(disk_csv_sub) == 0:
        return
    cand = disk_csv_sub[disk_csv_sub['label'] == label]
    if len(cand) == 0:
        print(f"      [verify] label={label!r} 在磁盘 ds={ds_name} sub={sub_idx_in_disk} 无候选")
        return
    # 取该 label 下磁盘 trial 0 对齐 (简单顺序对齐; 不是绝对正确但够诊断)
    row = cand.iloc[0]
    disk_arr = np.load(os.path.join(verify_root, row['data_path']))
    n = min(arr.shape[1], disk_arr.shape[1])
    nc = min(arr.shape[0], disk_arr.shape[0])
    if n < 2 or nc < 1:
        print(f"      [verify] 形状不匹配: new={arr.shape} disk={disk_arr.shape}")
        return
    corrs = [float(np.corrcoef(arr[i, :n], disk_arr[i, :n])[0, 1])
             for i in range(nc)]
    diff = np.abs(arr[:nc, :n] - disk_arr[:nc, :n])
    print(f"      [verify] ds={ds_name} label={label!r} disk_sub={sub_idx_in_disk} "
          f"shape new/disk={arr.shape}/{disk_arr.shape} "
          f"corr[min/mean/max]={min(corrs):.3f}/{np.mean(corrs):.3f}/{max(corrs):.3f} "
          f"|diff|[mean/max]={diff.mean():.2f}/{diff.max():.2f}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='MOABB -> Data_processed preprocessing (TUH-consistent pipeline)')
    ap.add_argument('--save-root', required=True,
                    help='输出根目录 (会在其下创建 Data_processed/)')
    ap.add_argument('--datasets', default=','.join(DATASET_REGISTRY.keys()),
                    help='逗号分隔的数据集名 (default: 全部已注册的 14 个)')
    ap.add_argument('--limit-subjects', type=int, default=0,
                    help='每个数据集只处理前 N 个 subject (0=全部)')
    ap.add_argument('--inclusive-offset', type=int,
                    default=DEFAULT_INCLUSIVE_OFFSET,
                    help='窗口右端 +N 采样点 (default=1, 与原 Data_processed 一致)')
    ap.add_argument('--dtype', choices=['float32', 'float64'],
                    default='float32', help='输出 dtype (default float32, 与 TUH 一致)')
    ap.add_argument('--verify', default=None,
                    help='验证模式: 给出已有 transformed2 根目录, 处理后逐 trial 比对相关系数')
    ap.add_argument('--dry-run', action='store_true',
                    help='不写盘, 只统计 trial 数')
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == 'float32' else np.float64

    # MOABB 数据下载到 MNE_DATA; 让用户已知原始数据的 Moabb_download 不影响
    # (MOABB 会优先使用已存在的缓存)
    import moabb
    from moabb import datasets as moabb_ds
    print(f"[env] moabb={moabb.__version__}  mne={mne.__version__}")

    ds_names = [s.strip() for s in args.datasets.split(',') if s.strip()]
    for n in ds_names:
        if n not in DATASET_REGISTRY:
            print(f"[warn] 未知数据集 {n!r}, 跳过. 已知: {list(DATASET_REGISTRY)}")
    ds_names = [n for n in ds_names if n in DATASET_REGISTRY]

    os.makedirs(os.path.join(args.save_root, 'Data_processed'), exist_ok=True)

    # verify 模式: 一次性载入 ref CSV (若存在), 用于 trial 对齐诊断
    ref_csv = None
    if args.verify:
        rcsv = os.path.join(args.verify, 'Data_processed', 'all_samples_info.csv')
        if os.path.exists(rcsv):
            ref_csv = pd.read_csv(rcsv)
            print(f"[verify] 载入 {rcsv}: {len(ref_csv)} 行")
        else:
            print(f"[verify] 未找到 {rcsv}, 跳过 verify")

    all_rows = []
    global_ids = 0
    for ds_name in ds_names:
        cls_name, kwargs = DATASET_REGISTRY[ds_name]
        DS = getattr(moabb_ds, cls_name)(**kwargs)
        subjects = list(DS.subject_list)
        if args.limit_subjects > 0:
            subjects = subjects[:args.limit_subjects]
        print(f"\n[ds] {ds_name} (moabb class={cls_name}, code={DS.code}) "
              f"subjects={len(subjects)}")

        ds_rows = []
        for s_idx, subj_id in enumerate(subjects):
            try:
                all_data = DS.get_data(subjects=[subj_id])
            except Exception as e:
                print(f"  [sub_{s_idx}/{len(subjects)-1}] moabb_id={subj_id} "
                      f"get_data FAILED: {e!r}")
                continue
            subj_sessions = all_data[subj_id]
            # session key 排序后映射到 0..N-1 (原 Data_processed 用 int session)
            session_keys = sorted(subj_sessions.keys())
            n_trials_subj = 0
            for sess_int, sess_label in enumerate(session_keys):
                runs = subj_sessions[sess_label]
                for run_label, raw in runs.items():
                    trials = process_subject_raw(
                        raw, inclusive_offset=args.inclusive_offset, dtype=dtype)
                    for label, arr, dur_s in trials:
                        if not args.dry_run:
                            data_path = save_trial(
                                arr, args.save_root, ds_name, s_idx, label, global_ids)
                        else:
                            data_path = (f"Data_processed/{ds_name}/sub_{s_idx}/"
                                         f"{label}/{global_ids}.npy")
                        chan_list = [str(c) for c in raw.info['ch_names']]  # 已 pick eeg
                        # 注意: process_subject_raw 内部对 raw 做了 copy + pick_types,
                        # 这里的 raw 还是原始的; 重新取一下 eeg 通道名以保持一致
                        eeg_ch_names = [ch for ch, t in zip(
                            raw.info['ch_names'], raw.get_channel_types()) if t == 'eeg']
                        row = {
                            'ids': global_ids,
                            'subject': s_idx,
                            'data_path': data_path,
                            'Dataset': ds_name,
                            'channels': str(eeg_ch_names),
                            'label': label,
                            'trials_len(s)': float(dur_s),
                            'session': sess_int,
                            'sfreq': TARGET_SFREQ,
                        }
                        ds_rows.append(row)
                        all_rows.append(row)
                        global_ids += 1
                        n_trials_subj += 1

                        if args.verify and ref_csv is not None and n_trials_subj == 1 \
                                and sess_int == 0 and run_label == list(runs.keys())[0]:
                            # 仅对每个 subject 第一 trial 做一次 verify 诊断
                            disk_sub_mask = ref_csv['Dataset'] == ds_name
                            if disk_sub_mask.any():
                                # 由于磁盘 subject 编号与 MOABB subj_id 没有可靠映射,
                                # 取磁盘同 Dataset 下的第一个 subject 做对照
                                min_sub = int(ref_csv.loc[disk_sub_mask, 'subject'].min())
                                disk_csv_sub = ref_csv[
                                    (ref_csv['Dataset'] == ds_name) &
                                    (ref_csv['subject'] == min_sub)]
                                verify_trial(arr, args.verify, ds_name,
                                             min_sub, label, disk_csv_sub)
                # 每个 subject 结束打印一次
            print(f"  [sub_{s_idx}/{len(subjects)-1}] moabb_id={subj_id} "
                  f"n_sessions={len(session_keys)} n_trials={n_trials_subj}")

        # 写 per-dataset CSV
        if ds_rows and not args.dry_run:
            df_ds = pd.DataFrame(ds_rows, columns=CSV_COLUMNS)
            df_ds.to_csv(os.path.join(args.save_root, 'Data_processed',
                                      ds_name, 'samples_info.csv'), index=False)
            print(f"[ds] {ds_name}: 写 {len(df_ds)} 行 samples_info.csv")

    # 写汇总 CSV
    if all_rows and not args.dry_run:
        df_all = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        out = os.path.join(args.save_root, 'Data_processed', 'all_samples_info.csv')
        df_all.to_csv(out, index=False)
        print(f"\n[done] 总计 {len(df_all)} trials, 写入 {out}")


if __name__ == '__main__':
    main()
