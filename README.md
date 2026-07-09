# EEGPT 预训练代码 — 最小复现包

从 `EEGPT_V2` 代码库提取的、用于完整复现以下两个 checkpoint 的最小代码集：

1. **`best-EEGPTV3_CAE_d8_e256_444_A000100_q0th0t0f0a1c0i0o2siam1s2030-ep184-vs0.8981.ckpt`**
2. **`best-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2026-ep193-vs1.3019.ckpt`**

所有训练超参数严格来自原仓库训练日志中的 `hparams.yaml`（权威来源），未做任何主观改动。

---

## 1. 目录结构

```
EEGPT_pretrain_extract/
├── README.md                       # 本文件
├── requirements.txt                # 预训练 pip 依赖 (training env, numpy>=2, PL>=2.0)
├── requirements-preprocess.txt     # 预处理 pip 依赖 (preprocess env, numpy<2, 含 mne/moabb)
├── environment.yml                 # 预训练 conda 环境定义 (推荐, 一键 recreate)
├── run_all.py                      # ★ 一键运行入口 (3 阶段编排 + dry-run + 前置检查)
├── _run_stage.py                   # run_all.py 的子进程助手 (用户一般不直接调用)
├── pretrain_template_bf16.py       # 核心训练模板 (LitEEGPT / init / main)
├── utils.py                        # WarmupCosineSchedule / CosineWDSchedule / grad_logger
├── run_eegptv3_s2030.py            # 入口: 复现 Checkpoint 1 (从头训练)
├── run_eegpt_mcae_s2025.py         # 入口: 复现 Checkpoint 2 阶段1 (从头训练)
├── run_eegpt_mcae_s2026.py         # 入口: 复现 Checkpoint 2 阶段2 (从阶段1微调)
├── preprocess_tuh.py               # TUH EEG v2.0.1 预处理 (生成 TUHEEG_Processed, 重建自 ipynb)
├── preprocess_moabb.py             # MOABB MI 数据集预处理 (生成 Data_processed, 流水线与 TUH 一致)
├── models/
│   ├── EEGPT_V2.py                 # 提供 seed_torch (init() 依赖)
│   ├── EEGPT_V3.py                 # Checkpoint 1 的模型主文件
│   ├── EEGPT_mcae_opt.py           # Checkpoint 2 的模型主文件 (注意是 _opt 版本)
│   └── modules/
│       ├── attention.py            # RoPE + Multihead Attention
│       ├── block.py                # Transformer Block / MLP / DropPath
│       ├── encoder.py              # EEGPTV3 完整编码器 (含 channel/time/both 嵌入)
│       ├── encoder_mcae.py         # EEGPT_mcae 精简编码器 (仅 time)
│       ├── decoder.py              # EEGPTV3 完整解码器
│       ├── decoder_mcae.py         # EEGPT_mcae 精简解码器
│       ├── predictor.py            # EEGPTV3 预测器
│       ├── predictor_mcae.py       # EEGPT_mcae 预测器
│       ├── electrode.py            # 227 通道电极名/索引映射
│       ├── projector.py            # ProjectionHead / PredictionHead
│       ├── utils.py                # NegativeCosineSimilarity / update_momentum / get_collapse_level ...
│       └── utils_mask.py           # make_masks / make_masks_indp_chan / apply_mask / make_target
├── training_logs/                  # ★ 原始训练日志 (来自 EEGPT_V2, 供参考/绘图)
│   ├── EEGPTV3_CAE_csv/            #   Ckpt1 (s2030) 的 CSV 指标 + hparams
│   ├── EEGPTV3_CAE_tb/             #   Ckpt1 (s2030) 的 TensorBoard 事件
│   ├── EEGPT_mcae_CAE_csv/         #   Ckpt2 阶段1 (s2025) + 阶段2 (s2026) 的 CSV
│   └── EEGPT_mcae_CAE_tb/          #   Ckpt2 阶段2 (s2026) 的 TensorBoard 事件
└── dataset/
    └── dataset_pretrain.py         # Fusion_eeg_dataset / get_data
```

> **源文件说明**:
> - `pretrain_template_bf16.py` 的 `init()` 中有硬编码 `from models.EEGPT_V2 import seed_torch`,
>   因此即便复现 mcae 也需要保留 `models/EEGPT_V2.py`。
> - 原仓库 `pretrain_template_bf16.py` 顶层有 `import models.models_liu` (别人的 CSFM 模型),
>   但仅当 `KEY=='CSFM_base'` 时使用, 本复现包的两个 checkpoint 不会触发。已改为 lazy import
>   (移入 CSFM_base 分支), 故无需拷贝 `models/models_liu/` 目录。
> - 其余 17 个源文件均经 md5 校验与原仓库字节一致。

> **`training_logs/` 说明**:
> 从 `EEGPT_V2/logs_EEGPTV3/` 和 `logs_EEGPT_mcae/` 提取的原始训练日志, 仅包含目标 seed (s2030 / s2025 / s2026), 排除了无关 seed (如 s2031)。
> 每个 seed 子目录含 `hparams.yaml` (超参数快照) + `metrics.csv` (逐 epoch 指标: train_loss / valid_loss / lr / collapse_level ...)。
> `_tb` 子目录额外含 TensorBoard 事件文件, 可用 `tensorboard --logdir training_logs` 可视化训练曲线。
> 复现训练后可将新生成的 metrics.csv 与此目录的原始日志对比, 验证复现质量。

---

## 2. 环境搭建

### 2.1 双环境说明（重要！）

**预处理** 与 **预训练** 使用不同的 Python 环境，二者不能混用：

| | 预处理 (`preprocess_tuh.py` / `preprocess_moabb.py`) | 预训练 (`run_eegptv3_*.py` / `run_eegpt_mcae_*.py`) |
|---|---|---|
| 依赖清单 | `requirements-preprocess.txt` | `requirements.txt` / `environment.yml` |
| conda env 名 | `eegpt-prep` | `eegpt` |
| Python | 3.10 | 3.11 |
| numpy | **1.26.4** ← moabb 1.2.0 不兼容 numpy>=2 | **2.4.3** |
| pytorch-lightning | 不需要 | **>= 2.0**（代码用 `precision="bf16-mixed"`, PL 1.x 会崩溃） |
| mne / moabb | 1.9.0 / 1.2.0 | 不需要 |
| torch | 不需要 | 2.6.0+cu118 |

> **如果数据已预处理完毕**（已有 `TUHEEG_Processed` / `Data_processed`），**只需搭建预训练环境**，跳过预处理环境。

### 2.2 已验证环境 (原训练服务器)

**预训练环境**（对应 `environment.yml`）:

| 项 | 值 |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| GPU | 5× NVIDIA GeForce RTX 3090 (24 GB) |
| NVIDIA Driver | 515.65.01 |
| CUDA (torch wheel) | 11.8 |
| Python | 3.11 |
| torch | 2.6.0+cu118 |
| pytorch-lightning | 2.6.5 |
| numpy / pandas / scipy | 2.4.3 / 3.0.3 / 1.17.1 |
| einops | 0.8.2 |

**预处理环境**（对应 `requirements-preprocess.txt`，与训练环境隔离）:

| 项 | 值 |
|---|---|
| Python | 3.10 |
| numpy / pandas / scipy | 1.26.4 / 1.5.3 / 1.10.1 |
| mne / mne-bids | 1.9.0 / 0.16.0 |
| moabb | 1.2.0 |

### 2.3 安装方式

**① 预训练环境** — 方式 A: conda（推荐，一键复现）:
```bash
conda env create -f environment.yml
conda activate eegpt
```

**① 预训练环境** — 方式 B: pip（往现有环境装）:
```bash
# 建议 conda create -n eegpt python=3.11 后再装
pip install -r requirements.txt
# 若 CUDA 版本不同, 到 https://pytorch.org/get-started/previous-versions/
# 找对应 torch wheel 后缀 (如 +cu121), 替换 requirements.txt
```

**② 预处理环境**（仅当需要从原始数据重新生成 `TUHEEG_Processed` / `Data_processed` 时）:
```bash
conda create -n eegpt-prep python=3.10
conda activate eegpt-prep
pip install -r requirements-preprocess.txt
```

> **Dreyer2023 数据集**需要 `moabb>=1.3.0`，当前 1.2.0 不含，已在 `preprocess_moabb.py` 中跳过。

### 2.4 环境变量

训练脚本头部已自行设置 (无需手动配置, 仅参考):
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NCCL_NSOCKS_PERTHREAD=4
NCCL_SOCKET_NTHREADS=8
```

### 2.5 硬件要求

- **原训练**: 5× RTX 3090 (`devices=[0,1,2,3,4]`), bf16-mixed 精度, 约 200 epoch × 3 阶段。
- **最低可跑**: 1× 任何支持 bf16 的 GPU (Ampere 及以上, 即 RTX 30xx / A100 / H100)。改 `devices=[0]` 即可, 但 `steps_per_epoch = ceil(len(train_loader)/len(devices))` 会变, LR 调度节奏随之改变 → 复现精度会有偏差 (见 §6)。
- **显存**: 单卡 batch_size=160 (V3) / 320 (mcae) 在 24GB 卡上够用; 若 OOM, 把入口脚本的 `batch_size` 减半并同步把 `max_lr` 减半。

### 2.6 快速验证 (Dry-run, 强烈建议首次运行先做)

无需任何数据, 用 fake 数据走通完整训练代码路径 (~30 秒/阶段, 1 GPU):

```bash
conda activate eegpt   # 预训练环境
python run_all.py --dry-run
```

预期: 3 个阶段依次跑完, 每阶段 1 epoch × 6 step, 产出 `./logs_dry_run/` 下的测试 ckpt (不可用于推理)。
若此步报错, 说明环境/代码有问题, 先解决再准备数据。详见 §4.1。

---

## 3. 数据准备 (关键)

`dataset/dataset_pretrain.py` 中 **硬编码** 了两个数据集路径 (按用户要求保留原样, 不做参数化):

| 变量 | 路径 | 说明 |
|---|---|---|
| `data_dir_1` | `/disks/HDD3/dataset/TUHEEG_Processed` | TUH EEG 衰减数据集 |
| `data_info_path_1` | `<data_dir_1>/all_samples_info.csv` | 样本索引 (含 ids / channels / trials_len / data_path) |
| `data_dir_2` | `/disks/SSD/data/transformed2` | 第二数据集根目录 (按 `data_path` 列拼接) |
| `data_info_path_2` | `/disks/HDD3/dataset/Data_processed/Data_processed/all_samples_info.csv` | 第二数据集索引 |

### 数据划分 (来自 `get_data()`)

- **训练集**: `info_1[ids<69000]` ∪ `info_2[Dataset ∉ {Cho2017, PhysionetMI, Zhou2016, BNCI2015_001, BNCI2014_001, AlexMI, BNCI2014_002, BNCI2014_004}]`
- **验证集**: `info_1[69000≤ids<69002]` ∪ `info_2[Dataset ∈ {Zhou2016, BNCI2015_001, AlexMI}]`
- **测试集** (实际用于 valid_loss 监控): `info_1[ids≥69032]` ∪ `info_2[Dataset==BNCI2014_001]`

### `all_samples_info.csv` 必需列

- `ids` (int): 样本唯一 id
- `channels` (str, 形如 `"['Fp1','Fp2',...]"`): 通道名列表 (eval 后为 list)
- `trials_len` (int): 原始 trial 长度 (采样点数)
- `data_path` (str): 相对样本文件路径
- `Dataset` (str, 仅 info_2): 数据集来源名

### 数据文件格式

- TUHEEG (`data_dir_1`): `{data_path}/{ids}_part_{i}.npy`,每片 1000 采样点, 形状 `(n_chan, 1000)`,按时间拼接取 5 秒窗口
- 第二数据集 (`data_dir_2`): `{data_path}` 直接 `np.load`, 形状 `(n_chan, T)`

> 如需在不同机器复现, 请在 `dataset/dataset_pretrain.py` 第 218/222/223 行替换路径, 或用软链接 `ln -s /your/path /disks/HDD3/dataset/TUHEEG_Processed` 等方式满足原路径。

---

## 4. 复现命令

### 4.1 一键运行 (推荐)

`run_all.py` 自动按依赖顺序跑完 3 个阶段, 含前置检查 (数据/GPU) 与 dry-run 模式:

```bash
cd EEGPT_pretrain_extract

# ★ 最常用: 一键复现全部 (需 5 GPU + 数据就绪, 耗时数小时~天)
python run_all.py

# 仅复现 Ckpt1:
python run_all.py --only v3

# 仅复现 Ckpt2 阶段2 (需先完成阶段1):
python run_all.py --only mcae2

# Dry-run: 无需数据, 验证代码可跑通 (~30 秒/阶段, 1 GPU)
python run_all.py --dry-run

# 跳过前置检查 (调试用):
python run_all.py --skip-prereqs -y
```

`run_all.py` 会:
1. 检查 `TUHEEG_Processed` / `Data_processed` 索引文件存在 (除非 `--dry-run` / `--skip-prereqs`)
2. 检查 GPU 数 ≥ 5 (除非 `--dry-run` / `--skip-prereqs`)
3. 检查依赖阶段产物 (mcae2 需要 mcae1 的 last.ckpt)
4. 子进程依次跑 `run_eegptv3_s2030` → `run_eegpt_mcae_s2025` → `run_eegpt_mcae_s2026`
5. 汇总各阶段状态 + 最终 ckpt 路径

### 4.2 单阶段手动运行 (与 run_all.py 等价, 便于调试)

#### Checkpoint 1: `best-EEGPTV3_...s2030...` (从头训练, 单阶段)

```bash
cd EEGPT_pretrain_extract
python run_eegptv3_s2030.py
```

产出: `./logs_EEGPTV3/checkpoints/best-EEGPTV3_CAE_d8_e256_444_A000100_q0th0t0f0a1c0i0o2siam1s2030-ep{E}-vs{V}.ckpt`

#### Checkpoint 2: `best-EEGPT_mcae_...s2026...` (两阶段)

**阶段 1** — 从头训练产出 seed=2025 的 last.ckpt:

```bash
python run_eegpt_mcae_s2025.py
```

产出: `./logs_EEGPT_mcae/checkpoints/last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-ep199-vs{V}.ckpt`

**阶段 2** — 从阶段 1 的 last.ckpt 微调产出目标 best.ckpt:

```bash
# 默认自动匹配阶段 1 的 last.ckpt
python run_eegpt_mcae_s2026.py

# 或显式指定
python run_eegpt_mcae_s2026.py \
    --ckpt_path logs_EEGPT_mcae/checkpoints/last-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2025-ep199-vs1.2880.ckpt
```

产出: `./logs_EEGPT_mcae/checkpoints/best-EEGPT_mcae_CAE_d8_e256_444_A000000_q0th0t0f0a1c0i0o2siam1s2026-ep{E}-vs{V}.ckpt`

---

## 5. 配置总览 (来自 hparams.yaml)

| 参数 | Ckpt1 (EEGPTV3) | Ckpt2 阶段1 (mcae s2025) | Ckpt2 阶段2 (mcae s2026) |
|---|---|---|---|
| KEY / SKEY | EEGPTV3 / CAE | EEGPT_mcae / CAE | EEGPT_mcae / CAE |
| seed | 2030 | 2025 | 2026 |
| max_epochs | 200 | 200 | 200 |
| max_lr | 5e-4 | 5e-4 | 5e-4 |
| batch_size | 160 | 320 | 320 |
| steps_per_epoch | 427 | 214 | 214 |
| mNCxy | (7,13,7,13) | (7,13,7,13) | (7,13,7,13) |
| depth / embed_dim / embed_num | 8 / 256 / (4,4,4) | 同左 | 同左 |
| num_heads | 16 | 16 | 16 |
| patch_size / output_num | 50 / 2 | 50 / 2 | 50 / 2 |
| enc_use_all_attn | F | F | F |
| enc_use_indp_chan_sum | F | F | F |
| enc_use_indp_time_sum | F | F | F |
| **dec_use_all_attn** | **T** ← A000100 | F | F |
| dec_use_indp_chan_sum | F | F | F |
| dec_use_indp_time_sum | F | F | F |
| use_time_indp_attn | T | T | T |
| use_siamese_forward | T | T | T |
| predictor_embed_dim / _inp | 128 / 256 | 128 / 256 | 128 / 256 |
| **predictor_depth** | 8 | **4** | **8** |
| use_predictor / use_combine_z | T / T | T / T | T / T |
| weight_loss_a | 1.0 | 1.0 | 1.0 |
| percentile_q / use_tanh | -1 / F | -1 / F | -1 / F |
| use_avg_ref | T | T | T |
| ckpt_path | (无,从头) | (无,从头) | **阶段1的last.ckpt** |
| devices | [0,1,2,3,4] | [0,1,2,3,4] | [0,1,2,3,4] |
| num_workers | 20 | 20 | 20 |

### 文件名版本串解码

`A000100` 等字段由 `pretrain_template_bf16.py:init()` 生成, 格式为:
```
A{enc_all}{enc_chan}{enc_time}{dec_all}{dec_chan}{dec_time}
```
故 `A000100` = enc(0,0,0) + dec(**1**,0,0) → 仅 `dec_use_all_attn=True`;`A000000` = 6 位全 0。

### 训练框架细节 (对所有运行相同)

- **精度**: `bf16-mixed`
- **优化器**: AdamW, 基础 lr=6e-5, 但 OneCycleLR `max_lr` 由入口脚本给定
- **LR 调度**: OneCycleLR (steps_per_epoch × max_epochs, pct_start=0.2, div_factor=2, final_div_factor=8)
- **WD 调度**: CosineWDSchedule (ref_wd=final_wd=1e-6)
- **动量编码器 EMA** (仅 CAE): 0.996 → 1.0 线性
- **梯度裁剪**: 1.0 (按 norm)
- **Checkpoint**: 监控 `valid_loss`, `mode=min`, `save_top_k=1`;另存一份 `last-` 每 `max_epochs` 保存一次

---

## 6. 复现不确定性说明

- **valid_loss 数值不会完全一致**: 多 GPU + bf16 + 随机数据采样 + cuDNN 非确定性 → checkpoint 文件名里的 `vs0.8981` / `vs1.3019` / `ep184` / `ep193` 会有波动。这是正常的, 不代表配置错误。
- **GPU 数量影响**: `steps_per_epoch` 随 `len(devices)` 变化, 进而改变 OneCycleLR 总步数。强烈建议保持 5 卡;若必须改, 需同步调整 `max_lr` / `batch_size` 以近似保持每 epoch 的总样本数与 lr 节奏。
- **数据采样顺序**: `Fusion_eeg_dataset.__getitem__` 用 `random.randint` 取窗口, 受 `seed_torch(seed)` 控制;改变 seed 会改变数据流。
- **Checkpoint 2 的两阶段依赖**: 阶段 2 的初始权重完全来自阶段 1 的 last.ckpt, 故阶段 1 必须先完整跑完 200 epoch。跳过阶段 1 直接从头训 seed=2026 得到的不是原 checkpoint。
