"""Internal helper for run_all.py — runs ONE entry stage in a fresh subprocess.

Not intended to be called directly by users; use ``run_all.py`` instead.

Usage (called by run_all.py):
    python _run_stage.py <entry_module> [--dry-run]

Where <entry_module> is one of:
    run_eegptv3_s2030      -> Ckpt1 (EEGPTV3 s2030, from scratch)
    run_eegpt_mcae_s2025   -> Ckpt2 stage1 (EEGPT_mcae s2025, from scratch)
    run_eegpt_mcae_s2026   -> Ckpt2 stage2 (EEGPT_mcae s2026, finetune from s2025)

Dry-run overrides:
    - use_fake_data=True   -> Fake_eeg_dataset (227-ch random tensors), no disk I/O
    - max_epochs=1         -> one epoch only
    - devices=[0]          -> single GPU (fast startup, no DDP)
    - num_workers=0        -> avoid subprocess pool overhead
    - ckpt_path=None       -> skip checkpoint loading (mcae2 won't need mcae1)
    - log_path/save_path   -> redirected to ./logs_dry_run/ to keep real logs clean
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        print("Usage: python _run_stage.py <entry_module> [--dry-run]", file=sys.stderr)
        sys.exit(2)
    module_name = sys.argv[1]
    dry_run = '--dry-run' in sys.argv[2:]

    # Ensure cwd is the extract root so relative imports/paths work
    os.chdir(HERE)
    sys.path.insert(0, HERE)

    # Import the entry module — its `if __name__=="__main__"` block does NOT run,
    # so only the ARGS class is defined. Side-effect: imports pretrain_template_bf16.
    entry = importlib.import_module(module_name)
    import pretrain_template_bf16 as pt

    if dry_run:
        a = entry.ARGS
        a.use_fake_data = True     # use Fake_eeg_dataset (no disk I/O)
        a.use_tiny_data = False
        a.max_epochs = 1
        a.devices = [0]            # single GPU
        a.num_workers = 0
        a.early_stop = 0           # disable early stopping
        a.ckpt_path = None         # never load any ckpt in dry-run
        a.train_phase = True
        a.valid_phase = False

    # init() builds version string, log/save paths, seeds torch
    pt.ARGS = entry.ARGS
    pt.init()

    if dry_run:
        # Redirect output to keep real-run logs/ckpts clean
        entry.ARGS.log_path = './logs_dry_run/'
        entry.ARGS.save_path = './logs_dry_run/checkpoints'
        print(f"[dry-run] stage={module_name}  seed={entry.ARGS.seed}  "
              f"KEY={entry.ARGS.KEY}  batch_size={entry.ARGS.batch_size}  "
              f"predictor_depth={entry.ARGS.predictor_depth}", flush=True)

    # main() reads ARGS.* at runtime — including our post-init overrides
    pt.main()


if __name__ == '__main__':
    main()
