# Alu_NN

Learn float32 ops from IEEE bit patterns (encoder → solver → decoder).

```bash
pip install -r requirements.txt
python train.py
```

Kaggle: Secret `WANDB_API_KEY`, `kaggle_bootstrap.load_wandb_key()` before `train.py`. Optional: `ALU_CHECKPOINT_DIR`, `PL_ACCELERATOR=gpu` on GPU. TPU/XLA: only `32-true`, `16-true`, `bf16-true` (default `bf16-true` in config — not `bf16-mixed`). If TPU init fails with `slice_builder_worker_addresses` / “Expected 8 worker addresses, got 1”, `train.py` clears `TPU_PROCESS_ADDRESSES` on Kaggle; still broken → restart kernel, then `pip uninstall -y tensorflow` + `pip install tensorflow-cpu`, or run `python train.py --no-wandb` to skip wandb.