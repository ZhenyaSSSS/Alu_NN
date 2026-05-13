# Alu_NN

Learn float32 ops from IEEE bit patterns (encoder → solver → decoder).

```bash
pip install -r requirements.txt
python train.py
```

Kaggle: create Secret `WANDB_API_KEY`, then `from kaggle_bootstrap import load_wandb_key; load_wandb_key()` before `train.py`. Optional: `ALU_CHECKPOINT_DIR`, `PL_ACCELERATOR=gpu` on GPU VMs.
