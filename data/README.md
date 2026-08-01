# data/

Default dataset root when `FALCON_DATA_ROOT` is NOT set.

- Co-author / fresh machine: just run `python scripts/prepare_data.py --datasets cifar10,mnist`
  — raw datasets download here and standardized pickles land in `processed/`.
- Machine with an existing torchvision root: set the env var instead and this folder stays empty:
  `setx FALCON_DATA_ROOT "D:\pythondata\torch data"` (new shells pick it up).

Nothing in this folder is committed except this README.
