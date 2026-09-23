# LeNet-5 on FashionMNIST

This project compares four LeNet-5 settings:

- Baseline: no dropout, weight decay, or batch normalization.
- Dropout: applied to both fully connected hidden layers.
- Weight decay: L2 regularization on convolutional and linear weight matrices.
- Batch normalization: applied before the hidden-layer ReLU activations.

## Installation

Use Python 3.11, which is compatible with the dependencies required by
D2L 1.0.3.

Install the dependencies:

```bash
python -m pip install d2l==1.0.3 torch torchvision matplotlib
```

Place `model.py`, `train.py`, and `test.py` in the same directory.

## Training

To train all four settings:

```bash
python train.py
```

FashionMNIST is downloaded automatically when needed. The script uses a
CUDA GPU if available and otherwise uses CPU.

Each model is trained for 30 epochs with:

- Batch size: 128
- Optimizer: Adam
- Initial learning rate: 0.001
- Learning-rate schedule: cosine annealing toward 0.00001
- Random seed: 42

The hyperparameters are fixed and we choose not to perform a hyperparameter
search.

The official training set is split into 54,000 training examples and 6,000
validation examples using a stratified 90/10 split. All settings use the
same split.

After every epoch, training and validation accuracy are measured in
evaluation mode. This disables dropout and freezes batch-normalization
running statistics.

For each setting, the checkpoint with the highest validation accuracy is
selected. Ties favor lower validation cross-entropy, then the earlier epoch.
All 30 epochs are completed.

After training all settings, the script evaluates their saved epoch states
on the official test set to produce the convergence curves. Test results
are not used for checkpoint selection.

### Training one setting separately

To train only one setting, replace this loop in `main()`:

```python
for setting in SETTINGS:
```

with one of the following:

```python
for setting in ("baseline",):
```

```python
for setting in ("dropout",):
```

```python
for setting in ("weight_decay",):
```

```python
for setting in ("batch_norm",):
```

Then run `python train.py`.

Restore `for setting in SETTINGS:` to generate the complete four-setting
comparison. 

## Output files

Files are saved in the `results/` directory:

| File | Description |
| --- | --- |
| `baseline_best.pt` | Validation-selected baseline checkpoint |
| `dropout_best.pt` | Validation-selected dropout checkpoint |
| `weight_decay_best.pt` | Validation-selected weight-decay checkpoint |
| `batch_norm_best.pt` | Validation-selected BN checkpoint |
| `history.json` | Configurations and per-epoch train, validation, and test measurements |
| `split.json` | Exact training and validation indices |
| `convergence.png` | Training and test accuracy curves |

Each checkpoint stores its setting, model configuration, selected epoch,
and model state. The state includes batch-normalization running statistics
where applicable.

Epoch states are held in memory while generating the curves; only the
validation-selected checkpoint for each setting is saved to disk.

## Testing saved weights

Use the accompanying `test.py`:

```bash
python test.py --checkpoint results/baseline_best.pt
python test.py --checkpoint results/dropout_best.pt
python test.py --checkpoint results/weight_decay_best.pt
python test.py --checkpoint results/batch_norm_best.pt
```

These commands reconstruct the model, load its saved state, and evaluate it
on the official 10,000-example FashionMNIST test set without retraining.

Testing uses the same preprocessing as training: pixels are converted to
tensors and normalized to [-1, 1]. Dropout is disabled, and batch normalization
uses its saved running statistics.

Checkpoints support evaluation but do not contain the optimizer and scheduler
states needed to resume training exactly.