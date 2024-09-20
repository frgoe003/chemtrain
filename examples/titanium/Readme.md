# Titanium

## Training

The following scripts train and evaluate models of titanium for different
values of the cutoff.

To train run the training script and specify devices used for the run
and overwrite some hyperparameters:

```bash
python train.py 0 --cutoff 0.45
```

It is possible to evaluate the training with the script `evaluate.py` providing
the output directory:

```bash
python evaluate.py output/titanium_r_cutoff_0.45...
```

