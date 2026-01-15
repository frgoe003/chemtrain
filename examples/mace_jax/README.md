# Foundational Models through MACE-JAX

## Installation

To install mace-jax with foundational model support, issue these commands:
```{bash}
git clone https://github.com/ACEsuit/mace-jax --checkout 7e9d467d1701290b6606a20ff2c625c27e973254 /tmp/mace-jax
sed -i 's/find:/find_namespace:/g' /tmp/mace-jax/setup.cfg
pip install /tmp/mace-jax
```

## Example

The script `train_spice_example.py` illustrates how to load a foundational MACE
model to **chemtrain**, fine-tune it on a dataset, and export it via
**chemtrain-deploy**.
