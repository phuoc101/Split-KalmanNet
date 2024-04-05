# Split-KalmanNet

## Split-KalmanNet: A Robust Model-Based Deep Learning Approach for State Estimation

Simulation parameters are in 'config.ini' file.

https://arxiv.org/abs/2210.09636

# Run experiments

- IMPORTANT:

  - Check that you have the correct parameter settings in `config.ini`
  - Toggle TRAIN = True or False at the beginning of the main scripts

## Uniform Circular Motion

### Convergence

```bash
python \(SyntheticNL\)\ main.py
python \(SyntheticNL\)\ plot_loss.py
```

### Noise heterogeneity and non-linearity

```bash
python \(SyntheticNL\)\ main_varied_noise_heterogeneity.py
python \(SyntheticNL\)\
plot_varied_noise_heterogeneity.py
```

## NCLT Dataset

```bash
python \(NCLT\)\ load_data.py
python \(NCLT\)\ main.py
python \(NCLT\)\ plot_trajectory.py
```
