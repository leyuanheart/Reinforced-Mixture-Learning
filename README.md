# Reinforced-Mixture-Learning

This repository is the official implementation of the paper `Reinforced-Mixture-Learning`.

## Requirements

- Python version: Python 3.6.8 :: Anaconda custom (64-bit)

### Main packages for the proposed estimator

- numpy==1.19.5
- pandas==1.1.5
- sklearn==0.24.2
- torch==1.8.0 (cpu)

### Additional packages for experiments

- os
- sys
- random
- matplotlib

### Hardware

- Precision Tower 7910
- CPU：Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz （2 physical CPUs with 10 cores each）

## Reproduce the results of experiments

We provide an example code in `example.ipynb`.

For **synthetic data analysis**, we consider Gaussian mixture settings and general settings. 

![fig](https://pic.imgdb.cn/item/643cf4a30d2dde577701f845.png)

You can run `gmm.py` and `model_free.py` in the `synthetic_data_analysis` to get the results in `results`. The files named like`figure_xx.py` are used to generate the figures in the paper. Each file is self-contained.

For **real data analysis**, we apply our method to three UCI benchmark datasets summarized in the table below.

![](https://pic.imgdb.cn/item/643cf49c0d2dde577701f0df.png)

You can run each `{dataset_name}.py` in the `real_data_analysis` to get the results in `results`. The files named like`figure_xx.py` are used to generate the figures in the paper. Each file is self-contained.

![](https://pic.imgdb.cn/item/643cf4a30d2dde577701f817.png)
