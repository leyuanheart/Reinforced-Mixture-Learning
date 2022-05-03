# Variable-Selection-with-Reinforcement-Learning

This repository is the official implementation of the paper `Variable-Selection-with-Reinforcement-Learning` submitted to ICML 2022.

## Requirements

- Python version: Python 3.6.8 :: Anaconda custom (64-bit)

### Main packages for the proposed estimator

- numpy == 1.18.1
- pandas == 1.0.3
- sklearn == 0.22.1
- pytorch == 1.4.0

### Additional packages for experiments

- os
- sys
- random
- matplotlib

### Hardware

- Precision Tower 7910
- CPU：Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz （2 physical CPUs with 10 cores each）

## Reproduce simulation results

We provide example codes in `regression.ipynb` and `classification.ipynb`.

For **synthetic data analysis**, we consider cases in the table below.

![simulation settings](https://pic.imgdb.cn/item/61f2a3a32ab3f51d91f27762.png)

For **real data analysis**, we apply our method in two UCI benchmark datasets summarized in the table below.

| Dataset               | Samples | Features | Task           |
| --------------------- | ------- | -------- | -------------- |
| Spambase              | 3680    | 57       | Classification |
| Communities and Crime | 1993    | 101      | Regression     |

You can run `.py` in the `synthetic_data_analysis` and `real_data_analysis` to get the results of this paper. Each file is self-contained.

