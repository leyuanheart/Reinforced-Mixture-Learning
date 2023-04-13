# Reinforced-Mixture-Learning

This repository is the official implementation of the paper `Reinforced-Mixture-Learning` submitted to *Neural Networks*.

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

![fig](https://pic.imgdb.cn/item/62728f9a0947543129d6f2e4.png)

You can run `gmm.py` and `model_free.py` in the `synthetic_data_analysis` to get the results in `results`. The files named like`figure_xx.py` are used to generate the figures in the paper. Each file is self-contained.

For **real data analysis**, we apply our method to three UCI benchmark datasets summarized in the table below.

| Dataset                      | Samples | Features | Number of clusters |
| ---------------------------- | ------- | -------- | ------------------ |
| Iris Plants                  | 150     | 4        | 3                  |
| Wine Recognition             | 178     | 13       | 3                  |
| Malware Executable Detection | 373     | 531      | 2                  |
| Breast Cancer Wisconsin      | 569     | 30       | 2                  |
| RNA-Seq PANCAN               | 801     | 20531    | 5                  |
| APS Failure at Scania Trucks | 2750    | 170      | 2                  |

You can run each `{dataset_name}.py` in the `real_data_analysis` to get the results in `results`. The files named like`figure_xx.py` are used to generate the figures in the paper. Each file is self-contained.
