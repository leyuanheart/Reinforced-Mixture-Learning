# Mixture-Learning-with-Reinforcement-Learning

This repository is the official implementation of the paper `Mixture-Learning-with-Reinforcement-Learning` submitted to NeurIPS 2022.

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

We provide an example code in `example.ipynb`.

For **synthetic data analysis**, we consider Gaussian mixture settings and general settings. 

![fig](https://pic.imgdb.cn/item/62728f9a0947543129d6f2e4.png)


For **real data analysis**, we apply our method to three UCI benchmark datasets summarized in the table below.

| Dataset                 | Samples | Features | Number of clusters |
| ----------------------- | ------- | -------- | ------------------ |
| Iris plants             | 150     | 4        | 3                  |
| Wine recognition        | 178     | 13       | 3                  |
| Breast cancer wisconsin | 569     | 30       | 2                  |

You can run `.py` in the `synthetic_data_analysis` and `real_data_analysis` to get the results of this paper.  Each file is self-contained.

