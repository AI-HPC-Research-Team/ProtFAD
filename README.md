# ProtFAD

[![arxiv](https://img.shields.io/badge/arxiv-2405.15158-red?logo=arxiv&logoColor=%23B31B1B&logoSize=auto&labelColor=red&color=gray&link=https%3A%2F%2Farxiv.org%2Fabs%2F2405.15158)](https://arxiv.org/abs/2405.15158) 

Here, we introduce **ProtFAD**, a synergistic integration approach for protein function prediction, including the **function-aware domain** pre-training and a **domain-joint contrastive learning** strategy for multi-modal representations. Our approach significantly and comprehensively outperforms the state-of-the-art methods on various benchmarks, and clearly differentiates proteins carrying distinct functions compared to the competitor. 

![main](figures/main.png)



## Environment

python=3.8; pytorch=1.12.1

```shell
pip install torch-geometric
pip install pandas omegaconf
```



## Data

- The pretrained model checkpoint can be found in https://1drv.ms/f/c/c3ec79dd757518a7/UgCnGHV13XnsIIDDw10AAAAAACSJtIYP40Q8jPI.

- Other data may be found in https://1drv.ms/f/c/c3ec79dd757518a7/UgCnGHV13XnsIIDDyl0AAAAAAICBCNJHCdloI3k.

- The data for all tasks can be obtained in https://github.com/hehefan/Continuous-Discrete-Convolution.



## Training

1、To train the model on EC/GO dataset, use

```shell
python train.py -C configs/[dataset]/[dataset]_mulpro_cl.yaml
```

[dataset] is the data name, including "ec, go_mf, go_bp, go_cc".



2、To train the model on FC/ER dataset, use

```
python train_fold.py -C configs/[dataset]/[dataset]_mulpro_cl.yaml
```

[dataset] is the data name, including "fold, func".



## Citation

If you find our work useful in your research, please consider citing:

```
@article{WANG2025127246,
title = {Aligning sequence and structure representations leveraging protein domains for function prediction},
journal = {Expert Systems with Applications},
volume = {278},
pages = {127246},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.127246},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425008681},
author = {Mingqing Wang and Zhiwei Nie and Yonghong He and Athanasios V. Vasilakos and Zhixiang Ren},
keywords = {Protein function prediction, Protein domain, Deep learning, Functional priors, Contrastive learning},
}
```

