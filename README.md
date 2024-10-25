# ProtFAD

[![arxiv](https://img.shields.io/badge/arxiv-2405.15158-red?logo=arxiv&logoColor=%23B31B1B&logoSize=auto&labelColor=red&color=gray&link=https%3A%2F%2Farxiv.org%2Fabs%2F2405.15158)](https://arxiv.org/abs/2405.15158) 

Here, we introduce **ProtFAD**, a synergistic integration approach for protein function prediction, including the **function-aware domain** pre-training and a **domain-joint contrastive learning** strategy for multi-modal representations. Our approach significantly and comprehensively outperforms the state-of-the-art methods on various benchmarks, and clearly differentiates proteins carrying distinct functions compared to the competitor. 

![main](figures/main.png)



## Environment

python=3.8   

pytorch=1.12.1

```shell
pip install torch-geometric
pip install pandas omegaconf
```



## Data

> [!NOTE]
>
> The data will be made public soon.

****



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

```
@article{wang2024protfad,
  title={ProtFAD: Introducing function-aware domains as implicit modality towards protein function perception},
  author={Wang, Mingqing and Nie, Zhiwei and He, Yonghong and Ren, Zhixiang},
  journal={arXiv preprint arXiv:2405.15158},
  year={2024}
}
```

