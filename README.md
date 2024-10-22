# Introducing domain modality for protein function prediction
This repository is for the paper 'ProtFAD: Introducing function-aware domains as implicit modality towards protein function prediction.'

**Note: The data will be made public after the paper is accepted.**



## Environment

python=3.8   pytorch=1.12.1

```shell
pip install torch-geometric
pip install pandas omegaconf
```



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
