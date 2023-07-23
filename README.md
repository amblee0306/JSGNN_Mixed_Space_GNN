   Joint Space Learning for Graph Neural Networks (JSGNN) in PyTorch
==================================================

## 1. Overview

This repository is an implementation of our paper in PyTorch, based on GIL's implementation.
Some of the libraries in the requirements.txt requires manual installation due to older version needed.

## 2. Setup
### 2.1 Requirements
python == 3.6.2<br>
torch == 1.1.0<br>
numpy == 1.16.4<br>
scipy == 1.3.0<br>
networkx == 2.3<br>
sage == 9.0<br>
geoopt ==0.0.1<br>
torch_scatter == 1.3.0<br>
torch_geometric == 1.3.0

## 3. Usage

### 3.1 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 

### 3.2  ```train.py```

We provide examples of training commands used to train JSGNN for link prediction and node classification. 

#### Link prediction for GIL

  * Cora JSGNN (Test ROC-AUC: 99.36):

```python train.py --task lp --dataset cora --model JSGNN --dropout 0.1 --weight-decay 0.0005 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1 --normalize-feats 0```

  * Pubmed JSGNN (Test ROC-AUC: 96.97):

```python train.py --task lp --dataset pubmed --model JSGNN --dropout 0.1 --weight-decay 0.0 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 1 --act relu --bias 1```


#### Node classification for GIL

  * Photo JSGNN (Test accuracy: 97.32):

```python train.py --model JSGNN --manifold PoincareBall  --lambda-uniform=0.2 --lambda-wasser=0.1 --task nc --dataset amazonphoto --dropout 0.1 --weight-decay 0.0 --lr 0.01 --dim 16 --num-layers 2 --act elu --bias 1 --use-feat=1 ```

  * Cora JSGNN (Test accuracy: 83.30):

```python train.py --lambda-wasser=0.005 --lambda-uniform=0.5 --task nc --dataset cora --model JSGNN --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --lr 0.01 --dim 16 --num-layers 3 --act elu --bias 1 ```

## Some of the code was forked from the following repositories
 
 * [hgcn](https://github.com/HazyResearch/hgcn)
 * [geoopt](https://github.com/geoopt/geoopt)
 * [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
 * [gil](https://github.com/CheriseZhu/GIL)
