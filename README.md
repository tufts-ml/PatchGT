# PatchGT: Transformer over Non-trainable Clusters for Learning Graph Representations


This repository contains PyTorch implementation of the submission: PatchGT: Transformer over Non-trainable Clusters for Learning Graph Representations 
## 0. Environment Setup
enviroment setup: "run conda install -f patchgt.yml"




## 1. Training
To list the arguments, run the following command:
```
python main_seq.py -h
```

To train the given model on ogbg dataset with PatchGT, run the following:

``` 
python run_ogb.py \
    --gnn_type <gin, deepergcn, gcn>                                  \
    --cluster_bar <0.1, 0.2, 0.5>                  \
    --dataset ogbg-molhiv                                  \                       
```    
   
   
   
         


To train the given model on TU dataset with PatchGT, run the following:

``` 
python run_TU.py \
    --gnn_type <gin, deepergcn, gcn>                                  \
    --cluster_bar <0.1, 0.2, 0.5>                  \
    --dataset DD                                 \                       
```    
   


