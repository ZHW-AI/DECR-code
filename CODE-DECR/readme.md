### A Diverse Embedding-based Composite Reconstruction Encoder-Decoder for Color Fabric Defect Detection

Official PyTorch implementation of the paper A Diverse Embedding-based Composite Reconstruction Encoder-Decoder for Color Fabric Defect Detection. 

Diverse Embedding-based Composite Reconstruction (DECR)
Pytorch Code of DECR method for Color Fabric Defect Detection on YDFID-1 dataset [1], MVTec dataset [2] and VisA dataset [3]. 


### 1. Prepare the datasets.

- (1) YDFID-1 Dataset [1]: The YDFID-1 dataset can be downloaded from this [website](https://github.com/ZHW-AI/YDFID-1).
 
  
- (2) MVTec Dataset [2]: The MVTec dataset can be downloaded from this [website](https://www.mvtec.com/company/research/datasets/mvtec-ad).

 
- (3) VisA Dataset [3]: The VisA dataset can be downloaded from this [website](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar).



### 2.  Environment

python 3.8
pytorch 1.7.1



### 3. Training and testing..

Train and test a model by:
```
python AD_main.py
```

In our code, training and testing are performed together

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.



###  4. References.

[1] Zhang H, Mi H, Lu S. Yarn-dyed fabric image dataset version 1[J]. 2022-06-30]. https://github.com/ZHW-AI/YDFID-1. Version1, 2021.

[2] Bergmann P, Fauser M, Sattlegger D, et al. MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9592-9600.

[3] Zou Y, Jeong J, Pemula L, et al. Spot-the-difference self-supervised pre-training for anomaly detection and segmentation[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022: 392-408.
