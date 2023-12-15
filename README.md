# BN's Blindness

## Abstract
Code release for paper ["Batch Normalization Is Blind to the First and Second Derivatives of the Loss"](https://arxiv.org/abs/2205.15146) (accepted by AAAI 2024). 

> We prove that when we do the Taylor series expansion of the loss function, the BN operation will block the influence of the first-order term and most influence of the second-order term of the loss. We also find that such a problem is caused by the standardization phase of the BN operation. We believe that proving the blocking of certain loss terms provides an analytic perspective for potential detects of a deep model with BN operations, although the blocking problem is not fully equivalent to significant damages in all tasks on benchmark datasets. Experiments show that the BN operation significantly affects feature representations in specific tasks.

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. Simply create an virtural environment with `python>=3.8` and run `pip install -r requirements.txt` to download the required packages. If you use `anaconda3` or `miniconda`, you can run following instructions to download the required packages in python. 
    ```bash
        conda create -y -n BN python=3.8
        conda activate BN
        pip install pip --upgrade
        pip install -r requirements.txt
        conda activate BN
        conda install pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.1 -c pytorch -c nvidia
    ```
    
---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[zqs1022@sjtu.edu.cn](mailto:zqs1022@sjtu.edu.cn)
