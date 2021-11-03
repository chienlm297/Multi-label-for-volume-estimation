# Multi-label-for-volume-estimation

# Proposal Network

```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 512, 1, 1]           --
|    └─Conv2d: 2-1                       [-1, 64, 112, 112]        9,408
|    └─BatchNorm2d: 2-2                  [-1, 64, 112, 112]        128
|    └─ReLU: 2-3                         [-1, 64, 112, 112]        --
|    └─MaxPool2d: 2-4                    [-1, 64, 56, 56]          --
|    └─Sequential: 2-5                   [-1, 64, 56, 56]          --
|    |    └─BasicBlock: 3-1              [-1, 64, 56, 56]          73,984
|    |    └─BasicBlock: 3-2              [-1, 64, 56, 56]          73,984
|    └─Sequential: 2-6                   [-1, 128, 28, 28]         --
|    |    └─BasicBlock: 3-3              [-1, 128, 28, 28]         230,144
|    |    └─BasicBlock: 3-4              [-1, 128, 28, 28]         295,424
|    └─Sequential: 2-7                   [-1, 256, 14, 14]         --
|    |    └─BasicBlock: 3-5              [-1, 256, 14, 14]         919,040
|    |    └─BasicBlock: 3-6              [-1, 256, 14, 14]         1,180,672
|    └─Sequential: 2-8                   [-1, 512, 7, 7]           --
|    |    └─BasicBlock: 3-7              [-1, 512, 7, 7]           3,673,088
|    |    └─BasicBlock: 3-8              [-1, 512, 7, 7]           4,720,640
|    └─AvgPool2d: 2-9                    [-1, 512, 1, 1]           --
├─Sequential: 1-2                        [-1, 512, 1, 1]           (recursive)
|    └─Conv2d: 2-10                      [-1, 64, 112, 112]        (recursive)
|    └─BatchNorm2d: 2-11                 [-1, 64, 112, 112]        (recursive)
|    └─ReLU: 2-12                        [-1, 64, 112, 112]        --
|    └─MaxPool2d: 2-13                   [-1, 64, 56, 56]          --
|    └─Sequential: 2-14                  [-1, 64, 56, 56]          (recursive)
|    |    └─BasicBlock: 3-9              [-1, 64, 56, 56]          (recursive)
|    |    └─BasicBlock: 3-10             [-1, 64, 56, 56]          (recursive)
|    └─Sequential: 2-15                  [-1, 128, 28, 28]         (recursive)
|    |    └─BasicBlock: 3-11             [-1, 128, 28, 28]         (recursive)
|    |    └─BasicBlock: 3-12             [-1, 128, 28, 28]         (recursive)
|    └─Sequential: 2-16                  [-1, 256, 14, 14]         (recursive)
|    |    └─BasicBlock: 3-13             [-1, 256, 14, 14]         (recursive)
|    |    └─BasicBlock: 3-14             [-1, 256, 14, 14]         (recursive)
|    └─Sequential: 2-17                  [-1, 512, 7, 7]           (recursive)
|    |    └─BasicBlock: 3-15             [-1, 512, 7, 7]           (recursive)
|    |    └─BasicBlock: 3-16             [-1, 512, 7, 7]           (recursive)
|    └─AvgPool2d: 2-18                   [-1, 512, 1, 1]           --
├─Sequential: 1-3                        [-1, 10]                  --
|    └─Linear: 2-19                      [-1, 512]                 524,800
|    └─ReLU: 2-20                        [-1, 512]                 --
|    └─BatchNorm1d: 2-21                 [-1, 512]                 1,024
|    └─Dropout: 2-22                     [-1, 512]                 --
|    └─Linear: 2-23                      [-1, 256]                 131,328
|    └─ReLU: 2-24                        [-1, 256]                 --
|    └─BatchNorm1d: 2-25                 [-1, 256]                 512
|    └─Dropout: 2-26                     [-1, 256]                 --
|    └─Linear: 2-27                      [-1, 10]                  2,570
├─Sigmoid: 1-4                           [-1, 10]                  --
==========================================================================================
Total params: 11,836,746
Trainable params: 11,836,746
Non-trainable params: 0
Total mult-adds (G): 3.66
==========================================================================================
Input size (MB): 1.15
Forward/backward pass size (MB): 35.23
Params size (MB): 45.15
Estimated Total Size (MB): 81.53
==========================================================================================
```