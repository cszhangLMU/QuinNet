# QuinNet
Official Pytorch Code base for "QuinNet: Quintuple U-shape Networks for Scale- and Shape-variant Lesion Segmentation"
## Introduction
Deep learning approaches have demonstrated remarkable efficacy in medical image segmentation. However, they continue to struggle with challenges such as the loss of global context information, inadequate aggregation of multi-scale context, and insufficient attention to lesion regions characterized by diverse shapes and sizes. To address these challenges, we propose a new medical image segmentation network, which consists of one main U-shape network (MU) and four auxiliary U-shape sub-networks (AU), leading to Quintuple U-shape networks in total, thus abbreviated as *QuinNet* hereafter. MU devises special attention-based blocks to prioritize important regions in the feature map. It also contains a multi-scale interactive aggregation module to aggregate multi-scale contextual information. To maintain global contextual information, AU encoders extract multi-scale features from the input images, then fuse them into feature maps of the same level in MU, while the decoders of AU refine features for the segmentation task and co-supervise the learning process with MU. Overall, the dual supervision of MU and AU is very beneficial for improving the segmentation performance on lesion regions of diverse shapes and sizes. We validate our method on four benchmark datasets, showing that it achieves significantly better  segmentation performance than the competitors. 
<p align="center">
  <img src="imgs/QuinNet.png"/>
</p>
## Using the code:
The code is stable while using Python 3.6.13, CUDA >=10.1
- Clone this repository:
```bash
git clone 
cd 
```
## Datasets

## Data Format

