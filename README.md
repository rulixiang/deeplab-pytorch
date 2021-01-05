# DeepLab-Pytorch

![](https://img.shields.io/badge/Language-Python-blue.svg)

Pytorch implementation of DeepLab series, including DeepLabV1-LargeFOV, DeepLabV2, DeepLabV3, and DeepLabV3+. The experiments are all conducted on PASCAL VOC 2012 dataset.

## Setup

### Install Environment with Conda
``` bash
conda create --name py36 python==3.6
conda activate py36
pip install -r requirements.txt
```
### Clone this repository
``` bash
git clone https://github.com/rulixiang/deeplab-pytorch.git
cd deeplab-pytorch
```

## Prepare PASCAL VOC 2012 dataset

### Download PASCAL VOC 2012 dataset

``` bash
## download voc 2012 dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
## untar
tar –xvf VOCtrainval_11-May-2012.tar
```
### Download the augmented annotations
The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ``` SegmentationClassAug.zip ```, you should unzip it and move it to ```VOCdevkit/VOC2012```. The directory sctructure should thus be 

``` bash
./VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```
The ```root_dir``` in the ```.yaml``` files under directory ```config``` should also be updated as your directory.

## Work List
- [x] [DeepLabV1-LargeFOV](#DeepLabV1)
  - [x] train
  - [x] test
  - [x] crf
- [ ] [DeepLabV2-ResNet101](#DeepLabV2)
- [ ] [DeepLabV3](#DeepLabV3)
- [ ] [DeepLabV3+](#DeepLabV3+)

## DeepLabV1
Chen, Liang-Chieh and Papandreou, George and Kokkinos, Iasonas and Murphy, Kevin and Yuille, Alan L, [**Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**](https://arxiv.org/abs/1412.7062), *ICLR 2015*. 
### Train and Test
Before training, you need to download an initial model from [this link](https://github.com/wangleihitcs/DeepLab-V1-PyTorch/blob/master/data/deeplab_largeFOV.pth) and move it to ```weights``` directory.
To train and test a DeepLabV1-LargeFOV network, you need at least one gpu device.

``` bash
## train
python v1/train_deeplabv1.py --gpu 0,1 --config config/deeplabv1_voc12.yaml
## test on trained model
python v1/test_deeplabv1.py --gpu 0 --config config/deeplabv1_voc12.yaml --crf True
```
### Results
The evaulation results are reported in the table below. *``` Random up-down flip ``` and ``` Random rotation ``` cannot improve the performance*.

<table>
<thead>
  <tr>
    <th>Train set</th>
    <th>Val set</th>
    <th>CRF</th>
    <th>Method</th>
    <th>Pixel Accuracy</th>
    <th>Mean IoU</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="8">trainaug</td>
    <td rowspan="8">val</td>
    <td rowspan="4">&#10008;</td>
    <td>DeepLabv1-step</td>
    <td>-</td>
    <td>62.25</td>
  </tr>
  <tr>
    <td>Ours-step</td>
    <td>89.87</td>
    <td>62.20</td>
  </tr>
  <tr>
    <td>DeepLabv1-poly</td>
    <td>-</td>
    <td>65.88</td>
  </tr>
  <tr>
    <td>Ours-poly</td>
    <td>91.31</td>
    <td>65.31</td>
  </tr>
  <tr>
    <td rowspan="4">&#10004;</td>
    <td>DeepLabv1-step</td>
    <td>-</td>
    <td>67.64</td>
  </tr>
  <tr>
    <td>Ours-step</td>
    <td>92.18</td>
    <td>67.96</td>
  </tr>
  <tr>
    <td>DeepLabv1-poly</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Ours-poly</td>
    <td>92.81</td>
    <td>69.11</td>
  </tr>
</tbody>
</table>

*```step``` means the original learning rate decay policy in deeplabv1*,
*```poly``` means the polynomial learning rate decay policy in deeplabv2*.

## DeepLabV2
Chen, Liang-Chieh and Papandreou, George and Kokkinos, Iasonas and Murphy, Kevin and Yuille, Alan L, [**Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs**](https://arxiv.org/abs/1606.00915), *IEEE TPAMI 2017*.

## DeepLabV3

## DeepLabV3+

## Citation
Please consider citing their works if you used DeepLab.
``` c++
@inproceedings{deeplabv1,
  title={Semantic image segmentation with deep convolutional nets and fully connected crfs},
  author={Chen, Liang-Chieh and Papandreou, George and Kokkinos, Iasonas and Murphy, Kevin and Yuille, Alan L},
  booktitle={International Conference on Learning Representations},
  year={2015}
}

@article{deeplabv2,
  title={Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs},
  author={Chen, Liang-Chieh and Papandreou, George and Kokkinos, Iasonas and Murphy, Kevin and Yuille, Alan L},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={40},
  number={4},
  pages={834--848},
  year={2017},
  publisher={IEEE}
}
```