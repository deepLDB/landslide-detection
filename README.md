 <h1>Constructing a Large-scale Landslide Database Across Heterogeneous Environments Using  Task-Specific Model Updates <h1>



**<h1> Introduction </h1>**

This is the official code and data for [Constructing a Large-scale Landslide Database Across Heterogeneous Environments Using Task-Specific Model Updates](https://ieeexplore.ieee.org/document/9780028). TSMU focuses on addressing the problem of continually updating a semantic segmentation network with each additional batch of data which can be obtained from a characteristically different ecoregion, with distinct landslide and background properties. The overall framework of TSMU is as shown below:
<img src='https://drive.google.com/uc?id=1Cvy410dbB27Jbxht3TQGztaC7pKIgKxl'>

Figure 1: Illustration of proposed TSMU, illustrated for data from 3 ecoregions acquired sequentially. The semantic segmentation
network contains a shared encoder that extracts features from images and also contains multiple decoders, one for each ecoregion
(task). Each decoder outputs a landslide segmentation (i.e., “decodes” features into a landslide annotation), and the ecoregion
determines which decoder’s output is used. The encoder follows the ResNet-34 architecture while the decoders use the
U-Net architecture. When a new ecoregion (new task) is encountered, instead of creating a completely new
model, TSMU adds a new decoder to the existing model. The goal of TSMU is to update the network parameters in the new
decoder and the shared encoder to (1) achieve acceptable accuracy on the new ecoregion (new task) while (2) maintaining or improving performance on the old ecoregions.

**<h1> Citation </h1>**

Please consider citing our work if it helps you,
```
@ARTICLE{9780028,
  author={Nagendra, Savinay and Kifer, Daniel and Mirus, Benjamin B and Pei, Te and Lawson, Kathryn and Li, Weixin and Nguyen, Hien and Qiu, Tong and Tran, Sarah and Shen, Chaopeng and Manjunatha, Srikanth Banagere},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Constructing a Large-scale Landslide Database Across Heterogeneous Environments Using Task-Specific Model Updates}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSTARS.2022.3177025}}

@inproceedings{nagendra2020efficient,
  title={An efficient deep learning mechanism for cross-region generalization of landslide events},
  author={Nagendra, Savinay and Banagere Manjunatha, S and Shen, Chaopeng and Kifer, Daniel and Pei, Te},
  booktitle={AGU Fall Meeting Abstracts},
  volume={2020},
  pages={NH030--0010},
  year={2020}
}
```
**<h1> Tutorial for Data Accquisition </h1>**
The images used in our paper for each figure are provided in the file data/[Coordinates for images used in figures](https://github.com/deepLDB/landslide-detection/blob/main/data/Coordinates%20for%20images%20used%20in%20figures.xlsx), which contains key information for each landslide record, such as location, event date and corresponding ecoregion as shown below:
 
![image](https://user-images.githubusercontent.com/35360830/170357672-4f61c791-500a-4330-870a-5499bd6b39c3.png)
 

