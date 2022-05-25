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
 
![image](https://user-images.githubusercontent.com/35360830/170359342-66906468-82fe-4684-9729-1767e1b7876c.png)
 
1. Import spreadsheet into [Google Earth Pro](https://earth.google.com/web/) using **Data Import Wizard** by clicking **Import** under **File**. Click *Next* and click Next.
 ![image](https://user-images.githubusercontent.com/35360830/170358190-0ae0b4c3-bd32-4a54-a59d-9d564c98bc66.png)

2. Select **Latitude** for latitude field and **Longitude** for longitude field and click Next.
 ![image](https://user-images.githubusercontent.com/35360830/170358469-c2945e84-95e8-475d-a819-188ce9df01c4.png)
 
3. Select datatypes for each column. You can typically use the default setting, unless some customized changes are necessary. Click **Finish**. A pop-up window will appear. Click **Yes** to create a style template. 
 ![image](https://user-images.githubusercontent.com/35360830/170358763-366a743a-7d14-4ad6-bc3c-596d139e705a.png)
 ![image](https://user-images.githubusercontent.com/35360830/170358777-58d9e898-fc04-4d74-9238-52f6aed836a3.png)
 
4. You should now be able to see all the data points from the spread sheet have been imported on Google Earth. All points are interactive. CLicking on a point should open a window with all the information about that data point.
 ![image](https://user-images.githubusercontent.com/35360830/170359062-75844e98-5954-47d9-827c-05f36c9f3965.png)
 ![image](https://user-images.githubusercontent.com/35360830/170359078-64257063-76e9-4f4b-8bf0-4211f3a6d3b9.png)
 
5. To collect Pre-Event Images and the corresponding Post-Event images, use the time-lapse function in Google Earth Pro to check if there are images that spot the events around the date, without any cloud-cover. You can zoom in or out using the scroll wheel. Click **R** on the keyboard to reset view angle after each zoom time.

 ![image](https://user-images.githubusercontent.com/35360830/170382336-f0cc26f1-cb4b-4f93-98f3-a71158bf0871.png)

6. Before saving the image, be sure to unselect the location points on the map and unselect all the layers. Also, remove map elements and set scaling to 1% in map options to remove any watermarks and labels from the image. Select the current resolution to avoid Google Earth cropping the current view when saving the image.

 ![image](https://user-images.githubusercontent.com/35360830/170382422-68d6c15e-6d33-4fb2-a0a0-9c97b28eae78.png)

7. The naming format for images is as follows:
GE_LAT(LL)_LONG(LL)_LAT(UR)_LONG(UR)_LS_ID_imageDate_PREorPOST
The meaning of each entry is as follows:
GE: Google Earth.
LAT(LL): latitude for the lower left corner
LONG(LL): longitude for the lower left corner
LAT(UR): latitude for the upper right corner
LONG(UR): longitude for the upper right corner
LS_ID: landslide id for the landslide event from the spreadsheet
imageDate: data of the image from Google Earth
PRE: pre-event image
POST: post-event image
For example, the two image names below correspond to one landslide image pair
GE_43.309023_-122.966736_43.310900_122.963295_LS_354_US_20110729_POST.jpg
GE_43.309023_-122.966736_43.310900_-122.963295_LS_354_US_20070730_PRE.jpg

 
8. Note that we need to record image coordinates for filenames manually. This can be done using the image overlay function by setting the bounding box for the image overlay to cover the current view. 
![image](https://user-images.githubusercontent.com/35360830/170382602-5cad952d-1bb8-4648-add3-32d3eb8e03b3.png)

 
**<h1> Tutorial for Data Labeling </h1>**
 
We use the labelme tool for polygon labeling by observing the pre-event and post-event images.

1. Go the Github repository https://github.com/wkentaro/labelme and follow the instructions provided.
2. Check if your system has the following requirements:
 a.	Ubuntu / macOS / Windows
 b. Python2 / Python3
 c.	[PyQt4 / PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro) / [PySide2](https://wiki.qt.io/PySide2_GettingStarted)

3. There are multiple ways to install the tool:
 a.	Platform agnostic installation: [Anaconda](https://github.com/wkentaro/labelme#anaconda), [Docker](https://github.com/wkentaro/labelme#docker)
 b.	Platform specific installation: Ubuntu, macOS, Windows
 
 Note: My suggestion: Use Anaconda installation. It is the cleanest and fastest way to install.

4. . Install Anaconda and Python3 (if you already have it, skip this step)
 a.	Go to https://www.python.org/downloads/ to install the latest Python3 version
 b.	Go to https://docs.anaconda.com/anaconda/install/windows/ to download and install Anaconda
 c.	You can watch multipleYouTube  videos for installing the above software

5. After successful installation of Anaconda and Python, Go back to https://github.com/wkentaro/labelme and scroll down to Anaconda installation for Python 3 (Go to Python2 if you have this). Open Anaconda prompt on your system (Search in search bar if you cannot find it). Follow line-by-line code from the instructions.

 ![image](https://user-images.githubusercontent.com/35360830/170383655-56f38ffb-3ba8-4588-87fd-1f1562ca5574.png)

6. Close all the windows.
7. Open Anaconda prompt again.
8. Type conda activate labelme
9. Type labelme --> this will open the labelme tool.
 ![image](https://user-images.githubusercontent.com/35360830/170383714-8f62ae69-4d4c-43fb-aad7-74dd62e69596.png)

10. Click Open Dir on labelme tool. Go to the image directory and select an image to annotate. 
 ![image](https://user-images.githubusercontent.com/35360830/170383746-3fa75d27-0276-4afc-a10b-3aad043a6565.png)

11. Observe the ‘Pre’ image to annotate the ‘Post’ image.
 ![image](https://user-images.githubusercontent.com/35360830/170383771-6abda236-97a5-4c9e-987a-722c29c38ea3.png)
 
12. Click Save button. The image will be saved as a .json file.

