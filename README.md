# Regression problem on tabular data

<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>

The task was to develop a model application that would be able to segment soil affected by erosion using a satellite image. The task as such is not completed to the end. There were attempts to use the Unet and DeepLab architectures, but in my implementation they did not provide the correct training of the model. In order to quickly see the effect of the model, I decided to reduce the size of the dataset to one where each image has a corresponding mask containing information in the form of segmentation, otherwise it was not saved. Used BinaryCrossEntropy loss and Jaccard loss. For images of this kind, there is a lot of room for successful augmentation.

Project completion plan (after deadline):
- fix model architecture for a better start
- experiment with data augmentation, use AutoAlbumentation, which promises a good increase in quality
- read and analyze the following papers, from them you can adopt the ideas of preprocessing of satellite images
https://paperswithcode.com/dataset/spacenet-1
https://paperswithcode.com/dataset/spacenet-2
- this dataset to a certain extent can be used for additional labeling and training
https://paperswithcode.com/dataset/cropandweed-dataset
https://paperswithcode.com/dataset/gid
- read and analyze this paper, which solves a similar problem
https://paperswithcode.com/paper/segmentation-of-roots-in-soil-with-u-net