# AI-Explainability-for-Multi-label-Image-Classification
Python code for implementation of Multi-label Image Classification
# OVERVIEW
  We are implementing a DL model which finds out the regions in a Multi-Label Image belonging to repective class.The goal is to achieve multi-label classification and explain what
  the CNN model is seeing as features to classify images in a particular way. We highlight the important portions of image which determine the features of the image. These
  regions are called **SUPERPIXELS**._
  We use a pre-trained model to classify the images. Here, we have used the **INCEPTION V3 MODEL**. We use this model to make the predictions on the pre-processed image and
  extract the top 5 predicted classes. The image is then passed through **skimage.segmentation.quickshift** which groups the pixels with similar properties into **SUPERPIXELS**._
  We then create 150 preturbations of the image with some of the superpixels turned off. The INCEPTION V3 model makes predictions on these preturbed images which are stored for
  later use. A **Linear Regression** model is then fit with **Superpixels** as input. Weights corresponding to each superpixel is determined. The greater the weight, more is the
  importance of superpixel. Image with top 4 features/superpixels turned on(rest are turned off) is displayed depicting the features which lead to the classification of image in
  particular way.

![alt text](https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/pert1.png)
