# AI-Explainability-for-Multi-label-Image-Classification
Python code for implementation of Multi-label Image Classification
# OVERVIEW
  We are implementing a DL model which finds out the regions in a Multi-Label Image belonging to repective class.The goal is to achieve multi-label classification and explain what
  the CNN model is seeing as features to classify images in a particular way. We highlight the important portions of image which determine the features of the image. These
  regions are called **SUPERPIXELS**.<br/>
  We use a pre-trained model to classify the images. Here, we have used the **INCEPTION V3 MODEL**. We use this model to make the predictions on the pre-processed image and
  extract the top 5 predicted classes. The image is then passed through **skimage.segmentation.quickshift** which groups the pixels with similar properties into **SUPERPIXELS**.<br/>
  We then create 150 preturbations of the image with some of the superpixels turned off. The INCEPTION V3 model makes predictions on these preturbed images which are stored for
  later use. A **Linear Regression** model is then fit with **Superpixels** as input. Weights corresponding to each superpixel is determined. The greater the weight, more is the
  importance of superpixel. Image with top 4 features/superpixels turned on(rest are turned off) is displayed depicting the features which lead to the classification of image in
  particular way.

# IMPLEMENTATION
  * **IMPORTING LIBRARIES**
     We import useful libraries like-
      * numpy
      * tensorflow
      * sklearn
      * skimage
      * copy etc
  * **IMPORTING OUR PRE-TRAINED MODEL**
      We import our model by using **tf.keras.applications.InceptionV3()**.
      Later the model attributes are also displayed.
  * **IMPORTING IMAGE AND PREPROCESSING**
      * We import image using **skimage.io.imread**
         * Imported image-
                
              ![alt text](https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/dog%20and%20cat.jpg)
      * The image is pre-processed as requred by INCEPTION model(reshaped and pixel values are changed)
  * **PREDICTION ON IMAGE AND SEGMENTS DISPLAY**  
      * Predictions are made on the preprocessed image and the top 5 predicted classes are stored.
      * Segmented image with superpixel regions is diplayed
         * Segmented image
            ![alt text](https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/segments.png)
  * **MASK CREATION**
      * 150 Random masks are created by using **np.random.binomial**
      * Image pixels are multiplied with masks to create Perturbed images
      <p float="left">
         <img src="https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/pert1.png" width="300" />
         <img src="https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/pert2.png" width="300" /> 
         <img src="https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/pert3.png" width="300" />
      </p>
      
     
