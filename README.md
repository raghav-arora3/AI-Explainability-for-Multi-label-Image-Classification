# AI-Explainability-for-Multi-label-Image-Classification
Python code for implementation of Multi-label Image Classification
# OVERVIEW
  We are implementing a DL model which finds out the regions in a Multi-Label Image belonging to respective class. The goal is to achieve multi-label classification and explain what
  the CNN model sees as features to classify images in a particular way. We highlight the essential portions of the image which determine the features of the image. These
  regions are called **SUPERPIXELS**.<br/>
  We use a pre-trained model to classify the images. Here, we have used the **INCEPTION V3 MODEL**. We use this model to make the predictions on the pre-processed image and
  extract the top 5 predicted classes. The image is then passed through **skimage.segmentation.quickshift** which groups the pixels with similar properties into **SUPERPIXELS**.<br/>
  We then create 150 perturbations of the image with some of the superpixels turned off. The INCEPTION V3 model makes predictions on these perturbed images, which are stored for
  later use. A **Linear Regression** model is then fit with **Superpixels** as input. Weights corresponding to each superpixel is determined. The greater the weight, more is the
  importance of superpixel. Image with top 4 features/superpixels turned on(rest are turned off) is displayed depicting the features which lead to the classification of image in
  a particular way.

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
      * Segmented image with superpixel regions is displayed
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
      
 *  **PREDICTIONS ON PERTURBED IMAGE**
      * We again use our pre-trained INCEPTION V3 MODEL to make predictions on the perturbed images. These predictions are stored in an array.
   
 *  **COMPUTING DISTANCE**
      * We then calculate the distances between the generated images and the original image. For this we use **Sklearn.metrics.pairwise_distances**.
 *  **FITTING REGRESSION MODEL**
      * We fit a linear regression model using predictions, perturbations and weights for the top prediction classes to be explained.
 *  **RESULTS**
      * The superpixels having greater weights would be of more importance. We display the final image with top 4 superpixels turned on.
      
      ![](https://raw.githubusercontent.com/raghav-arora3/AI-Explainability-for-Multi-label-Image-Classification/main/output.png)
      
 # REFERENCES
    [skimage.segmentation.quickshift] (https://towardsdatascience.com/classify-any-object-using-pre-trained-cnn-model-77437d61e05f)
    [trial link](https://www.google.com/search?q=markdown+cheat+sheet&oq=mark&aqs=chrome.0.69i59j69i57j46i433j69i60l3j69i65j69i60.861j0j7&sourceid=chrome&ie=UTF-8)
     
