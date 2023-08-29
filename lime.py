
"""
# Interpretable Machine Learning with LIME for Image Classification
By: Cristian Arteaga, [arteagac.github.io](https://arteagac.github.io)

In this post, we will study how LIME  (Local Interpretable Model-agnostic Explanations) 
([Ribeiro et. al. 2016](https://arxiv.org/abs/1602.04938)) generates explanations for image classification 
tasks. The basic idea is to understand why a machine learning model (deep neural network) predicts that an 
instance (image) belongs to a certain class (labrador in this case). For an introductory guide about how LIME 
works, I recommend you to check my previous blog post [Interpretable Machine Learning with LIME]
(https://nbviewer.jupyter.org/urls/arteagac.github.io/blog/lime.ipynb).

Also, the following YouTube video explains this notebook step by step:
https://www.youtube.com/embed/ENa-w65P1xM

## LIME explanation:
The following figure illustrates the basic idea behind LIME. 

The figure shows light and dark gray areas which are the decision boundaries for the classes 
for each (x1,x2) pairs in the dataset. LIME is able to provide explanations for the predictions of an 
individual record (blue dot). The  explanations are created by generating a new dataset of perturbations 
around the instance to be explained (colored markers around the blue dot). 

The output or class of each generated perturbation is predicted with the machine-learning model 
(colored markers inside and outside the decision boundaries). The importance of each perturbation 
is determined by measuring its distance from the original instance to be explained. 

These distances are converted to weights by mapping the distances to a zero-one scale using a 
kernel function (see color scale for the weights). All this information: the new generated dataset, 
its class predictions and its weights are used to fit a simpler model, such as a linear model (blue line), 
that can be interpreted. The attributes of the simpler model, coefficients for the case of a linear model, 
are then used to generate explanations.  
(https://arteagac.github.io/blog/lime_image/img/lime_illustration.png)

A detailed explanation of each step is shown below.

"""

import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings

# 19/8/23 DH:
import time, sys
import matplotlib.pyplot as plt
# 27/8/23 DH:
import pickle

# 29/8/23 DH:
from lime_utils import *

class Lime(object):

  def __init__(self) -> None:
    print("-------------------------------------")
    #print("TensorFlow version:", tf.__version__)
    print('Notebook running: keras ', keras.__version__)
    np.random.seed(222)
    print("-------------------------------------")

    # InceptionV3 initialization: use the pre-trained InceptionV3 model available in Keras.
    warnings.filterwarnings('ignore')
    self.inceptionV3_model = keras.applications.inception_v3.InceptionV3()
    self.lime_utils = LimeUtils()

  def getImagePrediction(self):
    # image is resized and pre-processed to be suitable for Inception V3.

    # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
    # "The different color bands/channels are stored in the third dimension, 
    #  such that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4."
    img = skimage.io.imread("https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg")

    img = skimage.transform.resize(img, (299,299))
    img = (img - 0.5)*2 #Inception pre-processing

    """### Predict class of input image
    The Inception V3 model is used to predict the class of the image. 
    The output of the classification is a vector of 1000 probabilities of belonging to each class 
    available in Inception V3. The description of these classes is shown and it can be seen that 
    the "Labrador Retriever" is the top class for the given image.
    """

    np.random.seed(222)
    print("\n'np.random.seed(222)' ???\n")
    #preds = inceptionV3_model.predict(Xi[np.newaxis,:,:,:])

    # 22/8/23 DH: 'inception_v3' needs 4-D input:
    # ValueError: Input 0 of layer "inception_v3" is incompatible with the layer: 
    # expected shape=(None, 299, 299, 3), found shape=(None, 299, 3)
    img2 = img[np.newaxis,:,:,:]

    self.preds = self.inceptionV3_model.predict(img2)
    print("inceptionV3_model.predict(): ",self.preds.shape)

    self.img = img

    print("\nDEBUG")
    self.superpixels = skimage.segmentation.quickshift(self.img, kernel_size=6,max_dist=200, ratio=0.2)

  def getTopPredictions(self):
    """
    22/8/23 DH:

    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/decode_predictions
    # Returns: "A list of lists of top class prediction tuples (class_name, class_description, score)"
    #
    # Added above: "from keras.applications.imagenet_utils import decode_predictions"

    /Users/doug/.pyenv/versions/3.9.15/lib/python3.9/site-packages/keras/applications/inception_v3.py :

    @keras_export("keras.applications.inception_v3.decode_predictions")
    def decode_predictions(preds, top=5):
      return imagenet_utils.decode_predictions(preds, top=top)
    """

    print(decode_predictions(self.preds)[0]) #Top 5 classes (as default)

    """The indexes (positions) of the top 5 classes are saved in the variable `top_pred_classes`"""

    # 22/8/23 DH: https://www.w3schools.com/python/ref_func_slice.asp
    # '[-5:]' ie take last 5 elements
    # '[::-1]' ie reverse order of elements using 'slice([start:stop:step])'
    self.top_pred_classes = self.preds[0].argsort()[-5:][::-1]
    print("\ntop_pred_classes (from 'inceptionV3_model.predict()'):",self.top_pred_classes) #Index of top 5 classes

  """The following function `perturb_image` perturbs the given image (`img`) based on a perturbation vector 
  (`perturbation`) and predefined superpixels (`segments`)."""

  def perturb_image(self, img, perturbation, segments):
    print("np.where(pertubation == 1):",np.where(perturbation == 1))
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
      mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    return perturbed_image
  
  # 24/8/23 DH:
  def highlight_image(self, img, segMask, currentSegsMask, segments, num_top_features, last_features):
    print("---------------- highlight_image() -------------------")
    print("np.where(segMask == 1):",np.where(segMask == 1))
    prev_active_pixels = np.where(segMask == 1)[0]
    print("np.where(currentSegsMask == 1):",np.where(currentSegsMask == 1))
    active_pixels = np.where(currentSegsMask == 1)[0]
    
    mask = np.zeros(segments.shape)
    for active in active_pixels:
      mask[segments == active] = 1

    print()
    print("prev_active_pixels (in index NOT TOP FEATURE ORDER):",prev_active_pixels)
    print("last_features:",last_features)
    cnt = 0

    #for prev in prev_active_pixels:
    # 28/8/23 DH: 'prev_active_pixels' was just previous segment top feature in index order
    for prev in last_features:
      # 'mask[segments == prev] = 0' makes segment black
      mask[segments == prev] = 0.5 # half transparent

      # 26/8/23 DH: 
      midPoint = self.lime_utils.getSegmentMidpoint(prev, segments)
      print("Mid point:",midPoint)

      # Add horizontal line at mid-point of segment (visible across all non-black segments)
      #mask[midPoint[0]] = 1

      (digitLabelSizeX, digitLabelSizeY) = self.lime_utils.digitLabelsDict[num_top_features-cnt].shape

      # 28/8/23 DH: *** NOTE: The image 'y' axis is the 2-D array 'x' index ***
      for x in range(digitLabelSizeX):
        for y in range(digitLabelSizeY):
          mask[midPoint[1]+x][midPoint[0]+y] = self.lime_utils.digitLabelsDict[num_top_features-cnt][x][y]

          #if x == (digitLabelSizeX -1) and y == (digitLabelSizeY -1):
          #  print("x:",x,",y:",y,"=",digitLabelsDict[featureNum][x][y])

      cnt += 1

    highlighted_image = copy.deepcopy(img)
    print("highlighted_image:", type(highlighted_image), highlighted_image.shape)
    print("mask/segments:", type(mask), mask.shape)
    print("segMask:", type(segMask), segMask.shape)
    print("currentSegsMask:", type(currentSegsMask), currentSegsMask.shape)
    print("---------------- END: highlight_image() -------------------")

    # 26/8/23 DH: [start:stop:step], 
    #             [:,:,np.newaxis], 'np.newaxis' = "add another layer" so make 'mask' 3D like 'highlighted_image'
    highlighted_image = highlighted_image * mask[:,:,np.newaxis]
    
    return highlighted_image

  def segmentImage(self):
    # https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift
    # kernel_size: Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
    # max_dist: ...Higher means fewer clusters
    # ratio: 0-1, Higher values give more weight to color-space.
    #
    # Returns: "Integer mask indicating segment labels."

    #superpixels = skimage.segmentation.quickshift(self.img, kernel_size=4,max_dist=200, ratio=0.2)

    self.superpixels = skimage.segmentation.quickshift(self.img, kernel_size=6,max_dist=200, ratio=0.2)
    self.num_superpixels = np.unique(self.superpixels).shape[0]
    # 71 superpixels
    print("\nNumber of superpixels:",self.num_superpixels,"(from 'skimage.segmentation.quickshift()' return:",
          self.superpixels.shape,")")
    print()

    #skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))
    #plt.show()

  def createRandomPertubations(self):
    """
    #### Create random perturbations
    In this example, 150 perturbations were used. However, for real life applications, a larger number of 
    perturbations will produce more reliable explanations. Random zeros and ones are generated and shaped as a 
    matrix with perturbations as rows and superpixels as columns. An example of a perturbation (the first one) 
    is show below. Here, `1` represent that a superpixel is on and `0` represents it is off. Notice that the 
    length of the shown vector corresponds to the number of superpixels in the image.
    """

    # 21/8/23 DH:
    #num_perturb = 150
    num_perturb = 100
    probSuccess = 0.5

    self.perturbations = np.random.binomial(1, probSuccess, size=(num_perturb, self.num_superpixels))
    print("Showing pertubation 0 (from",self.perturbations.shape,"pertubations 2-D array)")
    print(self.perturbations[0]) #Show example of perturbation

  # 27/8/23 DH: Pkl the 'predictions' array
  def getPredictions(self):
    predictionsFile = "predictions.pkl"

    try:
      with open(predictionsFile, 'rb') as fp:
        self.predictions = pickle.load(fp)
        print("\nLoaded 'predictions':",self.predictions.shape)

        #print(decode_predictions(preds)[0]) #Top 5 classes (as default)
        #top_pred_classes = preds[0].argsort()[-5:][::-1]
        
        # 28/8/23 DH: https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/decode_predictions
        # Returns:
        #  A list of lists of top class prediction tuples (class_name, class_description, score). 
        #  One list of tuples per sample in batch input. 
        topPrediction = decode_predictions(preds=self.predictions[0].argsort()[-1:][::-1], top=1)[0][0]

        print("Top prediction of first prediction:", topPrediction[1])
        print()
      
    except FileNotFoundError as e:
      self.predictions = []
      pertNum = 0
      for pert in self.perturbations:
        perturbed_img = self.perturb_image(self.img, pert, self.superpixels)
        pertNum += 1
        print("Pertubation",pertNum, "for 'inceptionV3_model.predict()'")
        # Get a trained 'inceptionV3_model' model prediction for the current pertubation
        pred = self.inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
        self.predictions.append(pred)

      self.predictions = np.array(self.predictions)
      #predictions.shape

      with open(predictionsFile, 'wb') as fp:
        pickle.dump(self.predictions, fp)

    return self.predictions

  def getDistanceWeights(self):
    original_image = np.ones(self.num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
    distances = sklearn.metrics.pairwise_distances(self.perturbations,original_image, metric='cosine').ravel()

    """#### Use kernel function to compute weights
    The distances are then mapped to a value between zero and one (weight) using a kernel function. 
    An example of a kernel function with different kernel widths is shown in the plot below. 

    Here the x axis represents distances and the y axis the weights. Depeding on how we set the kernel width, 
    it defines how wide we want the "locality" around our instance to be. This kernel width can be set based on 
    expected distance values. For the case of cosine distances, we expect them to be somehow stable 
    (between 0 and 1); therefore, no fine tunning of the kernel width might be required.

    <img src="https://arteagac.github.io/blog/lime_image/img/kernel.png" alt="Drawing" width="600"/>
    """

    kernel_width = 0.25
    self.weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    
  def getLinearRegressionCoefficients(self):
    class_to_explain = self.top_pred_classes[0]
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    simpler_model = LinearRegression()
    simpler_model.fit(X=self.perturbations, y=self.predictions[:,:,class_to_explain], sample_weight=self.weights)
    
    self.coeff = simpler_model.coef_[0]

    print("coeff:",self.coeff)
    print()
    print("np.argsort(coeff):",np.argsort(self.coeff))

  def displayTopFeatures(self):
    """#### Compute top features (superpixels)
    Now we just need to sort the coefficients to figure out which are the supperpixels that have larger 
    coefficients (magnitude) for the prediction of labradors. The identifiers of these top features or 
    superpixels are shown below. Even though here we use the magnitude of the coefficients to determine the 
    most important features, other alternatives such as forward or backward elimination can be used for feature 
    importance selection.
    """

    num_top_featureS = num_top_feature = 4

    # 24/8/23 DH:
    last_features = -1
    self.lime_utils.createDigitLabels()

    while num_top_feature > 0:
      # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
      # "It returns an array of indices ... in sorted order."
      # "kind=None: Sorting algorithm. The default is ‘quicksort’."
      top_features = np.argsort(self.coeff)[-num_top_feature:]
      print("\ntop_features:",top_features)

      currentSegmentsMask = np.zeros(self.num_superpixels)
      lastSegmentMask = np.zeros(self.num_superpixels)

      currentSegmentsMask[top_features] = True #Activate top superpixels
      print("Showing feature number ",num_top_feature)

      if last_features != -1:
        print("last_feature: ",last_features)
        lastSegmentMask[last_features] = True

        # def highlight_image(img, segMask, currentSegsMask, segments, featureNum):
        img2 = self.highlight_image(img, lastSegmentMask, currentSegmentsMask, self.superpixels, 
                              num_top_featureS, last_features)

        last_features.append(top_features[0])
      else: # first time
        last_features = []
        last_features.append(top_features[0])
        img = self.perturb_image(self.img/2+0.5,currentSegmentsMask,self.superpixels)
        img2 = img

      img3 = skimage.segmentation.mark_boundaries(img2, self.superpixels)
      skimage.io.imshow( img3 )
      plt.show()

      num_top_feature -= 1


""" ============================================ STEP 1/4 ==================================================
### Step 1: Create perturbations of image
For the case of image explanations, perturbations will be generated by turning on and off some of the 
segments in the image.
"""

""" ============================================ STEP 2/4 ==================================================
### Step 2: Use ML classifier to predict classes of new generated images
This is the most computationally expensive step in LIME because a prediction for each perturbed image is 
computed. From the shape of the predictions we can see for each of the perturbations we have the output 
probability for each of the 1000 classes in Inception V3.
"""

""" ============================================ STEP 3/4 ==================================================
### Step 3: Compute distances between the original image and each of the perturbed images and compute 
weights (importance) of each perturbed image.
The distance between each randomly generated perturbation and the image being explained is computed using 
the cosine distance. For the shape of the `distances` array it can be noted that, as expected, there is a 
distance for every generated perturbation.
"""

""" ============================================ STEP 4/4 ==================================================
### Step 4: Use `perturbations`, `predictions` and `weights` to fit an explainable (linear) model
A weighed linear regression model is fitted using data from the previous steps (perturbations, predictions 
and weights). Given that the class that we want to explain is labrador, when fitting the linear model we take 
from the predictions vector only the column corresponding to the top predicted class. Each coefficients in 
the linear model corresponds to one superpixel in the segmented image. These coefficients represent how 
important is each superpixel for the prediction of labrador.
"""

# 29/8/23 DH:
if __name__ == '__main__':
  #time.sleep(5)

  limeImage = Lime()

  limeImage.getImagePrediction()
  limeImage.getTopPredictions()

  #limeImage.lime_utils.displayDistrib(perturbations, num_perturb, num_superpixels, probSuccess)
  
  limeImage.segmentImage()
  limeImage.createRandomPertubations()
  limeImage.getPredictions()
  limeImage.getDistanceWeights()
  limeImage.getLinearRegressionCoefficients()

  limeImage.displayTopFeatures()
