# -*- coding: utf-8 -*-
"""Copy of lime_image.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XSGDpY7lbh1bOU5yOjOHMZ9VER5Va1BN

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
"""

"""
## Initialization

### Imports
Let's import some python utilities for manipulation of images, plotting and numerical analysis.
"""

#%tensorflow_version 1.x
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

print('Notebook running: keras ', keras.__version__)
np.random.seed(222)

"""### InceptionV3 initialization
We are going to use the pre-trained InceptionV3 model available in Keras.
"""

warnings.filterwarnings('ignore')
inceptionV3_model = keras.applications.inception_v3.InceptionV3() #Load pretrained model

"""### Read and pre-process image
The instance to be explained (image) is resized and pre-processed to be suitable for Inception V3. This 
image is saved in the variable `Xi`.
"""

# https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
# "The different color bands/channels are stored in the third dimension, 
#  such that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4."
Xi = skimage.io.imread("https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg")
print("Xi:",Xi.shape)
Xi = skimage.transform.resize(Xi, (299,299))
#skimage.io.imshow(Xi)
#plt.show()

print("Xi:",Xi.shape)
Xi = (Xi - 0.5)*2 #Inception pre-processing
#skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing
#skimage.io.imshow(Xi)
#plt.show()

"""### Predict class of input image
The Inception V3 model is used to predict the class of the image. 
The output of the classification is a vector of 1000 probabilities of belonging to each class 
available in Inception V3. The description of these classes is shown and it can be seen that 
the "Labrador Retriever" is the top class for the given image.
"""

np.random.seed(222)
#preds = inceptionV3_model.predict(Xi[np.newaxis,:,:,:])

# 22/8/23 DH: 'inception_v3' needs 4-D input:
# ValueError: Input 0 of layer "inception_v3" is incompatible with the layer: 
# expected shape=(None, 299, 299, 3), found shape=(None, 299, 3)
Xi_2 = Xi[np.newaxis,:,:,:]
print("Xi_2:",Xi_2.shape)

preds = inceptionV3_model.predict(Xi_2)
print("inceptionV3_model.predict(): ",preds.shape)

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

print(decode_predictions(preds)[0]) #Top 5 classes (as default)

"""The indexes (positions) of the top 5 classes are saved in the variable `top_pred_classes`"""

# 22/8/23 DH: https://www.w3schools.com/python/ref_func_slice.asp
# '[-5:]' ie take last 5 elements
# '[::-1]' ie reverse order of elements using 'slice([start:stop:step])'
top_pred_classes = preds[0].argsort()[-5:][::-1]
print("\ntop_pred_classes (from 'inceptionV3_model.predict()'):",top_pred_classes) #Index of top 5 classes

"""## LIME explanations
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
![alt text](https://arteagac.github.io/blog/lime_image/img/lime_illustration.png)

A detailed explanation of each step is shown below.

### Step 1: Create perturbations of image
For the case of image explanations, perturbations will be generated by turning on and off some of the 
superpixels in the image.

#### Extract super-pixels from image
Superpixels are generated using the quickshift segmentation algorithm. It can be noted that for the given 
image, 68 superpixels were generated. The generated superpixels are shown in the image below.
"""

# https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift
# kernel_size: Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
# max_dist: ...Higher means fewer clusters
# ratio: 0-1, Higher values give more weight to color-space.
#
# Returns: "Integer mask indicating segment labels."

#superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
superpixels = skimage.segmentation.quickshift(Xi, kernel_size=6,max_dist=200, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]
# 71 superpixels
print("\nNumber of superpixels:",num_superpixels,"(from 'skimage.segmentation.quickshift()' return:",
      superpixels.shape,")")
print()

#skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))
#plt.show()

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

perturbations = np.random.binomial(1, probSuccess, size=(num_perturb, num_superpixels))
print("Showing pertubation 0 (from",perturbations.shape,"pertubations 2-D array)")
print(perturbations[0]) #Show example of perturbation

# 24/8/23 DH:
def displayDistrib(distribData, samples, segments, distribSuccess):
  print("\nDisplaying:",distribData.shape)
  #print(distribData.shape[1])

  tally = np.zeros(distribData.shape[1])
  #print("Tally:",tally)
  for distrib in distribData:
    tally += distrib

  #print("Tally (",type(tally),"):",tally)
  print("Tally total:", np.sum(tally), "(for", round(distribSuccess * 100),
        "% chance of segment mask inclusion from", samples * segments, "tests)")

  # 24/8/23 DH: 'ply.hist' from 'real_bday_paradox.py' does not display anything...
  #             ...when use bin number rather than bin values...!!!
  #plt.hist(tally, bins = np.arange(0,39))
  plt.hist(tally, bins = np.arange(0,100))

  # https://matplotlib.org/3.6.2/api/_as_gen/matplotlib.pyplot.bar.html
  #plt.bar(np.arange(0,39), tally)

  plt.title("np.random.binomial(trial=1,prob=0.5,for "+str(samples)+"*"+str(segments)+" events)")
  plt.xlabel("$n$")
  plt.ylabel("Number of occurences")

  plt.show()

  #sys.exit(0)

displayDistrib(perturbations, num_perturb, num_superpixels, probSuccess)

"""The following function `perturb_image` perturbs the given image (`img`) based on a perturbation vector 
(`perturbation`) and predefined superpixels (`segments`)."""

def perturb_image(img, perturbation, segments):
  print("np.where(pertubation == 1):",np.where(perturbation == 1))
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
    mask[segments == active] = 1
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image

# 26/8/23 DH:
def getSegmentMidpoint(prev, segments, cnt):
  midPoint = []

  segPrevArray = (segments == prev)
  # 26/8/23 DH: 'np.where()' returns a 'tuple' (a static list) which needs to get converted into 'ndarray'
  segPrevArrayTrue = np.array(np.where(segPrevArray == True))
  
  """
  print("segments == prev:")
  print(segPrevArray, type(segPrevArray), segPrevArray.shape)
  print("segPrevArray = True", segPrevArrayTrue, type(segPrevArrayTrue), segPrevArrayTrue.shape)

  segPrevArrayFalse = np.array(np.where(segPrevArray == False))
  print("segPrevArray = False", segPrevArrayFalse, type(segPrevArrayFalse), segPrevArrayFalse.shape)
  """
  
  pixelsY = segPrevArrayTrue[0]
  pixelsX = segPrevArrayTrue[1]
  
  pixelsXSorted = np.sort(pixelsX)
  pixelsYSorted = np.sort(pixelsY)

  minX = pixelsXSorted[0]
  maxX = pixelsXSorted[pixelsXSorted.shape[0] - 1]
  midX = round( (minX + maxX) / 2 )

  minY = pixelsYSorted[0]
  maxY = pixelsYSorted[pixelsYSorted.shape[0] - 1]
  midY = round( (minY + maxY) / 2 )

  """
  print("=====================",cnt,"=======================")
  for i in range(pixelsX.shape[0]):
    print(pixelsX[i],",",pixelsY[i])

  print("Min X:",minX)
  print("Max X:",maxX)
  print("Mid X:",midX)

  print("Min Y:",minY)
  print("Max Y:",maxY)
  print("Mid Y:",midY)
  print("===================== END: ",cnt,"=======================")
  """
  
  midPoint.append(midX)
  midPoint.append(midY)

  return midPoint

# 24/8/23 DH:
def highlight_image(img, segMask, currentSegsMask, segments, num_top_features, last_features):
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
    midPoint = getSegmentMidpoint(prev, segments, cnt)
    print("Mid point:",midPoint)

    # Add horizontal line at mid-point of segment (visible across all non-black segments)
    #mask[midPoint[0]] = 1

    (digitLabelSizeX, digitLabelSizeY) = digitLabelsDict[num_top_features-cnt].shape

    # 28/8/23 DH: *** NOTE: The image 'y' axis is the 2-D array 'x' index ***
    for x in range(digitLabelSizeX):
      for y in range(digitLabelSizeY):
        mask[midPoint[1]+x][midPoint[0]+y] = digitLabelsDict[num_top_features-cnt][x][y]

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

"""
#Let's use the previous function to see what a perturbed image would look like:

print("Showing the result of pertub_image() for pertubations[0]")
skimage.io.imshow( perturb_image(Xi/2+0.5, perturbations[0], superpixels) )
plt.show()
"""

"""### Step 2: Use ML classifier to predict classes of new generated images
This is the most computationally expensive step in LIME because a prediction for each perturbed image is 
computed. From the shape of the predictions we can see for each of the perturbations we have the output 
probability for each of the 1000 classes in Inception V3.
"""

# 27/8/23 DH: Pkl the 'predictions' array
def getPredictions():
  predictionsFile = "predictions.pkl"

  try:
    with open(predictionsFile, 'rb') as fp:
      predictions = pickle.load(fp)
      print("\nLoaded 'predictions':",predictions.shape)

      #print(decode_predictions(preds)[0]) #Top 5 classes (as default)
      #top_pred_classes = preds[0].argsort()[-5:][::-1]
      
      # 28/8/23 DH: https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/decode_predictions
      # Returns:
      #  A list of lists of top class prediction tuples (class_name, class_description, score). 
      #  One list of tuples per sample in batch input. 
      topPrediction = decode_predictions(preds=predictions[0].argsort()[-1:][::-1], top=1)[0][0]

      print("Top prediction of first prediction:", topPrediction[1])
      print()
    
  except FileNotFoundError as e:
    predictions = []
    pertNum = 0
    for pert in perturbations:
      perturbed_img = perturb_image(Xi,pert,superpixels)
      pertNum += 1
      print("Pertubation",pertNum, "for 'inceptionV3_model.predict()'")
      # Get a trained 'inceptionV3_model' model prediction for the current pertubation
      pred = inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
      predictions.append(pred)

    predictions = np.array(predictions)
    #predictions.shape

    with open(predictionsFile, 'wb') as fp:
      pickle.dump(predictions, fp)

  return predictions

predictions = getPredictions()

"""### Step 3: Compute distances between the original image and each of the perturbed images and compute 
weights (importance) of each perturbed image.
The distance between each randomly generated perturnation and the image being explained is computed using 
the cosine distance. For the shape of the `distances` array it can be noted that, as expected, there is a 
distance for every generated perturbation.
"""

original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
distances.shape

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
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
weights.shape

"""### Step 4: Use `perturbations`, `predictions` and `weights` to fit an explainable (linear) model
A weighed linear regression model is fitted using data from the previous steps (perturbations, predictions 
and weights). Given that the class that we want to explain is labrador, when fitting the linear model we take 
from the predictions vector only the column corresponding to the top predicted class. Each coefficients in 
the linear model corresponds to one superpixel in the segmented image. These coefficients represent how 
important is each superpixel for the prediction of labrador.
"""

class_to_explain = top_pred_classes[0]
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]

print("coeff:",coeff)
print()
print("np.argsort(coeff):",np.argsort(coeff))

"""#### Compute top features (superpixels)
Now we just need to sort the coefficients to figure out which are the supperpixels that have larger 
coefficients (magnitude) for the prediction of labradors. The identifiers of these top features or 
superpixels are shown below. Even though here we use the magnitude of the coefficients to determine the 
most important features, other alternatives such as forward or backward elimination can be used for feature 
importance selection.
"""

"""
while float(self.accuracyPercent) < 0.90:
  self.desiredIncrease += 0.05
  self.rlRunPart(rl)
"""

# 26/8/23 DH: 
def createDigitLabels():
  
  digitLabelsDict = {}

  """
  # 28/8/23 DH: '0' is black pixel
  digitLabelsDict[0] = np.asarray([
                          [1,1,1,1,1,1],
                          [1,1,0,0,1,1],
                          [1,0,1,1,0,1],
                          [1,0,1,1,0,1],
                          [1,0,1,1,0,1],
                          [1,1,0,0,1,1],
                          [1,1,1,1,1,1]])
  
  digitLabelsDict[4] = np.asarray([
                              [  1,  1,  1,  1,  1,  1],
                              [  1,  1,  1,  0,  0,  1],
                              [  1,  1,  0,  1,  0,  1],
                              [  1,  0,  1,  1,  0,  1],
                              [  0,  1,  1,  1,  0,  1],
                              [  0,  0,  0,  0,  0,  0],
                              [  1,  1,  1,  0,  0,  1],
                              [  1,  1,  1,  0,  0,  1],
                              [  1,  1,  1,  1,  1,  1]
                              ])
  """

  # 28/8/23 DH: --------------- Taken from 'mnist-training-errors/tf_config_image.py' ----------------
  # 6 * 7 = 42 pixels
  
  digitLabelsDict[0] = np.asarray(
                            [[  0,  0,255,255,  0,  0],
                              [  0,255,255,255,255,  0],
                              [255,255,  0,  0,255,255],
                              [255,  0,  0,  0,  0,255],
                              [255,255,  0,  0,255,255],
                              [  0,255,255,255,255,  0],
                              [  0,  0,255,255,  0,  0]])

  digitLabelsDict[1] = np.asarray(
                            [[  0,255,255,255,  0,  0],
                              [  0,255,255,255,  0,  0],
                              [  0,  0,  0,255,  0,  0],
                              [  0,  0,  0,255,  0,  0],
                              [  0,  0,  0,255,  0,  0],
                              [  0,255,255,255,255,  0],
                              [  0,255,255,255,255,  0]])
  
  digitLabelsDict[2] = np.asarray(
                            [[  0,  0,255,255,  0,  0],
                              [  0,255,  0,255,255,  0],
                              [  0,  0,  0,  0,255,  0],
                              [  0,  0,  0,  0,255,  0],
                              [  0,  0,  0,255,255,  0],
                              [  0,  0,255,255,  0,  0],
                              [  0,255,255,255,255,255]])
  
  digitLabelsDict[3] = np.asarray(
                            [[  0,  0,255,255,255,  0],
                              [  0,255,  0,  0,255,255],
                              [  0,  0,  0,  0,  0,255],
                              [  0,  0,  0,255,255,  0],
                              [  0,  0,  0,  0,  0,255],
                              [  0,255,  0,  0,255,255],
                              [  0,  0,255,255,255,  0]])
  
  digitLabelsDict[4] = np.asarray(
                            [[  0,  0,  0,255,255,  0],
                              [  0,  0,255,  0,255,  0],
                              [  0,255,  0,  0,255,  0],
                              [255,  0,  0,  0,255,  0],
                              [255,255,255,255,255,255],
                              [  0,  0,  0,  0,255,  0],
                              [  0,  0,  0,  0,255,  0]])

  digitLabelsDict[5] = np.asarray(
                            [[255,255,255,255,  0,  0],
                              [255,  0,  0,  0,  0,  0],
                              [255,255,255,255,  0,  0],
                              [  0,  0,  0,255,255,  0],
                              [  0,  0,  0,  0,255,  0],
                              [255,  0,  0,255,255,  0],
                              [255,255,255,255,  0,  0]])

  digitLabelsDict[6] = np.asarray(
                            [[  0,  0,255,255,  0,  0],
                              [  0,255,  0,  0,  0,  0],
                              [  0,255,  0,  0,  0,  0],
                              [  0,255,255,255,255,  0],
                              [  0,255,  0,  0,255,  0],
                              [  0,255,  0,  0,255,  0],
                              [  0,  0,255,255,255,  0]])

  digitLabelsDict[7] = np.asarray(
                            [[255,255,255,255,255,255],
                              [255,255,255,255,255,255],
                              [  0,  0,  0,  0,255,255],
                              [  0,  0,  0,255,255,  0],
                              [  0,  0,255,255,  0,  0],
                              [  0,255,255,  0,  0,  0],
                              [255,255,  0,  0,  0,  0]])
  
  digitLabelsDict[8] = np.asarray(
                            [[  0,  0,255,255,255,  0],
                              [  0,255,255,  0,255,255],
                              [  0,255,  0,  0,  0,255],
                              [  0,  0,255,255,255,  0],
                              [  0,255,  0,  0,  0,255],
                              [  0,255,255,  0,255,255],
                              [  0,  0,255,255,255,  0]])

  digitLabelsDict[9] = np.asarray(
                            [[  0,255,255,255,  0,  0],
                              [  0,255,  0,  0,255,  0],
                              [  0,255,  0,  0,255,  0],
                              [  0,  0,255,255,255,  0],
                              [  0,  0,  0,  0,255,  0],
                              [  0,  0,  0,  0,255,  0],
                              [  0,  0,  0,  0,255,  0]])
  
  return digitLabelsDict

  
num_top_featureS = num_top_feature = 4

# 24/8/23 DH:
last_features = -1
digitLabelsDict = createDigitLabels()

while num_top_feature > 0:
  # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
  # "It returns an array of indices ... in sorted order."
  # "kind=None: Sorting algorithm. The default is ‘quicksort’."
  top_features = np.argsort(coeff)[-num_top_feature:]
  print("\ntop_features:",top_features)

  currentSegmentsMask = np.zeros(num_superpixels)
  lastSegmentMask = np.zeros(num_superpixels)

  currentSegmentsMask[top_features] = True #Activate top superpixels
  print("Showing feature number ",num_top_feature)

  if last_features != -1:
    print("last_feature: ",last_features)
    lastSegmentMask[last_features] = True

    # def highlight_image(img, segMask, currentSegsMask, segments, featureNum):
    img2 = highlight_image(img, lastSegmentMask, currentSegmentsMask, superpixels, 
                           num_top_featureS, last_features)

    last_features.append(top_features[0])
  else: # first time
    last_features = []
    last_features.append(top_features[0])
    img = perturb_image(Xi/2+0.5,currentSegmentsMask,superpixels)
    img2 = img

  img3 = skimage.segmentation.mark_boundaries(img2, superpixels)
  skimage.io.imshow( img3 )
  plt.show()

  num_top_feature -= 1

#time.sleep(5)