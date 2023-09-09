
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
# 27/8/23 DH:
import pickle

# 29/8/23 DH:
from lime_utils import *
import inspect

class Lime(object):

  def __init__(self) -> None:
    print("--------------------------------------------------------------------------")
    print('Running: keras ', keras.__version__, "(which is a wrapper around '",keras.backend.backend(),"' backend)" )
    print("--------------------------------------------------------------------------")

    np.random.seed(222)

    # InceptionV3 initialization: use the pre-trained InceptionV3 model available in Keras.
    warnings.filterwarnings('ignore')
    self.inceptionV3_model = keras.applications.inception_v3.InceptionV3()
    self.lime_utils = LimeUtils()

  # --------------------------------- Prelims --------------------------------

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

    # 22/8/23 DH: 'inception_v3' needs 4-D input:
    # ValueError: Input 0 of layer "inception_v3" is incompatible with the layer: 
    # expected shape=(None, 299, 299, 3), found shape=(None, 299, 3)
    img2 = img[np.newaxis,:,:,:]

    self.preds = self.inceptionV3_model.predict(img2)
    print("inceptionV3_model.predict(): ",self.preds.shape)

    self.img = img

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

    print( decode_predictions(self.preds) ) #Top 5 classes (as default)

    """The indexes (positions) of the top 5 classes are saved in the variable `top_pred_classes`"""

    # 22/8/23 DH: https://www.w3schools.com/python/ref_func_slice.asp
    # '[-5:]' ie take last 5 elements
    # '[::-1]' ie reverse order of elements using 'slice([start:stop:step])'
    self.top_pred_classes = self.preds[0].argsort()[-5:][::-1]
    print("\ntop_pred_classes (from 'inceptionV3_model.predict()'):",self.top_pred_classes)
    print("top pred index (from 'inceptionV3_model.predict()'):",self.top_pred_classes[0])
    
    """
    # 3/9/23 DH: Printout prediction class index with associated details:
    
    /Users/doug/.pyenv/versions/3.9.15/lib/python3.9/site-packages/keras/applications/imagenet_utils.py :
      def decode_predictions(preds, top=5):
        ...
        for pred in preds:
          #top_indices = pred.argsort()[-top:][::-1]
          top_indices = pred.argsort()[-1:][::-1]

          result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
          result.sort(key=lambda x: x[2], reverse=True)

          print("decode_predictions(): index",top_indices,"=>",result)

          results.append(result)   
    """

  def segmentImage(self):
    # https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift
    # kernel_size: Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
    # max_dist: ...Higher means fewer clusters
    # ratio: 0-1, Higher values give more weight to color-space.
    #
    # Returns: "Integer mask indicating segment labels."

    #imgSegmentMask = skimage.segmentation.quickshift(self.img, kernel_size=4, max_dist=200, ratio=0.2)

    # 3/9/23 DH: 'kernel_size=6' is min number of segments (28) to correlate with human image detection
    #            ['kernel_size=7' gives 18 segments and leads to unimportant segment choice]
    self.imgSegmentMask = skimage.segmentation.quickshift(self.img, kernel_size=6, max_dist=200, ratio=0.2)
    self.numSegments = np.unique(self.imgSegmentMask).shape[0]
    # 28 segments
    print("\nNumber of segments:",self.numSegments,"(from 'skimage.segmentation.quickshift()' return:",
          self.imgSegmentMask.shape,"for input img:",self.img.shape,")")
    print()

  # --------------------------------- END: Prelims ---------------------------
  
  # ----------------------------- Step 1/4 ---------------------------------

  def createRandomPertubations(self):
    """
    #### Create random perturbations
    In this example, 150 perturbations were used. However, for real life applications, a larger number of 
    perturbations will produce more reliable explanations. Random zeros and ones are generated and shaped as a 
    matrix with perturbations as rows and imgSegmentMask as columns. An example of a perturbation 
    (the first one) is show below. Here, `1` represent that a pixel is on and `0` represents it is off. 
    Notice that the length of the shown vector corresponds to the number of imgSegmentMask in the image.
    """

    # 21/8/23 DH:
    #num_perturb = 150
    self.num_perturb = 100
    self.probSuccess = 0.5

    # 4/9/23 DH: https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
    # "Samples are drawn from a binomial distribution with specified parameters, n trials and p probability 
    #  of success" (therefore "n=1" means that the result will be 0 or 1 necessary for the mask)
    #
    # size: "Output shape. If the given shape is, e.g., (m, n, k), then m * n * k SAMPLES ARE DRAWN."
    #
    self.perturbations = np.random.binomial(1, self.probSuccess, size=(self.num_perturb, self.numSegments))
    
    print("Showing pertubation 0 (from",self.perturbations.shape,"pertubations 2-D array)")
    print(self.perturbations[0]) #Show example of perturbation

  # ----------------------------- Step 2/4 ---------------------------------

  # 29/8/23 DH: Also used in first segment of 'displayTopFeatures()'
  """
  The following function `perturb_image` perturbs the given image (`img`) based on a perturbation vector 
  (`perturbation`) and predefined imgSegmentMask (`segments`).
  """
  def perturb_image(self, img, perturbation, segments):
    print("perturb_image() => img:",img.shape,", pertubation:",perturbation.shape,", segments:",segments.shape)

    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
      mask[segments == active] = 1

    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    return perturbed_image

  # 27/8/23 DH: Pkl the 'predictions' array
  def getPredictions(self):
    predictionsFile = "predictions.pkl"

    try:
      with open(predictionsFile, 'rb') as fp:
        self.predictions = pickle.load(fp)
        print("\nLoaded 'predictions':",self.predictions.shape)
        
        # 28/8/23 DH: https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/decode_predictions
        # Returns:
        #  A list of lists of top class prediction tuples (class_name, class_description, score). 
        #  One list of tuples per sample in batch input. 

        #topPrediction2DTuple = np.array( decode_predictions(preds=self.predictions[0] , top=1) )
        topPredictions2DTuple = np.array( decode_predictions(preds=self.predictions[0]) )
        print("topPredictions2DTuple:",topPredictions2DTuple.shape)

        # Class prediction tuples: 0)class_name, 1)class_description, 2)score
        firstPerturbationMaskIndex = 0
        topPredictionIndex         = 0
        descriptionIndex           = 1
        class_description = topPredictions2DTuple[firstPerturbationMaskIndex][topPredictionIndex][descriptionIndex]

        print(" Top prediction of first perturbation:", self.perturbations[0],"=",class_description)

        class_to_explain = self.top_pred_classes[0]
        yVal = self.predictions[0,:,class_to_explain]
        print(" class_to_explain:",class_to_explain,", yVal:",yVal)
        
        print(" Elements of 'self.predictions[0][0]' around full image prediction:",
              self.predictions[0][0][class_to_explain-2:class_to_explain+2])
        print()
      
    except FileNotFoundError as e:
      self.predictions = []
      pertNum = 0
      # Loop through binomial distrib samples of segment mask inclusion (from 'createRandomPertubations()')
      for pert in self.perturbations:
        perturbed_img = self.perturb_image(self.img, pert, self.imgSegmentMask)
        pertNum += 1
        print("Pertubation",pertNum, "for 'inceptionV3_model.predict()'")

        # **************************************************************************************
        # *** Get a trained 'inceptionV3_model' model prediction for the current pertubation ***
        # **************************************************************************************
        pred = self.inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
        
        self.predictions.append(pred)

      self.predictions = np.array(self.predictions)

      with open(predictionsFile, 'wb') as fp:
        pickle.dump(self.predictions, fp)

    return self.predictions

  # ----------------------------- Step 3/4 ---------------------------------

  def getDistanceWeights(self):
    original_image = np.ones(self.numSegments)[np.newaxis,:] #Perturbation with all imgSegmentMask enabled
    # Binomial distrib sample set of segment mask inclusion (from 'createRandomPertubations()')
    distances = sklearn.metrics.pairwise_distances(X=self.perturbations,Y=original_image, metric='cosine').ravel()
    print("sklearn.metrics.pairwise_distances() of segment mask binomial sample set:",distances.shape)
    sortedDistances = np.argsort(distances)
    print("eg...lowest:", distances[sortedDistances[0]], ", highest:",distances[sortedDistances[-1]], "\n")

    """#### Use kernel function to compute weights
    The distances are then mapped to a value between zero and one (weight) using a kernel function. 

    Here the x axis represents distances and the y axis the weights. Depending on how we set the kernel width, 
    it defines how wide we want the "locality" around our instance to be. This kernel width can be set based on 
    expected distance values. For the case of cosine distances, we expect them to be somehow stable 
    (between 0 and 1); therefore, no fine tunning of the kernel width might be required.
    """

    kernel_width = 0.25

    # 4/9/23 DH: sqrt (e ^(-d^2 / width^2))
    #  Removing the 'sqrt()' made no difference to the Liner Regression coefficient order (ie segment order)
    #self.weights = np.exp( -(distances**2) / kernel_width**2 )
    
    self.weights = np.sqrt( np.exp( -(distances**2) / kernel_width**2 ) )
    
  
  # ----------------------------- Step 4/4 ---------------------------------

  def getLinearRegressionCoefficients(self):
    class_to_explain = self.top_pred_classes[0]
    
    print("-------------------------------------")
    # inspect.stack()
    print("Final step (4/4) in:",inspect.currentframe().f_code.co_name)
    print("                     -------------------------------")
    print(" to correlate InceptionV3 prediction of full image (class index",class_to_explain,")" )
    print(" with masked image predictions, inceptionV3_model.predict()  (in Step 2/4, 'Lime.getPredictions()')")
    print(" using weights obtained for mask distance from full image    (in Step 3/4, 'Lime.getDistanceWeights()')")
    print()

    """
    # 7/9/23 DH: Python slice: 'slice([start:stop:step])'
    
      Slicing with commas is a Numpy (not python) thing, it uses a tuple for indexing...and gets REGEXtastic...!
        https://numpy.org/devdocs/user/basics.indexing.html
    """
    # 7/9/23 DH:
    # 'self.predictions' is 3-D array (the mask and relative value for each of trained 1000 images) 
    # so take the 2-D array that 'self.inceptionV3_model.predict()' returned for each of the perturbed images 
    # ("foreign keyed" by segment mask) for the full image top class index
    yVals = self.predictions[:,:,class_to_explain]

    # 5/9/23 DH: Testing whether complete mask to correlate LinearRegression makes a difference...it does...!
    Xvals = self.perturbations
    #Xvals = self.lime_utils.getMaskForLinearRegression(self.perturbations, yVals, index_start=0)

    print("class_to_explain:",class_to_explain,", from all InceptionV3 trained classes:",self.predictions.shape)
    print("LinearRegression.fit(): mask perturbations:",Xvals.shape,", prediction:",yVals.shape)
    print(" eg...Xvals[0]:", Xvals[0],", yVals[0]:", yVals[0])
    print()

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    simpler_model = LinearRegression(fit_intercept=True)

    """
    /Users/doug/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/linear_model/_base.py :
      def fit():
        ...
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=_normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )
    """

    # 6/9/23 DH: https://en.wikipedia.org/wiki/Linear_regression#Simple_and_multiple_linear_regression
    #simpler_model.fit(X=Xvals, y=yVals, sample_weight=self.weights, debug=True)
    #print("fit() intercept_:",simpler_model.intercept_,", get_params():",simpler_model.get_params())

    simpler_model.fit(X=Xvals, y=yVals, sample_weight=self.weights)

    self.simpler_model = simpler_model
    self.Xvals = Xvals
    self.yVals = yVals
    self.coeff = simpler_model.coef_[0]

    print()
    print("Multiple LinearRegression() coeffs (from weights):",self.coeff.shape)
    sortedCoeffs = np.argsort(self.coeff)
    print(" eg...lowest:",self.coeff[sortedCoeffs[0]],", highest:",self.coeff[sortedCoeffs[-1]])
    print(" (100 pertubation masks for 28 segments leads to a linear correlation line of importance of each",
            "segment for the predicted top ID)")
    print()

  # ----------------------------- END: Step 4/4 ---------------------------------

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
A weighted linear regression model is fitted using data from the previous steps (perturbations, predictions 
and weights). Given that the class that we want to explain is labrador, when fitting the linear model we take 
from the predictions vector only the column corresponding to the top predicted class. Each coefficient in 
the linear model corresponds to one segment in the segmented image. These coefficients represent how 
important is each segment for the prediction of labrador.
"""

# 29/8/23 DH:
if __name__ == '__main__':
  #time.sleep(5)
  #sys.exit(0)

  limeImage = Lime()

  limeImage.getImagePrediction()
  limeImage.getTopPredictions()
  limeImage.segmentImage()

  # Step 1/4
  limeImage.createRandomPertubations()
  limeImage.lime_utils.displayDistrib(limeImage)

  # Step 2/4
  limeImage.getPredictions()
  # Step 3/4
  limeImage.getDistanceWeights()
  # Step 4/4
  limeImage.getLinearRegressionCoefficients()

  # 6/9/23 DH: Plotting regression lines from sequential predicted values is meaningless for a 
  #            Multiple Linear Regression system (of prediction place of full prediction vs 28 bit segment mask)
  #limeImage.lime_utils.displayRegressionLines(limeImage, plot_limit=4)
  #limeImage.lime_utils.displayRegressionLines(limeImage)
  #limeImage.lime_utils.displayRegressionLines(limeImage, model_output=True)

  # 3/9/23 DH:
  limeImage.lime_utils.displayCoefficients(limeImage)

  # 30/8/23 DH: Interface to utils "wrapper" by sending a copy of the Lime object with necessary attribs
  #             ...nicely fractal...
  limeImage.lime_utils.displayTopFeatures(limeImage)
