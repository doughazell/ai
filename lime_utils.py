import matplotlib.pyplot as plt
import numpy as np

# 30/8/23 DH:
import copy
import skimage.io
import skimage.segmentation
from lime import *

class LimeUtils(object):

  def __init__(self) -> None:
    pass

  # 26/8/23 DH: 
  def createDigitLabels(self):
    
    self.digitLabelsDict = {}

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
    
    self.digitLabelsDict[0] = np.asarray(
                              [[  0,  0,255,255,  0,  0],
                                [  0,255,255,255,255,  0],
                                [255,255,  0,  0,255,255],
                                [255,  0,  0,  0,  0,255],
                                [255,255,  0,  0,255,255],
                                [  0,255,255,255,255,  0],
                                [  0,  0,255,255,  0,  0]])

    self.digitLabelsDict[1] = np.asarray(
                              [[  0,255,255,255,  0,  0],
                                [  0,255,255,255,  0,  0],
                                [  0,  0,  0,255,  0,  0],
                                [  0,  0,  0,255,  0,  0],
                                [  0,  0,  0,255,  0,  0],
                                [  0,255,255,255,255,  0],
                                [  0,255,255,255,255,  0]])
    
    self.digitLabelsDict[2] = np.asarray(
                              [[  0,  0,255,255,  0,  0],
                                [  0,255,  0,255,255,  0],
                                [  0,  0,  0,  0,255,  0],
                                [  0,  0,  0,  0,255,  0],
                                [  0,  0,  0,255,255,  0],
                                [  0,  0,255,255,  0,  0],
                                [  0,255,255,255,255,255]])
    
    self.digitLabelsDict[3] = np.asarray(
                              [[  0,  0,255,255,255,  0],
                                [  0,255,  0,  0,255,255],
                                [  0,  0,  0,  0,  0,255],
                                [  0,  0,  0,255,255,  0],
                                [  0,  0,  0,  0,  0,255],
                                [  0,255,  0,  0,255,255],
                                [  0,  0,255,255,255,  0]])
    
    self.digitLabelsDict[4] = np.asarray(
                              [[  0,  0,  0,255,255,  0],
                                [  0,  0,255,  0,255,  0],
                                [  0,255,  0,  0,255,  0],
                                [255,  0,  0,  0,255,  0],
                                [255,255,255,255,255,255],
                                [  0,  0,  0,  0,255,  0],
                                [  0,  0,  0,  0,255,  0]])

    self.digitLabelsDict[5] = np.asarray(
                              [[255,255,255,255,  0,  0],
                                [255,  0,  0,  0,  0,  0],
                                [255,255,255,255,  0,  0],
                                [  0,  0,  0,255,255,  0],
                                [  0,  0,  0,  0,255,  0],
                                [255,  0,  0,255,255,  0],
                                [255,255,255,255,  0,  0]])

    self.digitLabelsDict[6] = np.asarray(
                              [[  0,  0,255,255,  0,  0],
                                [  0,255,  0,  0,  0,  0],
                                [  0,255,  0,  0,  0,  0],
                                [  0,255,255,255,255,  0],
                                [  0,255,  0,  0,255,  0],
                                [  0,255,  0,  0,255,  0],
                                [  0,  0,255,255,255,  0]])

    self.digitLabelsDict[7] = np.asarray(
                              [[255,255,255,255,255,255],
                                [255,255,255,255,255,255],
                                [  0,  0,  0,  0,255,255],
                                [  0,  0,  0,255,255,  0],
                                [  0,  0,255,255,  0,  0],
                                [  0,255,255,  0,  0,  0],
                                [255,255,  0,  0,  0,  0]])
    
    self.digitLabelsDict[8] = np.asarray(
                              [[  0,  0,255,255,255,  0],
                                [  0,255,255,  0,255,255],
                                [  0,255,  0,  0,  0,255],
                                [  0,  0,255,255,255,  0],
                                [  0,255,  0,  0,  0,255],
                                [  0,255,255,  0,255,255],
                                [  0,  0,255,255,255,  0]])

    self.digitLabelsDict[9] = np.asarray(
                              [[  0,255,255,255,  0,  0],
                                [  0,255,  0,  0,255,  0],
                                [  0,255,  0,  0,255,  0],
                                [  0,  0,255,255,255,  0],
                                [  0,  0,  0,  0,255,  0],
                                [  0,  0,  0,  0,255,  0],
                                [  0,  0,  0,  0,255,  0]])
    
    return self.digitLabelsDict


  # 24/8/23 DH: 
  # 2/9/23 DH: limeImage.perturbations, limeImage.num_perturb, limeImage.numSegments, limeImage.probSuccess
  #            (distribData)             (samples)              (segments)             (distribSuccess)
  def displayDistrib(self, limeImage):
    print("\nDisplaying:",limeImage.perturbations.shape)
    #print(distribData.shape[1])

    tally = np.zeros(limeImage.perturbations.shape[1])
    #print("Tally:",tally)
    for distrib in limeImage.perturbations:
      tally += distrib

    #print("Tally (",type(tally),"):",tally)
    print("Tally total:", np.sum(tally), "(for", round(limeImage.probSuccess * 100),
          "% chance of segment mask inclusion from", limeImage.num_perturb * limeImage.numSegments, "tests)")

    # 24/8/23 DH: 'ply.hist' from 'real_bday_paradox.py' does not display anything...
    #             ...when use bin number rather than bin values...!!!
    #plt.hist(tally, bins = np.arange(0,39))
    plt.hist(tally, bins = np.arange(0,100))

    # https://matplotlib.org/3.6.2/api/_as_gen/matplotlib.pyplot.bar.html
    #plt.bar(np.arange(0,39), tally)

    plt.title("np.random.binomial(trial=1,prob=0.5,for " + str(limeImage.num_perturb) + "*"
              + str(limeImage.numSegments) + " events)")
    plt.xlabel("$n$")
    plt.ylabel("Number of occurences")

    plt.show()

    #sys.exit(0)

  # 26/8/23 DH:
  def getSegmentMidpoint(self, prev, segments):
    midPoint = []

    segPrevArray = (segments == prev)
    # 26/8/23 DH: 'np.where()' returns a 'tuple' (a static list) which needs to get converted into 'ndarray'
    segPrevArrayTrue = np.array(np.where(segPrevArray == True))
    
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
    
    midPoint.append(midX)
    midPoint.append(midY)

    return midPoint
  
  # 24/8/23 DH:
  def highlight_image(self, img, currentSegsMask, segments, num_top_features, last_features):
    
    active_pixels = np.where(currentSegsMask == 1)[0]
    
    mask = np.zeros(segments.shape)
    for active in active_pixels:
      mask[segments == active] = 1

    cnt = 0

    # 28/8/23 DH: 'prev_active_pixels' was just previous segment top feature in index order
    for prev in last_features:
      # 'mask[segments == prev] = 0' makes segment black
      mask[segments == prev] = 0.5 # half transparent

      # 26/8/23 DH: 
      midPoint = self.getSegmentMidpoint(prev, segments)

      # Add horizontal line at mid-point of segment (visible across all non-black segments)
      #mask[midPoint[0]] = 1

      (digitLabelSizeX, digitLabelSizeY) = self.digitLabelsDict[num_top_features-cnt].shape

      # 28/8/23 DH: *** NOTE: The image 'y' axis is the 2-D array 'x' index ***
      for x in range(digitLabelSizeX):
        for y in range(digitLabelSizeY):
          mask[midPoint[1]+x][midPoint[0]+y] = self.digitLabelsDict[num_top_features-cnt][x][y]

      cnt += 1

    highlighted_image = copy.deepcopy(img)
    
    # 26/8/23 DH: [start:stop:step], 
    #             [:,:,np.newaxis], 'np.newaxis' = "add another layer" so make 'mask' 3D like 'highlighted_image'
    highlighted_image = highlighted_image * mask[:,:,np.newaxis]
    
    return highlighted_image
  
  # 30/8/23 DH: Encapsulated requirements in 'limeImage':
  #             coeff, numSegments, imgSegmentMask, img, perturb_image(), 
  def displayTopFeatures(self, limeImage):
    """#### Compute top features (segments)
    Now we just need to sort the coefficients to figure out which of the segments have larger coefficients 
    (magnitude) for the prediction of labradors. The identifiers of these top features or segments are shown 
    below. 

    (Even though here we use the magnitude of the coefficients to determine the most important features, 
    other alternatives such as forward or backward elimination can be used for feature importance selection.)
    """

    num_top_featureS = num_top_feature = 4

    # 24/8/23 DH:
    last_features = -1
    self.createDigitLabels()

    print("-------------------------------------")
    while num_top_feature > 0:
      # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
      # "It returns an array of indices ... in sorted order."
      # "kind=None: Sorting algorithm. The default is ‘quicksort’."

      top_features = np.argsort(limeImage.coeff)[-num_top_feature:]
      top_feature = np.argsort(limeImage.coeff)[-num_top_feature]
      
      print("\ntop_feature:",top_feature,"=",limeImage.coeff[top_feature])

      currentSegmentsMask = np.zeros(limeImage.numSegments)
      lastSegmentMask = np.zeros(limeImage.numSegments)

      currentSegmentsMask[top_features] = True #Activate top imgSegmentMask

      if last_features != -1:
        print("last_feature: ",last_features)
        lastSegmentMask[last_features] = True

        img2 = self.highlight_image(img, currentSegmentsMask, limeImage.imgSegmentMask, 
                                    num_top_featureS, last_features)

        last_features.append(top_features[0])
      else: # first time
        last_features = []
        last_features.append(top_features[0])
        img = limeImage.perturb_image(limeImage.img/2+0.5, currentSegmentsMask, limeImage.imgSegmentMask)
        img2 = img

      img3 = skimage.segmentation.mark_boundaries(img2, limeImage.imgSegmentMask)
      skimage.io.imshow( img3 )
      plt.show()

      num_top_feature -= 1

  # 2/9/23 DH:
  def displayRegressionLines(self, limeImage):
    # 2/9/23 DH: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

    # 2/9/23 DH: 'Xvals' is sample set from binomial distrib of segment mask centred on 50% all segments
    y_pred = limeImage.simpler_model.predict(limeImage.Xvals)

    # https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/linear_model/_base.py#L650
    #   "y :  ...Will be cast to X's dtype if necessary."
    # /Users/doug/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/linear_model/_base.py
    
    # 2/9/23 DH: "cannot reshape array of size 100 into shape (100,28)"
    #yPredCast = np.reshape(y_pred, (y_pred.shape[0], Xvals[0].shape[0]) )

    # 2/9/23 DH: 'resize()' was what I thought 'reshape()' did, last night at 1am...!
    #            ...leads to "dirty memory" issue with index values being toggled...!
    #yPredCast = np.resize( y_pred, limeImage.Xvals.shape )
    #y_pred.resize( limeImage.Xvals.shape )

    # 3/9/23 DH: 'np.ndarray()' alse lead to "dirty memory" issues
    #y_predCasted = np.ndarray(limeImage.Xvals.shape)

    y_predCasted = np.zeros(limeImage.Xvals.shape)
    
    for i in range(y_pred.shape[0]):
      y_predCasted[i][0] = y_pred[i][0]

    print("What does:\n",y_predCasted[0],"\nfrom:\n",limeImage.Xvals[0],"\nactually mean...???")

    # 2/9/23 DH: Add the scatter values (in this case just 2, a pair for each prediction/distrib mask)
    plt.scatter(limeImage.Xvals, y_predCasted, color="black")
    # 2/9/23 DH: Plot the linear regression line (ie just join up the pairs so 1 regression per sample)
    plt.plot(limeImage.Xvals, y_predCasted, color="blue", linewidth=1)

    #plt.title("Linear regression lines from " + str(limeImage.num_perturb) + " samples\n")
    plt.title("What does: " + str(y_predCasted[0][0]) + "\nfrom " + str(limeImage.Xvals[0]) + "\nmean...?")

    plt.xlabel("Binomial sample for 50% all segments")
    plt.ylabel("Prediction correlation with all segments")

    plt.show()

  # 3/9/23 DH:
  def displayCoefficients(self, limeImage):
    # 2/9/23 DH: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

    Xvals = range(limeImage.coeff.shape[0])
    yVals = limeImage.coeff

    plt.scatter(Xvals, yVals, color="black")
    
    plt.axhline(y=0.1, color='blue', linestyle='-')

    plt.title("Coefficients for " + str(limeImage.numSegments ) + " segments\n")
    plt.xlabel("Segment number")
    plt.ylabel("Coefficent to linear regression")

    plt.show()


