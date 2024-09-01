# Huggin API
### Introduction
This is my Repo to learn about Transformer Neural Networks which has become a https://github.com/huggingface/transformers/tree/main/src/transformers API.

This has been a major journey for me which started by learning about Keras Sequential networks for https://en.wikipedia.org/wiki/MNIST_database processing with https://github.com/doughazell/mnist-training-errors.  Then moved onto using https://nbviewer.org/url/arteagac.github.io/blog/lime_image.ipynb with https://github.com/doughazell/ai/blob/main/lime.py (diagram below of LIME process).



# LIME process
LIME (https://arxiv.org/abs/1602.04938) works by having a Binomial Distribution of image masking (ie removing segments of the full image) in order to perturb the image prior to calling 'keras.applications.inception_v3.InceptionV3().predict(perturbed_img)'.  Then get the RMS segment diff from the orig image to each of Binomially Distributed segment masks.  Finally correlate the RMS diff with the place of the predicted full image.  This then provides an order to segment importance of the final prediction (to compare how you would ID the same image and therefore gain confidence in the prediction).

A good place to start with understanding LIME is:
```
'lime.py::runLimeAnalysis()'

  # Step 1/4
  limeImage.createRandomPertubations()

  # Step 2/4
  limeImage.getPredictions()
  
  # Step 3/4
  limeImage.getDistanceWeights()
  
  # Step 4/4
  limeImage.getLinearRegressionCoefficients()

    -------------------------------------
    Final step (4/4) in: getLinearRegressionCoefficients
                         -------------------------------
     to correlate InceptionV3 prediction of full image (class index 208 )
     with masked image predictions, inceptionV3_model.predict()  (in Step 2/4, 'Lime.getPredictions()')
     using weights obtained for mask distance from full image    (in Step 3/4, 'Lime.getDistanceWeights()')

    class_to_explain: 208 , from all InceptionV3 trained classes: (100, 1, 1000)
    LinearRegression.fit(): mask perturbations: (100, 28) , prediction: (100, 1)
     eg...Xvals[0]: [1 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 1] , yVals[0]: [0.07313019]

    Multiple LinearRegression() coeffs (from weights): (28,)
     eg...lowest: -0.10693367968250382 , highest: 0.3781445941220827
     (100 pertubation masks for 28 segments leads to a linear correlation line of importance of each segment in full image)
    -------------------------------------
```

### Diagram overview
* Top Row: Get mask from Binomial Distribution of which of 28 segments to include

  eg Top prediction of first perturbation: [1 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 1] = conch 
  
* Row 2: Get prediction place for 'Labrador_retriever' (ie top prediction for full image) + RMS diff between segment mask used and full image mask

  eg [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
  
* Row 3: Perform a multiple linear regression for prediction place of 'Labrador_retriever' vs 28 bit segment mask
* Bottom Row: 'InceptionV3().predict()' produces order of 1000 known images (eg 'Labrador_retriever', 'conch' etc) and need to correlate placing of image 208 (ie 'Labrador_retriever') for each of 100 * 28 bit masks.

![alt text](https://github.com/doughazell/ai/blob/main/LIME-flow-diag.jpg)

### Running 'lime.py'
```
$ python lime.py
```
Gives:

![alt text](https://github.com/doughazell/ai/blob/main/segment_coeffs.png)

and

![alt text](https://github.com/doughazell/ai/blob/main/top_segments.png)


