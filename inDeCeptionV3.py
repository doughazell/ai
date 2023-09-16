# 12/9/23 DH: Black-out top segments for primary prediction then re-predict

from lime import Lime

if __name__ == '__main__':
  print()
  print("Hello blackout segments...I'd rather be a distance than a weight")
  print("----------------------------------------------------------------")
  print()

  limeImage = Lime()

  limeImage.getSegmentedPrediction()
  limeImage.getCoefficientsFromMaskedPredictions()
  newImg = limeImage.lime_utils.displayTopFeaturesRemoved(limeImage)

  limeImage.getSegmentedPrediction(img=newImg)
  limeImage.getCoefficientsFromMaskedPredictions()
  newImg = limeImage.lime_utils.displayTopFeaturesRemoved(limeImage)

  # 15/9/23 DH:
  limeImage.printPredictionSummary()

  print()
  print("----------------------------------------------------------------")
  print("...nailed it")
