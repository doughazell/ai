# 13/11/24 DH: Created for skimage PR segment numbering test harness with: 
#
#              (skimage-dev) scikit-image$ spin run python ../seg_numbering.py

import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation

# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/axes/_axes.py#L5900
#
# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/image.py#L663
# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/image.py#L809

# ===============================================================================================================
# == 'Axes.setSegmentMaskNumbers(ax, imgSegmentMask)'
# ===============================================================================================================
def oldplugin_imshow(ax, img):
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
  ax_im = ax.imshow(img)
  
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(ax_im, cax=cax)
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.tight_layout.html
  ax.get_figure().tight_layout()

def setSegmentMaskNumbers(ax, imgSegmentMask):
  
  oldplugin_imshow(ax, imgSegmentMask)

  # 2-D array of integers (segment "pixel") to 1-D ordered list of numbers
  sortedSegmentNums = np.unique( np.sort(imgSegmentMask) )

  endX = imgSegmentMask.shape[0]
  endY = imgSegmentMask.shape[1]

  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figtext.html
  
  # Add custom location for unusual shape NEEDS TO BE SPECIFIED IN 'kwargs'
  plt.figtext(0.25, 0.88, "10")

  xLeft = 0.14
  xRight = 0.8
  yTop = 0.95
  yBottom = 0.06

  for num in sortedSegmentNums:
    (midX, midY) = getSegmentMidpointFigText(imgSegmentMask, num)

    #                        1       2                           1       2
    # Segment X increases: Left -> Right, FigText X increases: Left -> Right
    xFigText = ((xRight - xLeft) * (midX / endX)) + xLeft

    #                                   3       4                             4       3
    # Segment Y increases COUNTER FIG: Top -> Bottom, FigText Y increases: Bottom -> Top
    yFigText = (-(yTop - yBottom) * (midY / endY)) + yTop

    if (num != 10):
      plt.figtext(xFigText, yFigText, num)

def getSegmentMidpointFigText(imgSegmentMask, segNum):

  # Now need to set 'startX' & 'startY' for each segment
  (startX, startY) = getSegmentStarts(imgSegmentMask, segNum)

  endX = startX
  endY = startY

  rowNum = 0
  for row in imgSegmentMask:

    colNum = 0 
    for col in row:
      
      try:
        if (imgSegmentMask[rowNum][colNum] == segNum):
          if (colNum > endX):
            endX = colNum
          if (rowNum > endY):
            endY = rowNum
        
      except IndexError:
        pass

      colNum += 1
    
    rowNum += 1

  # Now return the mid-point
  midDeltaX = round( (endX - startX) / 2 )
  midDeltaY = round( (endY - startY) / 2 )

  midX = startX + midDeltaX
  midY = startY + midDeltaY

  return (midX, midY)

def getSegmentStarts(imgSegmentMask, segNum):
  searchStartX = imgSegmentMask.shape[0]
  searchStartY = imgSegmentMask.shape[1]

  rowNum = 0
  for row in imgSegmentMask:

    colNum = 0 
    for col in row:
      
      try:
        if (imgSegmentMask[rowNum][colNum] == segNum):
          if (colNum < searchStartX):
            searchStartX = colNum
          if (rowNum < searchStartY):
            searchStartY = rowNum
        
      except IndexError:
        pass

      colNum += 1
    
    rowNum += 1

  return (searchStartX, searchStartY)

# ===============================================================================================================
# == END Add
# ===============================================================================================================

def displayImgSegments(imgSegmentMask):
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
  fig, ax = plt.subplots()

  # EVENTUALLY: ax.setSegmentMaskNumbers(imgSegmentMask)
  setSegmentMaskNumbers(ax, imgSegmentMask)

  # Get around 'plt.figtext' not being scalable with the image
  fig.savefig('numbered_segments.png')

  plt.show()

def getSegmentedImg():
  img = skimage.io.imread("https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg")

  img = skimage.transform.resize(img, (299,299))
  img = (img - 0.5)*2 #Inception pre-processing

  imgSegmentMask = skimage.segmentation.quickshift(img, kernel_size=6, max_dist=200, ratio=0.2)

  return imgSegmentMask

if __name__ == '__main__':
  
  print()
  print(f"skimage version: {skimage.__version__}")
  print()

  imgSegmentMask = getSegmentedImg()

  displayImgSegments(imgSegmentMask)