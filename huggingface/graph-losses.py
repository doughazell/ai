# 25/5/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

gTrainerLoss_log = "loss-by-epochs.log"

def collectLosses(lossLog):
  lossDict = {}
  
  try:
    with open(lossLog) as source :
      print()
      print(f"Filename: {lossLog}")
      print( "--------")

      textLines = [line.strip() for line in source.readlines() if line.strip()]
  except FileNotFoundError:
    print(f"Filename: {lossLog} NOT FOUND")
    exit(0)
  
  # eg "Epoch: 50 of: 14754 = 3.9882020950317383 (Grad steps: 1)"
  for line in textLines:
    
    lineSplit = line.split(": ")
    splitLen = len(lineSplit)
    if len(lineSplit) > 1:
      
      subSplit = lineSplit[1].split(" ")
      epochNum = int(subSplit[0])

      subSplit2 = lineSplit[2].split(" ")
      loss = float(subSplit2[2])

      lossDict[epochNum] = loss
  
  return lossDict

def graphLosses(lossDict):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  titleStr = f"Loss from each training epoch"

  plt.title(titleStr)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")

  print(f"  \"{titleStr}\"")

  xVals = list(lossDict.keys())
  yVals = [lossDict[key] for key in lossDict.keys()]
  plt.plot(xVals, yVals)

  # 26/5/24 DH: Adding best fit line to exponential decay of loss (modulated by form of Pretrained BERT)
  # ----------------------------------------------------------------------------------------------------
  # find LINEAR line of best fit
  """
  a, b = np.polyfit(xVals, yVals, 1)
  
  yValsBestfit = [a*x+b for x in xVals]
  # https://matplotlib.org/stable/gallery/color/named_colors.html
  plt.plot(xVals, yValsBestfit, color='yellow', linestyle='--', linewidth=2)
  """

  # find EXPONENTIAL line of best fit
  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
  def func(x, a, b, c):
    # https://docs.python.org/3/library/warnings.html#the-warnings-filter
    # Ignore: "RuntimeWarning: overflow encountered in exp"
    import warnings
    warnings.filterwarnings("ignore")

    return a * np.exp(-b * x) + c

  popt, pcov = curve_fit(func, xVals, yVals)

  yValsExpfit = [func(x, *popt) for x in xVals]
  plt.plot(xVals, yValsExpfit, color='red', linestyle='--', linewidth=2)
  

  # 25/5/24 DH: Since 'loss-by-epochs.log' is overwritten every time 'run_qa.py' is run 
  #             then prevent overwriting the graph image with an incrementing filename
  graphNum = ""
  graphFilename = f"{weightsGraphDir}/losses-by-epochs{graphNum}.png"
  num = 0
  while os.path.isfile(graphFilename):
    num += 1
    graphNum = num
    graphFilename = f"{weightsGraphDir}/losses-by-epochs{graphNum}.png"
  
  plt.savefig(graphFilename)

  plt.show(block=True)

if __name__ == "__main__":
  if len(sys.argv) == 2:
    output_dir = os.path.abspath(sys.argv[1])
    lossLog = os.path.join(output_dir, gTrainerLoss_log)

    weightsGraphDir = os.path.join(output_dir, "weights-graphs")
    Path(weightsGraphDir).mkdir(parents=True, exist_ok=True)
  else:
    print(f"You need to provide an 'output_dir'")
    exit(0)
  
  lossDict = collectLosses(lossLog)
  
  graphLosses(lossDict)
