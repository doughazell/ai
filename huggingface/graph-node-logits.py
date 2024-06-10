# 8/6/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys, os, re
import numpy as np
from ast import literal_eval

# https://matplotlib.org/stable/api/pyplot_summary.html
import matplotlib.pyplot as plt
from scipy import stats

# 8/6/24 DH: Hard-coded to prevent needing to add: "HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))" code
# ORIG LOG IN 'graph-logits.py': trainer_log = "seq2seq_qa_INtrainer.log"
trainer_log = "weights/node287-logits.log"

def collectLogits():
  recordsDict = {}

  if len(sys.argv) == 2:
    output_dir = os.path.abspath(sys.argv[1])
    logitsLog = os.path.join(output_dir, trainer_log)
  else:
    print(f"You need to provide an 'output_dir'")
    exit(0)
  
  try:
    with open(logitsLog) as source :
      print()
      print(f"Filename: {logitsLog}")
      print( "--------")

      textLines = [line.strip() for line in source.readlines() if line.strip()]
  except FileNotFoundError:
    print(f"Filename: {logitsLog} NOT FOUND")
    exit(0)  
  
  # Used for non-training run when there are no epochs for a unique record key (so need to add incremented counter)
  keyCounter = 1
  for line in textLines:
    # https://docs.python.org/3/library/re.html#re.split, "The solution is to use Python’s raw string notation for regular expression patterns; 
    #   backslashes are not handled in any special way in a string literal prefixed with 'r'"  

    # TRAINING EXAMPLE:     "0-12: [-0.8225892782211304, -0.24359458684921265, ...]"
    # NON-TRAINING EXAMPLE:  "-1-154: [0.0373508557677269, 0.10611902177333832, ...]"
    if ":" in line:
      
      """ 'graph-logits.py'
      # EG: "2024-03-29 12:45:50,085 [INFO] input_ids[0]: tensor([ 101, 2043, 2079, 8220, 2707, 1029,  102, 8220, 2707, 2044, 4970,  102])"
      if "input_ids" in line:
        # [Regex not GLOB]
        #lineSplit = re.split(r'input_ids\[.*\]: ', line)

        # Split with either "(" or ")" with each needing to be escaped in a "[]" set (SADLY WRITE-ONLY CODE IS REQUIRED)
        #lineSplit = re.split(r'[\(\)]', line)
      """
      lineSplit = re.split(r': ', line)
      if len(lineSplit) > 1:
        """ Epoch number parsed in 'graphLogits(...)'
        subSplit = lineSplit[0].split("-")
        epochNum = subSplit[0]
        bertSelfAttention_cnt = subSplit[1]
        """
        
        epochLayerCnt = lineSplit[0]
        linePart = lineSplit[1]
        
        # 10/6/24 DH: For a non-training run there are no epochs so multiple runs have the same key 
        #             (with 'recordsDict' just holding the last run if not checked for existing key entry)
        if epochLayerCnt in recordsDict:
          keyCounter += 1
          epochLayerCnt = f"{epochLayerCnt},{keyCounter}"
  
        recordsDict[epochLayerCnt] = np.array( literal_eval(linePart) )
  
  return recordsDict

def displayLogits(recordsDict):
  print()
  print(f"recordsDict")
  print( "-----------")
  for key in recordsDict:
    # 'graph-logits.py' displayed differently for: "logits", "input_ids"
    printElems = 5
    print(f"{key}: ({recordsDict[key].__class__} first {printElems} elems) {recordsDict[key][:printElems]}...")
  print( "-----------")


# ------------------------------------------------ GRAPHING --------------------------------------------------
def graphLossLines(recordsDict, keyVal, xVals):
  yVals = recordsDict[keyVal].values()

  if "end_loss" in keyVal:
    plt.plot(xVals, yVals, label=f"end_loss", linestyle='dashed')
  else:
    plt.plot(xVals, yVals, label=f"start_loss")


# Taken from: 'ai/huggingface/graph-logits.py'
def graphLogitsByLayer(recordsDict, layerNum, lastGraph=False):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  # 9/6/24 DH: 'recordsDict' keys are "{epoch}-{layer num}"
  epochList = [key.split("-")[0] for key in list(recordsDict.keys())]
  firstEpoch = epochList[0]
  lastEpoch = epochList[-1]
  
  #selectedLogits = 100
  # Used to print out all logits for a non-training run (which are dependent on Q+Context token length)
  selectedLogits = None

  epochsWanted = [0, 18, 19]
  # 9/6/24 DH: Key is now "{epoch}-{layer num}-{token length}"
  for key in recordsDict:
    # 10/6/24 DH: For non-training run 'keySplit[2]' will be token length, eg "-1-154"
    keySplit = key.split("-")
    epoch = keySplit[0]
    layer = keySplit[1]

    # Hopefully, https://en.wikipedia.org/wiki/Short-circuit_evaluation ...spookily connected with AI...
    # epochsWanted = ['0', '19']
    #
    # "__contains__" means that '0' will match '0' and '10'
    #if any(map(key.__contains__, epochsWanted)) and int(key.split("-")[1]) == layerNum:

    def plotLine(lwVal, labelStr):
      xVals = range(len(recordsDict[key]))
      # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
      yVals = recordsDict[key].tolist()

      # Label with epoch or token length
      plt.plot(xVals[:selectedLogits], yVals[:selectedLogits], label=f"{labelStr}", linewidth=lwVal)

    try:
      if any(map(int(epoch).__eq__, epochsWanted)) and int(layer) == layerNum:
        lwVal = (int(epoch)+4) / (int(lastEpoch))
        plotLine(lwVal, labelStr=epoch)
        # Used in graph title (+ cmd line feedback)
        lineType = "epoch"
        # Used in graph title
        layerName = f"layer {layerNum}"

    # This will catch ALL LINES from non-training run (due to no epoch)
    # EXAMPLE: "-1-154: [0.0373508557677269, 0.10611902177333832, ...]"
    except ValueError: # "invalid literal for int() with base 10: ''" (ie non-training run)
      if len(keySplit) > 2:
        tokenLen = keySplit[2]
      else:
        print(f"NON-training run key '{key}' did not contain a token length")

      # Check for non-training run KEY DUP layer eg "1,2" (prob no longer needed with "-1-154" ie with token length appended)
      # ---------------------------------------------------------------------------------------------------------------------
      layerSplit = layer.split(",")
      if len(layerSplit) > 1:
        layer = layerSplit[0]
      # ---------------------------------------------------------------------------------------------------------------------

      # https://www.statology.org/matplotlib-line-thickness/ 
      #   "By default, the line width is 1.5 but you can adjust this to any value greater than 0."
      if int(layer) == layerNum:
        plotLine(lwVal=0.5, labelStr=tokenLen)
        # Used in graph title
        layerName = f"layer {layerNum}"

      # Debug of graph line shapes by adding all lines together
      if int(layerNum) == -1:
        plotLine(lwVal=0.5, labelStr=tokenLen)
        # Used in graph title
        layerName = "all layers"
      
      # Used in graph title (+ cmd line feedback)
      lineType = "token length"
      

  # 12/5/24 DH: Providing more feedback to output stage
  # 10/6/24 DH: Change title based on epoch for training OR token length for non-training
  titleStr = f"Logits from BertSelfAttention, {layerName}, node 287 by {lineType}"

  plt.title(titleStr)
  plt.xlabel("Logit ID")
  plt.ylabel("Logit value")

  print(f"\"{titleStr}\"")
  
  plt.legend(loc="best")

  #legendStr = f"Start logits: Solid line\nEnd logits:   Dotted line"
  #plt.figtext(0.15, 0.2, legendStr)
  
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  #plt.draw()
  plt.show(block=lastGraph)

# -------------------------------------------------END: GRAPHING ---------------------------------------------

if __name__ == "__main__":
  # 'trainer_log = "weights/node287-logits.log"' centric
  recordsDict = collectLogits()

  displayLogits(recordsDict)

  # Debug of graph line shapes by adding all lines together
  graphLogitsByLayer(recordsDict, layerNum=-1)

  graphLogitsByLayer(recordsDict, layerNum=1)
  graphLogitsByLayer(recordsDict, layerNum=12, lastGraph=True)
  