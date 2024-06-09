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
  
  for line in textLines:
    # https://docs.python.org/3/library/re.html#re.split, "The solution is to use Python’s raw string notation for regular expression patterns; 
    #   backslashes are not handled in any special way in a string literal prefixed with 'r'"  

    # EXAMPLE: "0-12: [-0.8225892782211304, -0.24359458684921265, ...]"
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
  selectedLogits = 100

  epochsWanted = ['0', '19']
  # 9/6/24 DH: Key is now "{epoch}-{layer num}"
  for key in recordsDict:
    
    # Hopefully, https://en.wikipedia.org/wiki/Short-circuit_evaluation ...spookily connected with AI...
    if any(map(key.__contains__, epochsWanted)) and int(key.split("-")[1]) == layerNum:
      xVals = range(len(recordsDict[key]))
      # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
      yVals = recordsDict[key].tolist()

      #lwVal = (int(epoch) / epochListLen) / 3
      epoch = key.split("-")[0]
      lwVal = (int(epoch)+1) / (int(lastEpoch)+1)

      plt.plot(xVals[:selectedLogits], yVals[:selectedLogits], label=f"{epoch}", linewidth=lwVal)

  # 12/5/24 DH: Providing more feedback to output stage
  titleStr = f"Logits by Logit ID from BertSelfAttention Layer {layerNum} Node 287 by epoch"

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
  graphLogitsByLayer(recordsDict, layerNum=1)
  graphLogitsByLayer(recordsDict, layerNum=12, lastGraph=True)
  