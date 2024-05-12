# 29/3/24 DH:
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

trainer_log = "seq2seq_qa_INtrainer.log"

# 30/3/24 DH:
def getLoss(recordsDict, line, lossName, lossType):
  # Search for f") {lossName}: "
  lineSplit = re.split(rf"\) {lossName}: ", line)

  if len(lineSplit) > 1:
    linePart = lineSplit[1]
    otherPart = lineSplit[0].strip()

    loss = round(float(linePart),2)

    lineSplit = re.split(r' ', otherPart)
    epochNum = lineSplit[-1]

    try:
      recordsDict[lossType][epochNum] = loss
    except KeyError as e:
      recordsDict[lossType] = {}
      recordsDict[lossType][epochNum] = loss

def getLogits(recordsDict, line, logitsName, logitsType):
  # Search for f"{logitsName} (+2): "
  lineSplit = re.split(rf"{logitsName} \(.*\): ", line)
  if len(lineSplit) > 1:
    linePart = lineSplit[1]
    otherPart = lineSplit[0].strip()

    logitsArray = np.array( literal_eval(linePart) )

    # Split with either " " or ")"
    lineSplit = re.split(r'[ \)]', otherPart)
    epochNum = lineSplit[-2]

    try:
      # From 'ai/huggingface/sort_error_logs.py'
      #strDict[searchStr].append(line)

      recordsDict[logitsType][epochNum] = logitsArray
    except KeyError as e:
      #strDict[searchStr] = []
      #strDict[searchStr].append(line)

      recordsDict[logitsType] = {}
      recordsDict[logitsType][epochNum] = logitsArray

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

    # EXAMPLE: 
    # "2024-03-29 12:45:50,085 [INFO] input_ids[0]: tensor([ 101, 2043, 2079, 8220, 2707, 1029,  102, 8220, 2707, 2044, 4970,  102])"
    if "input_ids" in line:
      
      # [Regex not GLOB]
      #lineSplit = re.split(r'input_ids\[.*\]: ', line)

      # Split with either "(" or ")" with each needing to be escaped in a "[]" set (SADLY WRITE-ONLY CODE IS REQUIRED)
      lineSplit = re.split(r'[\(\)]', line)

      if len(lineSplit) > 1:
        linePart = lineSplit[1]
        recordsDict['input_ids'] = np.array( literal_eval(linePart) )
  
    # EXAMPLE: 
    # "2024-03-29 12:45:50,086 [INFO] 1) startLogitsList (+2): [0.32, -0.04, -0.34, 0.07, -0.06, -2.18, -0.3, -0.12, 0.12, 2.51, 0.1, -0.22, -3.4, -3.77]"
    if "startLogitsList" in line:
      # 'recordsDict' is mutable so "pass-by-reference"
      getLogits(recordsDict, line, "startLogitsList", "start_logits")

    # EXAMPLE: 
    # "2024-03-29 12:45:50,087 [INFO] 1) endLogitsList (+2): [0.41, 0.23, 0.15, 0.55, 0.22, -1.71, -0.26, 0.8, 0.16, 0.19, 2.75, -0.13, -2.76, -2.68]"
    if "endLogitsList" in line:
      getLogits(recordsDict, line, "endLogitsList", "end_logits")

    # EXAMPLE:
    # "2024-03-30 19:31:16,695 [INFO]   10) start_loss: 0.323818176984787"
    if "start_loss" in line:
      getLoss(recordsDict, line, "start_loss", "start_loss")
    
    # EXAMPLE:
    # "2024-03-30 19:31:16,696 [INFO]   10) end_loss: 0.35229402780532837"
    if "end_loss" in line:
      getLoss(recordsDict, line, "end_loss", "end_loss")

  return recordsDict

def displayLogits(recordsDict):
  print()
  print(f"recordsDict")
  print( "-----------")
  for key in recordsDict:
    if "logits" in key:
      print(f"{key}:")
      for logitsKey in recordsDict[key]:
        # Truncates to 12 per line so need to handcraft the layout
        #logitsVals = recordsDict[key][logitsKey]

        logitsVals = ""
        cnt = 0
        for val in recordsDict[key][logitsKey]:
          cnt += 1
          logitsVals += f"{val}, "

        print(f"  {logitsKey} ({cnt}): {logitsVals}")
    elif "input_ids" in key:
      input_ids_len = len(recordsDict[key])
      print(f"{key} ({input_ids_len}): {recordsDict[key]}")
    else:
      print(f"{key}: {recordsDict[key]}")
  print( "-----------")

# 30/3/24 DH: There is an A & B order to data with B being consistently different in a proportion of the epochs
def pruneLogits(recordsDict):
  print()
  print("Pruning:")
  print("-------")

  try:
    tokenLen = len(recordsDict['input_ids'])
  except KeyError:
    print(f"There are no 'input_ids' so check '{trainer_log}'")
    exit(0)

  deleteList = []
  for key in recordsDict:
    if "logits" in key:
      print(f"{key}:")
      
      for epochKey in recordsDict[key]:
        logitsLen = len(recordsDict[key][epochKey])
        if logitsLen != tokenLen + 2:
          print(f"  '{epochKey}') logitsLen: {logitsLen}, tokenLen: {tokenLen}")
          deleteList.append(epochKey)
    # END: --- if "logits" in key ---

    # Only add the epochKey once so use https://docs.python.org/3/tutorial/datastructures.html#sets
    #   "A set is an unordered collection with no duplicate elements."
    deleteSet = set(deleteList)
    
    if "logits" in key or "loss" in key:
      print(f"Deleting '{key}' keys ({deleteSet.__class__}): {deleteSet}")
      for item in deleteSet:
        try:
          del recordsDict[key][item]
        except KeyError as e:
          print(f"Unable to delete KEY: {key}, ITEM: {item}")
    
  # END: --- for key in recordsDict ---
print("-------")

# ------------------------------------------------ GRAPHING --------------------------------------------------
def graphLossLines(recordsDict, keyVal, xVals):
  yVals = recordsDict[keyVal].values()

  if "end_loss" in keyVal:
    plt.plot(xVals, yVals, label=f"end_loss", linestyle='dashed')
  else:
    plt.plot(xVals, yVals, label=f"start_loss")

# 30/3/24 DH: The 'start_logits' should correlate with 'end_logits' so only need to check 'start_logits'
def getEpochList(recordsDict, firstOnly):
  epochList = []
  epochs = list(recordsDict['start_logits'].keys())
  epochsLen  = len(epochs)
  middleIdx = round(epochsLen / 2)

  # 12/5/24 DH: Tidying up user output
  if firstOnly == False:
    print(f"    Initial epochs: {epochs}, middleIdx: {middleIdx}")
    print(f"      Reduced to: [{epochs[0]}, {epochs[middleIdx]}, {epochs[-1]}] (as detailed in the graph legend)")

  try:
    epochList.append(epochs[0])
    epochList.append(epochs[middleIdx])
    epochList.append(epochs[-1])
  except IndexError:
    print()
    print(f"*** THERE APPEARS TO BE AN ERROR ***")
    print( "with: ")
    displayLogits(recordsDict)
    exit(0)

  return epochList

def graphLogitLines(recordsDict, keyVal, xVals, firstOnly=False):
  # 30/3/24 DH: Only graph {start, middle, end} logit values for default graph
  epochList = getEpochList(recordsDict, firstOnly)
  epochListLen = len(epochList) - 1

  # 1/4/24 DH: Graph at epoch 1 (which is different to untrained) to compare against 
  #            'qa.py' run of specified epoch trained model
  if firstOnly:
    epochList = epochList[0]
    epochListLen = 1
  
  firstEnd = True
  firstStart = True
  for epoch in epochList:
    lwVal = (int(epoch) / epochListLen) / 3

    yVals = recordsDict[keyVal][epoch]
    
    xValsLen = len(xVals)
    yValsLen = len(yVals)
    if xValsLen == yValsLen:
      
      firstTitle = ""
      if "end" in keyVal:
        if firstEnd:
          firstTitle = "End "
          firstEnd = False
        
        plt.plot(xVals, yVals, label=f"{firstTitle}#{epoch}", linestyle='dashed', linewidth=lwVal)
      else:
        if firstStart:
          firstTitle = "Start "
          firstStart = False
        
        plt.plot(xVals, yVals, label=f"{firstTitle}#{epoch}", linewidth=lwVal)

    # Shouldn't need this anymore after 'pruneLogits()'
    else:
      print(f"  KEY: {epoch}, xValsLen: {xValsLen}, yValsLen: {yValsLen}")

# Taken from: 'ai/huggingface/stop_trainer_utils.py'
def graphLogits(recordsDict):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  # 12/5/24 DH: Providing more feedback to output stage
  titleStr = "Logits by Token from different training epochs"

  plt.title(titleStr)
  plt.xlabel("Token ID")
  plt.ylabel("Logit value")

  xValNum = recordsDict['input_ids'].shape[0]
  xVals = range(xValNum + 2)

  print(f"\"{titleStr}\"")
  print("  Adding start logits lines")
  graphLogitLines(recordsDict, "start_logits", xVals)
  print("  Adding end logits lines")
  graphLogitLines(recordsDict, "end_logits", xVals)
  print()
  plt.legend(loc="upper left")

  #legendStr = f"Start logits: Solid line\nEnd logits:   Dotted line"
  #plt.figtext(0.15, 0.2, legendStr)
  
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  #plt.draw()
  plt.show(block=False)

# 30/3/24 DH: Add the {start_loss + end_loss} line graph
def graphLosses(recordsDict):
  plt.figure()

  plt.title("Losses from different training epochs")
  plt.xlabel("Training epoch")
  plt.ylabel("Loss value")

  xVals = recordsDict["start_loss"].keys()
  # Convert key strings into integers
  xVals = [int(item) for item in xVals]

  graphLossLines(recordsDict, "start_loss", xVals)
  graphLossLines(recordsDict, "end_loss", xVals)
  #plt.xticks(np.arange(0, 11, step=1))
  plt.legend(loc="upper right")

  plt.show(block=False)

def graphFirstLogits(recordsDict):
  plt.figure()

  # 12/5/24 DH: Providing more feedback to output stage
  titleStr = "Logits by Token from first training epoch"

  plt.title(titleStr)
  plt.xlabel("Token ID")
  plt.ylabel("Logit value")

  xValNum = recordsDict['input_ids'].shape[0]
  xVals = range(xValNum + 2)

  print(f"\"{titleStr}\"")
  print("  (This reuses 'graphLogitLines()')")
  print("  Adding start logits line")
  graphLogitLines(recordsDict, "start_logits", xVals, firstOnly=True)
  print("  Adding end logits line")
  graphLogitLines(recordsDict, "end_logits", xVals, firstOnly=True)
  print()
  plt.legend(loc="upper left")
  
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  plt.show()
# -------------------------------------------------END: GRAPHING ---------------------------------------------

if __name__ == "__main__":
  # 'trainer_log = "seq2seq_qa_INtrainer.log"' centric
  recordsDict = collectLogits()

  pruneLogits(recordsDict)
  displayLogits(recordsDict)
  
  graphLogits(recordsDict)
  graphLosses(recordsDict)
  graphFirstLogits(recordsDict)