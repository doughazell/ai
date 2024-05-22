# 15/5/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys, os, re
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

gTrainer_log = "weights.log"
gTrainer_full_log = "weights-full.log"

# 12/5/24 DH: MUTABLE variables (don't need to be accessed with 'global' to prevent local scope overlay)
#             (a task centric version of 'hugging_utils.py::weightMapDict')
weightMapDict = {
  0: "Start",
  "Start": 0,
  1: "End",
  "End": 1,
}

def collectWeights(weightsLog):
  percentChgDictListDict = {}
  
  try:
    with open(weightsLog) as source :
      print()
      print(f"Filename: {weightsLog}")
      print( "--------")

      textLines = [line.strip() for line in source.readlines() if line.strip()]
  except FileNotFoundError:
    print(f"Filename: {weightsLog} NOT FOUND")
    exit(0)
  
  # eg "19-End: [0.046, -0.002, -0.008, 0.015, ...]"
  #
  # "xVals = percentChgDictList[idx].keys()" ie from "{0: 0.022, 1: 0.023, 2: 0.047, ..., 767: 0.085}"
  # 'huggin_utils.py::graphWeights(...)' => "epochNum = Trainer.stateAPI.global_step"
  #                                      => List of Dict (like above) for each of start+end weights (so graph has 2 lines)
  for line in textLines:
    lineSplit = line.split(": ")
    if len(lineSplit) > 1:
      subSplit = lineSplit[0].split("-")

      epoch = subSplit[0]
      lineType = subSplit[1]
      lineTypeIdx = weightMapDict[lineType]
      #print(f"Epoch: {epoch}, Line Type: {lineType} (ie Idx: {lineTypeIdx})")

      # The easy way to parse an "[array, string]"
      weightList = np.array( literal_eval(lineSplit[1]) )

      print(f"{epoch}-{lineType}: {weightList[:10]}...")
      # 15/5/24 DH: This needs to be defined here because if it is defined outside this scope then the same MUTABLE type will be used
      #             for all "epoch-[start/end]" keys resulting in all having the values of "19-end"
      percentChgDict = {}
      for idx in range(len(weightList)):
        percentChgDict[idx] = weightList[idx]

      try:
        percentChgDictListDict[epoch].append(percentChgDict)
        print()
      except KeyError:
        print(f"  Creating start/end list for {epoch}")
        percentChgDictListDict[epoch] = []
        percentChgDictListDict[epoch].append(percentChgDict)
      
  return percentChgDictListDict

def printCollectedDict(percentChgDictListDict):
  for key in percentChgDictListDict:
    print(f"{key}:")

    idx = 0
    for iDict in percentChgDictListDict[key]:
      print(f"  {idx} (ie {weightMapDict[idx]}):")

      weightsSublist = []
      sublistIdx = 0
      for weightsKey in percentChgDictListDict[key][idx]:
        if sublistIdx < 10:
          weightsSublist.append(percentChgDictListDict[key][idx][weightsKey])
        
        sublistIdx += 1
      
      mylist = ""
      for elem in weightsSublist: 
        mylist += f"{elem}, ";
      print(f"    [{mylist}...]")
      
      idx += 1
      print("    ------")
    print("  ======")

    print()

# Taken from: 'ai/huggingface/huggin_utils.py'
def graphWeightsKeyed(percentChgDictList, epochNum, lastGraph=False):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  # 12/5/24 DH: Providing more feedback to output stage
  if "complete" not in epochNum:
    titleStr = f"Weight change by node from start/end layer for epoch {epochNum}"
  else:
    titleStr = f"Total weight change by node from start/end layer"

  plt.title(titleStr)
  plt.xlabel("Node number")
  plt.ylabel("Weight")

  print(f"  \"{titleStr}\"")

  listLen = len(percentChgDictList)
  idx = 0
  for iDict in percentChgDictList:
    xVals = percentChgDictList[idx].keys()
    yVals = [percentChgDictList[idx][key] for key in percentChgDictList[idx]]
      
    lwVal = (idx + 1) / listLen
    print(f"    {weightMapDict[idx]} lwVal: {lwVal}")
    plt.plot(xVals, yVals, label=f"{weightMapDict[idx]}", linewidth=lwVal)

    idx += 1
  
  plt.legend(loc="lower right")

  #legendStr = f"Start logits: Solid line\nEnd logits:   Dotted line"
  #plt.figtext(0.15, 0.2, legendStr)
  
  #plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  #plt.draw()
  plt.show(block=lastGraph)

# 22/5/24 DH: Taken from 'hugging_utils.py::getWeightStats(...)'
def getWeightDiffs(percentChgDictListDict, lineType):
  percentChgDict = {}

  # CURRENTLY hard coding collected epoch keys for first and last epochs to: '0', '19'
  #   (Elem 0 of epoch '0' is start weights)
  firstEpochWeights = percentChgDictListDict['0'][lineType]
  lastEpochWeights = percentChgDictListDict['19'][lineType]
  for key in firstEpochWeights.keys():
    currWeight = lastEpochWeights[key]
    prevWeight = firstEpochWeights[key]

    # Need percent change from previous
    diff = currWeight - prevWeight
    percentChgFromPrev = round(diff/prevWeight*100, 3)

    """ "EXTRA EXTRA, read all about it..."
    mTxt = f"{key} Diff:"
    extraTxt = ""
    if percentChgFromPrev > 100 or percentChgFromPrev < -100:
      extraTxt = f", CURRENT: {currWeight}, PREV: {prevWeight}"
    print(f"{mTxt:>17} {diff}, Percent from prev: {percentChgFromPrev}% {extraTxt}")
    """

    percentChgDict[key] = percentChgFromPrev

  return percentChgDict

# 22/5/24 DH: WRAPPER around 'getWeightDiffs()' for each line
def calcAndGraphTrgDiffs(percentChgDictListDict):
  percentChgLineList = []

  for key in weightMapDict.keys():
    # 'weightMapDict' = "{0: "Start", "Start": 0, ...}"
    if isinstance(key, int):
      lineType = key
      print(f"CALLING: 'getWeightDiffs()' for {weightMapDict[lineType]}")
      percentChgLine = getWeightDiffs(percentChgDictListDict, lineType)
      percentChgLineList.append(percentChgLine)
      
  graphWeightsKeyed(percentChgLineList, "complete", lastGraph=True)


if __name__ == "__main__":
  if len(sys.argv) == 2:
    output_dir = os.path.abspath(sys.argv[1])
    weightsLog = os.path.join(output_dir, gTrainer_log)
    fullweightsLog = os.path.join(output_dir, gTrainer_full_log)
  else:
    print(f"You need to provide an 'output_dir'")
    exit(0)

  """
  """
  # --------------------------- PERCENTAGE DIFFS BY EPOCH ------------------------------------
  percentChgDictListDict = collectWeights(weightsLog)
  printCollectedDict(percentChgDictListDict)

  keyNum = len(percentChgDictListDict.keys())

  print(f"The shape of the graphs is similar to Bert/SQuAD training (despite this training being custom JSON)")
  print("(See \"open file:///Users/doug/Desktop/devlogeD/2024/doc/b6-feb24.html#label-Results\")")
  print()

  idx = 0
  for key in percentChgDictListDict:
    # NOTE: Last call to 'graphWeightsKeyed()' needs to call 'plt.show()' (ie without 'block=False') in order to see all graphs before program ends
    if idx + 1 == keyNum:
      graphWeightsKeyed(percentChgDictListDict[key], key, lastGraph=True)
    else:
      graphWeightsKeyed(percentChgDictListDict[key], key)
    idx += 1
  
  # --------------------------- START/END WEIGHTS + PERCENTAGE DIFF --------------------------
  # 22/5/24 DH: Having graphed all the rounded, percentage diffs then we need to 
  #             calculate + graph the percentage diff between the first and last full value weights
  percentChgDictListDict = collectWeights(fullweightsLog)

  # Firstly graph the start/end full weights
  for key in percentChgDictListDict:
    graphWeightsKeyed(percentChgDictListDict[key], key)
  
  # Now calculate + graph the percentage diff
  calcAndGraphTrgDiffs(percentChgDictListDict)
  
