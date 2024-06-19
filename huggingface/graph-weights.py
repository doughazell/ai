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
from pathlib import Path

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

# 16/6/24 DH: Now wanting to just save graphs (rather than display them as default)
gShowFlag = False

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

      #print(f"{epoch}-{lineType}: {weightList[:10]}...")
      # 15/5/24 DH: This needs to be defined here because if it is defined outside this scope then the same MUTABLE type will be used
      #             for all "epoch-[start | end]" keys resulting in all having the values of "19-end"
      percentChgDict = {}
      for idx in range(len(weightList)):
        percentChgDict[idx] = weightList[idx]

      try:
        percentChgDictListDict[epoch].append(percentChgDict)
      except KeyError:
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
def graphWeightsKeyed(percentChgDictList, epochNum, weights=False, lastGraph=False, lastEpoch=None):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  # 12/5/24 DH: Providing more feedback to output stage
  if "complete" not in epochNum:
    if weights:
      titleStr = f"Weight by node from 'qa_outputs' layer for epoch {epochNum}"
    else:
      titleStr = f"Weight change by node from 'qa_outputs' layer for epoch {epochNum}"
  
  else:
    titleStr = f"Total node weight change from 'qa_outputs' after {lastEpoch} epochs"

  plt.title(titleStr)
  plt.xlabel("Node number (NOT token ID)")
  plt.ylabel("Weight")

  print(f"  \"{titleStr}\" (USING: 'abs(prevWeight)')")

  listLen = len(percentChgDictList)
  idx = 0
  for iDict in percentChgDictList:
    xVals = percentChgDictList[idx].keys()
    yVals = [percentChgDictList[idx][key] for key in percentChgDictList[idx]]
      
    lwVal = (idx + 1) / listLen

    plt.plot(xVals, yVals, label=f"{weightMapDict[idx]}", linewidth=lwVal)

    idx += 1
  
  plt.legend(loc="best")

  #legendStr = f"Start logits: Solid line\nEnd logits:   Dotted line"
  #plt.figtext(0.15, 0.2, legendStr)
  
  #plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  # 15/6/24 DH: Adjust space around y-axis label to accom large axis values (eg "25000")
  #             https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
  plt.tight_layout()

  # 16/6/24 DH: 'gWeightsGraphDir' assigned in "if __name__ == "__main__""
  print(f"    ...saving graph (in '{gWeightsGraphDir}')")
  if "complete" in epochNum:
    plt.savefig(f"{gWeightsGraphDir}/total-weight-change.png")
  else:
    if weights:
      plt.savefig(f"{gWeightsGraphDir}/{epochNum}-fullValues.png")
    else:
      plt.savefig(f"{gWeightsGraphDir}/{epochNum}-percentDiffs.png")

  if gShowFlag:
    #plt.draw()
    plt.show(block=lastGraph)

# 7/6/24 DH:
def displayLargestValues(startLineIdxDict, endLineIdxDict):
  print("Start line indexes:")
  for key in startLineIdxDict:
    print(f"  Idx: {key} = {startLineIdxDict[key]}")
  print()
  print("End line indexes:")
  for key in endLineIdxDict:
    print(f"  Idx: {key} = {endLineIdxDict[key]}")
  print()

# 7/6/24 DH:
def getLargestValues(percentChgLineList):
  topListNum = 10
  startLineIdxDict = {}
  endLineIdxDict = {}

  lineIdx = 0
  for lineDict in percentChgLineList:
    lineKeys = lineDict.keys()
    sortedIdxs = sorted(range(len(lineKeys)), key=lambda i: abs(lineDict[i]), reverse=True)[:topListNum]

    """
    startStr = "Top"
    print(f"{startStr:>10} '{topListNum}' selected")
    startStr = "---"
    print(f"{startStr:>10}--------------")
    for idx in sortedIdxs:
      print(f"{idx:>10}: {lineDict[idx]}")
    print()
    """

    # If start logits line then save it for later comparison
    if lineIdx == 0:
      prevSortedIdxs = sortedIdxs
      prevLineDict = lineDict

    # Now get top values from both lines by finding whether GREATER THAN CUTOFF from largest value of both lines
    else:
      cutoffPercent = 0.025
      # Find top value from each sortedIdx '0'
      if abs(lineDict[sortedIdxs[0]]) > abs(prevLineDict[prevSortedIdxs[0]]):
        print(f"TOP VALUE: '{weightMapDict[1]}' idx: {sortedIdxs[0]} = {lineDict[sortedIdxs[0]]}")
        cutoff = round(abs(lineDict[sortedIdxs[0]] * cutoffPercent))
        print(f"CUTOFF FOR INCLUSION: +- {cutoff} (at {cutoffPercent*100}%)")
      else:
        print(f"TOP VALUE: '{weightMapDict[0]}' idx: {prevSortedIdxs[0]} = {prevLineDict[prevSortedIdxs[0]]}")
        cutoff = round(abs(prevLineDict[prevSortedIdxs[0]] * cutoffPercent))
        print(f"CUTOFF FOR INCLUSION: +- {cutoff} (at {cutoffPercent*100}%)")
      
      for idx in sortedIdxs:
        if abs(lineDict[idx]) > cutoff:
          endLineIdxDict[idx] = lineDict[idx]
      
      for idx in prevSortedIdxs:
        if abs(prevLineDict[idx]) > cutoff:
          startLineIdxDict[idx] = prevLineDict[idx]
        
    # END: 'if lineIdx == 0: else:'

    lineIdx += 1

  return (startLineIdxDict, endLineIdxDict)

# 22/5/24 DH: Taken from 'hugging_utils.py::getWeightStats(...)'
def getWeightDiffs(percentChgDictListDict, lineType):
  percentChgDict = {}

  # https://docs.python.org/3/library/collections.html#ordereddict-objects
  #   "built-in dict class gained the ability to remember insertion order"
  keyList = list(percentChgDictListDict.keys())
  
  # 25/5/24 DH: Running 'huggingface/qa.py' after 'huggingface/run_qa.py' OVERWROTE 'weights-full.log' + 'weights.log'
  try:
    startEpoch = keyList[0]

    if "Start" in weightMapDict[lineType]:
      print()
      print("  Calculating total weight diffs")
      print("  ------------------------------")

    print(f"  {weightMapDict[lineType]} line")
    print(f"    starting at epoch: {startEpoch}")
  except IndexError:
    print()
    print("There is no epoch data...exiting")
    exit(0)
  
  # 24/5/24 DH:
  try:
    endEpoch = keyList[1]

    print(f"    ending at epoch: {endEpoch}")
    if "End" in weightMapDict[lineType]:
      print()
  except IndexError:
    print()
    print("There is no end epoch data...exiting")
    exit(0)

  firstEpochWeights = percentChgDictListDict[startEpoch][lineType]
  lastEpochWeights = percentChgDictListDict[endEpoch][lineType]

  # ORIG HARD-CODING:
  # (Elem 0 of epoch '0' is start weights)
  #firstEpochWeights = percentChgDictListDict['0'][lineType]
  #lastEpochWeights = percentChgDictListDict['19'][lineType]

  for key in firstEpochWeights.keys():
    currWeight = lastEpochWeights[key]
    prevWeight = firstEpochWeights[key]

    # Need percent change from previous
    # 22/5/24 DH: 'diff/prevWeight' needs 'abs(prevWeight)'
    diff = currWeight - prevWeight
    percentChgFromPrev = round(diff/ abs(prevWeight) * 100, 3)

    """ "EXTRA EXTRA, read all about it..."
    mTxt = f"{key} Diff:"
    extraTxt = ""
    if percentChgFromPrev > 100 or percentChgFromPrev < -100:
      extraTxt = f", CURRENT: {currWeight}, PREV: {prevWeight}"
    print(f"{mTxt:>17} {diff}, Percent from prev: {percentChgFromPrev}% {extraTxt}")
    """

    percentChgDict[key] = percentChgFromPrev

  return (percentChgDict, endEpoch)

# 22/5/24 DH: WRAPPER around 'getWeightDiffs()' for each line
def calcAndGraphTrgDiffs(percentChgDictListDict):
  percentChgLineList = []

  # 'weightMapDict' = "{0: "Start", "Start": 0, ...}"
  for key in weightMapDict.keys():
    
    if isinstance(key, int):
      lineType = key
      
      (percentChgLine, endEpoch) = getWeightDiffs(percentChgDictListDict, lineType)
      percentChgLineList.append(percentChgLine)
  
  # 7/6/24 DH: Now display largest +ve/-ve values from both start + end lines
  (startLineIdxDict, endLineIdxDict) = getLargestValues(percentChgLineList)
  displayLargestValues(startLineIdxDict, endLineIdxDict)
  
  graphWeightsKeyed(percentChgLineList, "complete", lastGraph=True, lastEpoch=endEpoch)

  return percentChgLineList

# 16/6/24 DH: https://docs.python.org/3/faq/programming.html#what-are-the-rules-for-local-and-global-variables-in-python
#  "If a variable is assigned a value anywhere within the function’s body, it’s assumed to be a local unless explicitly declared as global."
# HOWEVER: this "main" is NOT A FUNCTION SO ALL VARIABLES ARE GLOBAL 
#          (and leads to "SyntaxError: name ??? is assigned to before global declaration" if used (prob due to python compiler/loader))
if __name__ == "__main__":
  if len(sys.argv) > 1:
    output_dir = os.path.abspath(sys.argv[1])
    weightsLog = os.path.join(output_dir, gTrainer_log)
    fullweightsLog = os.path.join(output_dir, gTrainer_full_log)

    gWeightsGraphDir = os.path.join(output_dir, "weights-graphs")
    Path(gWeightsGraphDir).mkdir(parents=True, exist_ok=True)
  else:
    print(f"You need to provide an 'output_dir'")
    exit(0)
  
  # --------------------------- PERCENTAGE DIFFS BY EPOCH ------------------------------------
  percentChgDictListDict = collectWeights(weightsLog)
  #printCollectedDict(percentChgDictListDict)

  keyNum = len(percentChgDictListDict.keys())
  
  print( "-------------------------------------------------")
  print(f"   NOT GRAPHING PERCENTAGE DIFFS FOR {keyNum} epochs")
  print( "-------------------------------------------------")

  """
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
  """
  # --------------------------- 'qa_outputs' WEIGHTS + PERCENTAGE DIFF --------------------------
  # 22/5/24 DH: Having graphed all the rounded, percentage diffs then we need to 
  #             calculate + graph the percentage diff between the first and last full value weights
  percentChgDictListDict = collectWeights(fullweightsLog)

  # Firstly graph the 'qa_outputs' full weights
  # 26/5/24 DH: During the Ctrl-C checkpointing save delay (to prevent partial saving) we sometimes get multiple end full weights
  #             ('getWeightDiffs(...) uses "startEpoch = keyList[0]", "endEpoch = keyList[1]")
  keyList = list(percentChgDictListDict.keys())
  startEpoch = keyList[0]
  endEpoch = keyList[1]
  if len(keyList) > 2:
    print(f"  (not graphing epochs: ", end='')
    for key in keyList[2:]:
      print(f"{key}, ", end='')
    print(")")
    print()

  # 16/6/24 DH: "show" cmd line arg (like 'graph-node-logits.py') is needed in 'graphWeightsKeyed(...)'
  #             (which is called from 'calcAndGraphTrgDiffs(...)' wrapper)
  if len(sys.argv) > 2 and "show" in sys.argv[2]:
    gShowFlag = True
  
  graphWeightsKeyed(percentChgDictListDict[startEpoch], startEpoch, weights=True)
  graphWeightsKeyed(percentChgDictListDict[endEpoch], endEpoch, weights=True)

  # Now calculate + graph the percentage diff
  percentChgLineList = calcAndGraphTrgDiffs(percentChgDictListDict)

  if not gShowFlag:
    print()
    print("NOT SHOWING images (please add 'show' to cmd line args if images wanted)")
    print()

  
  
