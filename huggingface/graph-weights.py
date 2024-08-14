# 15/5/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

"""
See DESIGN in 'graph-weights-history.py'
"""

import sys, os, re, shutil
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from pathlib import Path

gTrainer_log = "weights.log"
gTrainer_full_log = "weights-full.log"
# 29/7/24 DH:
gTrainer_rounded_log = "weights-rounded.log"

# 12/5/24 DH: MUTABLE variables (don't need to be accessed with 'global' to prevent local scope overlay)
#             (a task centric version of 'hugging_utils.py::weightMapDict')
weightMapDict = {
  0: "Start",
  "Start": 0,
  1: "End",
  "End": 1,
}
# 26/7/24 DH: Increased readability for "Dict-List-Dict" data structure
gStartIdx = 0
gEndIdx = 1

# 16/6/24 DH: Now wanting to just save graphs (rather than display them as default)
gShowFlag = False

# 23/7/24 DH: Originally 'huggin_utils::logWeightings(weight_tensor)' logged diffs via 'huggin_utils::checkWeightsForAllSets()'
#             ('huggin_utils::checkWeightsForAllSets()': "WRAPPER to convert full weight values to weight diff")
#   THEN 'logWeights(weight_tensor)' was used to ALSO LOG FULL WEIGHTS (and hence 'graph-weights::collectWeights(...)' has 'percentChg...' legacy)
def collectWeights(weightsLog):
  percentChgDictListDict = {}
  
  try:
    with open(weightsLog) as source :
      print()
      print(f"Filename: {weightsLog}")
      print( "--------")

      textLines = [line.strip() for line in source.readlines() if line.strip()]
  except FileNotFoundError:
    print(f"Filename: {weightsLog} NOT FOUND...exiting")
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

# 24/7/24 DH: Refactored in 'graph-weights-history::printLogitsDict(...)'
def printCollectedDict(percentChgDictListDict):
  for key in percentChgDictListDict:
    print(f"{key}:")

    idx = 0
    for iDict in percentChgDictListDict[key]:
      print(f"  {idx} (ie {weightMapDict[idx]}) % chg by node:")

      weightsSublist = []
      sublistIdx = 0
      for weightsKey in percentChgDictListDict[key][idx]:
        if sublistIdx < 10:
          weightsSublist.append(percentChgDictListDict[key][idx][weightsKey])
        
        sublistIdx += 1
      
      mylist = ""
      for elem in weightsSublist: 
        mylist += f"{elem}, "
      
      lastIdx = list(percentChgDictListDict[key][idx])[-1]
      print(f"    {mylist}...(Idx:{lastIdx})")
      
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
    titleStr = f"Total node weight % change from 'qa_outputs' after {lastEpoch} epochs"

  plt.title(titleStr)

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
  #plt.tight_layout()

  # 16/6/24 DH: 'gWeightsGraphDir' assigned in "if __name__ == "__main__""

  if "complete" in epochNum:
    plt.xlabel("Node number (NOT token ID)")
    plt.ylabel("Weight chg")
    plt.tight_layout()

    totalWeightChgFilename = f"{gWeightsGraphDir}/total-weight-change.png"
    print(f"    SAVING: {totalWeightChgFilename}")
    plt.savefig(totalWeightChgFilename)
  else:
    if weights:
      plt.xlabel("Node number (NOT token ID)")
      plt.ylabel("Weight")
      plt.tight_layout()

      # 1/8/24 DH: Adding in ID marker between graphs
      plt.axhline(y=0.006, color='green', linestyle='dashed', linewidth=0.5)
      plt.axhline(y=-0.003, color='green', linestyle='dashed', linewidth=0.5)

      fullValsFilename = f"{gWeightsGraphDir}/{epochNum}-fullValues.png"
      print(f"    SAVING: {fullValsFilename}")
      plt.savefig(fullValsFilename)
    else:
      plt.xlabel("Node number (NOT token ID)")
      plt.ylabel("Weight chg")
      plt.tight_layout()

      percentDiffsFilename = f"{gWeightsGraphDir}/{epochNum}-percentDiffs.png"
      print(f"    SAVING: {percentDiffsFilename}")
      plt.savefig(percentDiffsFilename)
  print()

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
        #print(f"TOP VALUE: '{weightMapDict[1]}' idx: {sortedIdxs[0]} = {lineDict[sortedIdxs[0]]}")
        cutoff = round(abs(lineDict[sortedIdxs[0]] * cutoffPercent))
        #print(f"CUTOFF FOR INCLUSION: +- {cutoff} (at {cutoffPercent*100}%)")
      else:
        #print(f"TOP VALUE: '{weightMapDict[0]}' idx: {prevSortedIdxs[0]} = {prevLineDict[prevSortedIdxs[0]]}")
        cutoff = round(abs(prevLineDict[prevSortedIdxs[0]] * cutoffPercent))
        #print(f"CUTOFF FOR INCLUSION: +- {cutoff} (at {cutoffPercent*100}%)")
      
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
  
  # 25/5/24 DH: Running 'huggingface/qa.py' AFTER 'huggingface/run_qa.py' OVERWROTE 'weights-full.log' + 'weights.log'
  try:
    # '/Users/doug/src/ai/t5/previous_output_dir-Google-BERT/old-logs/weights-full.log' contains:
    #   EPOCHS: 0, 1025, 1026 (due to length of time to stop with Ctrl-C saving an extra end value)
    # THEREFORE ONLY 1st + 2nd EPOCHS OF THE FILE ARE TAKEN
    startEpoch = keyList[0]

    """
    if "Start" in weightMapDict[lineType]:
      print("  Calculating total weight diffs")
      print("  ------------------------------")
    """

  except IndexError:
    print()
    print("There is no epoch data...exiting")
    exit(0)
  
  # 24/5/24 DH:
  try:
    endEpoch = keyList[1]
    
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
    percentChgFromPrev = round(diff / abs(prevWeight) * 100, 3)

    """ 25/7/24 DH: DEBUG
    #if key == 287:
    selectedKeys = [467, 752, 110, 224, 326, 287, 34, 569]
    if any(map(key.__eq__, selectedKeys)):
      print(f"  {key}: from abs(first epoch): {abs(prevWeight)}, diff: {diff}, % chg: {percentChgFromPrev}")
    """

    percentChgDict[key] = percentChgFromPrev

  return (percentChgDict, endEpoch)

# 22/5/24 DH: WRAPPER around 'getWeightDiffs()' for each line (ie start logits/end logits)
def calcAndGraphTrgDiffs(percentChgDictListDict, lastGraph=True, showGraph=True):
  percentChgLineList = []

  # 'weightMapDict' = "{0: "Start", "Start": 0, ...}"
  for key in weightMapDict.keys():
    
    if isinstance(key, int):
      lineType = key
      
      (percentChgLine, endEpoch) = getWeightDiffs(percentChgDictListDict, lineType)
      percentChgLineList.append(percentChgLine)
  
  # 7/6/24 DH: Now get largest +ve/-ve values from both start + end lines
  (startLineIdxDict, endLineIdxDict) = getLargestValues(percentChgLineList)
  
  print()
  print("  NO LONGER CALLING: 'displayLargestValues(...)'")
  #displayLargestValues(startLineIdxDict, endLineIdxDict)
  print()
  
  # 25/7/24 DH: Added to calc 'startLineIdxDict', 'endLineIdxDict' without showing the graph from 'graph-weights-history.py'
  if showGraph:
    graphWeightsKeyed(percentChgLineList, "complete", lastGraph=lastGraph, lastEpoch=endEpoch)

  return (startLineIdxDict, endLineIdxDict)

# 24/7/24 DH: So can be called via "import graph_weights" (ie 'graph-weights-history.py')
def assignPaths(weightsDir):
  # 19/6/24 DH: 'output_dir' now is 'previous_output_dir-Google-BERT/weights' (FROM: checkpointing.py::weightPath = f"{logPath}/weights")
  #             GIVING: '~/weights/weights-graphs'
  output_dir = os.path.abspath(weightsDir)
  weightsLog = os.path.join(output_dir, gTrainer_log)
  fullweightsLog = os.path.join(output_dir, gTrainer_full_log)
  # 29/7/24 DH:
  roundedLog = os.path.join(output_dir, gTrainer_rounded_log)

  global gWeightsGraphDir
  gWeightsGraphDir = os.path.join(output_dir, "weights-graphs")
  Path(gWeightsGraphDir).mkdir(parents=True, exist_ok=True)

  return (weightsLog, fullweightsLog, roundedLog)

# 16/6/24 DH: https://docs.python.org/3/faq/programming.html#what-are-the-rules-for-local-and-global-variables-in-python
#  "If a variable is assigned a value anywhere within the function’s body, it’s assumed to be a local unless explicitly declared as global."
# HOWEVER: this "main" is NOT A FUNCTION SO ALL VARIABLES ARE GLOBAL 
#          (and leads to "SyntaxError: name ??? is assigned to before global declaration" if used (prob due to python compiler/loader))
if __name__ == "__main__":
  if len(sys.argv) > 1:
    # 24/7/24 DH:
    (weightsLog, fullweightsLog, roundedLog) = assignPaths(sys.argv[1])
  else:
    print(f"You need to provide an '\"output_dir\"/weights' path")
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
  weightDictListDict = collectWeights(fullweightsLog)

  # 14/8/24 DH: Copy "weights-full.log" to "weights-full-0.log" if it contains epoch 0
  #             This is then used for graphing % change during MID-SECTION TRAINING DUE TO CHECKPOINT
  # (https://docs.python.org/3/library/os.html#os.symlink)
  # (https://docs.python.org/3/library/os.html#os.rename)
  # (https://docs.python.org/3/library/shutil.html#shutil.copy)
  if '0' in weightDictListDict.keys():
    # Better to copy than rename
    #os.rename(fullweightsLog, f"{sys.argv[1]}/weights-full-0.log")
    shutil.copy(fullweightsLog, f"{sys.argv[1]}/weights-full-0.log")

  # Firstly graph the 'qa_outputs' full weights
  # 26/5/24 DH: During the Ctrl-C checkpointing save delay (to prevent partial saving) we sometimes get multiple end full weights
  #             ('getWeightDiffs(...) uses "startEpoch = keyList[0]", "endEpoch = keyList[1]")
  keyList = list(weightDictListDict.keys())
  startEpoch = keyList[0]
  # 'endEpoch' is taken as 2nd in "weights-full.log" (see comment above)
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

  # Display full weights per node of start/end lines for 'startEpoch' (ie 0)
  graphWeightsKeyed(weightDictListDict[startEpoch], startEpoch, weights=True)
  # Display full weights per node of start/end lines for 'endEpoch' (ie 19 for custom JSON)
  graphWeightsKeyed(weightDictListDict[endEpoch], endEpoch, weights=True)

  # Now calculate + graph the percentage diff
  # 14/8/24 DH: If 'startEpoch' is NOT ZERO then repopulate 'weightDictListDict' with epoch 0 from "weights-full-0.log"
  if startEpoch != '0':
    zeroWeightDLDict = collectWeights(f"{sys.argv[1]}/weights-full-0.log")

    zeroKeyList = list(zeroWeightDLDict.keys())
    for zeroKey in zeroKeyList:
      if zeroKey != '0':
        print(f"Deleting '{zeroKey}' from 'zeroWeightDLDict'")
        del zeroWeightDLDict[zeroKey]

    zeroWeightDLDict[endEpoch] = weightDictListDict[endEpoch]
    (startLineIdxDict, endLineIdxDict) = calcAndGraphTrgDiffs(zeroWeightDLDict)

  else:
    (startLineIdxDict, endLineIdxDict) = calcAndGraphTrgDiffs(weightDictListDict)

  if not gShowFlag:
    print()
    print("NOT SHOWING images (please add 'show' to cmd line args if images wanted)")
    print()

  
  
