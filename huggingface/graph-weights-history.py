# 23/7/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

"""
BASED ON: 'graph-weights.py' which uses f"{logPath}/weights" created by 'checkpointing::createLoggers(...)'

It creates:
  full weights graph of 768 nodes at:
    start of training
    end of training
  weight chg graph of 768 nodes

NEED to track the weight chg of 10 BIGGEST CHANGING (found by 'graph-weights::getLargestValues(...)' nodes by epoch from the start

ISSUE: "weights.log" for each epoch is PERCENTAGE CHANGE from prev epoch...!!!
  OPTION 1: Keep running weight value from start [WORKS]
  OPTION 2: Have rolling weight chg [CHOSEN, but leads to "63ln(x)+111" shift from % chg from start (29/7/24)]

DESIGN:
Data structure [Dict-List-Dict] (code topo)
-------------------------------
***********************************************************************************************************************
*** In retro-spect it would have been BETTER TO USE "Dict-Dict-Dict" (rather than have a List for start/end logits) ***
***********************************************************************************************************************

1) Dict :: 'percentChgDictListDict'
   KEY :: Epoch
   2) List
      3-0) Dict :: Start logits
           KEY :: Idx of (768 Node weight in Final NN Layer)
      3-1) Dict :: End logits
           KEY :: Idx of (768 Node weights in Final NN Layer)

'graph-weights.collectWeights("weights.log")'
---------------------------------------------
  eg "19-End: [0.046, -0.002, -0.008, 0.015, ...]"
    
    Get epoch
    3-0, 3-1) Add each element of weights array to Dict with index KEY
    2) Then add Nodes Dict to a List in order used by HuggingFace (with 1st entry being "Start" + 2nd entry being "End")
    1) Then add "Start/End" List to Dict ('percentChgDictListDict') with epoch KEY

'calcTotalChg(chgDictListDict, epoch, prevChgDLDict, prevEpoch)'
----------------------------------------------------------------
  LOOP THROUGH 'lineType' Dict
    for nodeKey in weightsDLDict[currEpoch][lineType]:
      totalChgDict[nodeKey] = currNode + prevNode

    # Add dict for each (start/end) logits line
    totalChgLDict.append(totalChgDict)

  return totalChgLDict

"""

# 23/7/24 DH: NEEDED: $ cd ~/huggingface; ln -s graph-weights.py graph_weights.py
from graph_weights import *
import graph_weights
# 28/7/24 DH: NEEDED: $ cd ~/huggingface; ln -s create-weights-csv.py create_weights_csv.py
from create_weights_csv import *

import numpy as np
from scipy.interpolate import interp1d

def pruneWeights(weightsDictListDict, epochsWanted, startNodes, endNodes):
  deleteEpochs = []
  deleteNodes = []

  for epoch in weightsDictListDict:
    # Test against list FROM: 'graph-node-logits::graphLogitsByLayer(...)' (see comment re "__eq__" vs "__contains__")
    if epochsWanted and not any(map(int(epoch).__eq__, epochsWanted)):
      # Delete list mechanism from: 'graph-logits::pruneLogits(...)'
      deleteEpochs.append(epoch)
    
    # Now delete unwanted Start logit Nodes
    # -------------------------------------
    startIdx = graph_weights.weightMapDict['Start']
    for node in weightsDictListDict[epoch][startIdx]:
      if not any(map(int(node).__eq__, startNodes)):
        deleteNodes.append(node)
        
    for node in deleteNodes:
      try:
        del weightsDictListDict[epoch][startIdx][node]
      except KeyError as e:
        print(f"Unable to delete node KEY: {node} from epoch: {epoch}, {graph_weights.weightMapDict[startIdx]} logits ")

    # Reset nodes to delete
    deleteNodes = []
    # -------------------------------------

    # Now delete unwanted End logit Nodes
    # -------------------------------------
    endIdx = graph_weights.weightMapDict['End']
    for node in weightsDictListDict[epoch][endIdx]:
      if not any(map(int(node).__eq__, endNodes)):
        deleteNodes.append(node)
        
    for node in deleteNodes:
      try:
        del weightsDictListDict[epoch][endIdx][node]
      except KeyError as e:
        print(f"Unable to delete node KEY: {node} from epoch: {epoch}, {graph_weights.weightMapDict[endIdx]} logits ")

    # Reset nodes to delete
    deleteNodes = []
    # -------------------------------------

  # END: --- "for epoch in weightsDictListDict" ---

  for epoch in deleteEpochs:
    try:
      del weightsDictListDict[epoch]
    except KeyError as e:
      print(f"Unable to delete epoch KEY: {epoch}")

def calcTotalChg(weightsDLDict, currEpoch, prevWeightsDLDict, prevEpoch):
  totalChgLDict = []
  
  #print(f"Calculating totals between epochs: {prevEpoch} and {currEpoch}")

  for key in weightMapDict.keys():
    # 'weightMapDict' = "{0: "Start", "Start": 0, ...}"
    if isinstance(key, int):
      lineType = key

      totalChgDict = {}
      for nodeKey in weightsDLDict[currEpoch][lineType]:

        currNode = weightsDLDict[currEpoch][lineType][nodeKey]
        prevNode = prevWeightsDLDict[prevEpoch][lineType][nodeKey]

        totalChgDict[nodeKey] = currNode + prevNode
      # END --- "for nodeKey in weightsDLDict[currEpoch][lineType]" ---

      # Add dict for each (start/end) logits line
      totalChgLDict.append(totalChgDict)
      
    # END --- "if isinstance(key, int)" ---

  return totalChgLDict

def getRollingWeightChgs(chgDictListDict):
  prevChgDLDict = None
  totalChgDLDict = {}

  for epoch in chgDictListDict:

    if prevChgDLDict:
      # When just using epoch as key (now saving epoch in key 'epoch')
      #prevEpoch = list(prevWeightsDLDict.keys())[0]

      prevEpoch = prevChgDLDict['epoch']

      totalChgDLDict[epoch] = calcTotalChg(chgDictListDict, epoch, prevChgDLDict, prevEpoch)
      
      # Now reset the prev to the last totals
      prevChgDLDict['epoch'] = epoch
      prevChgDLDict[epoch] = totalChgDLDict[epoch]
      
    else:
      # 27/7/24 DH: Total chg for first epoch is just the first value
      totalChgDLDict[epoch] = chgDictListDict[epoch]

      prevChgDLDict = {}
      prevChgDLDict['epoch'] = epoch
      prevChgDLDict[epoch] = chgDictListDict[epoch]
    
  # END: --- "for epoch in chgDictListDict" ---

  # 28/7/24 DH: Now copying 'create-weights-csv.py' (but for total "rolling" changes rather than incremental changes by epoch)
  startLineCSVfilename = "rollingChgs-start.csv"
  endLineCSVfilename = "rollingChgs-end.csv"
  
  writeCSVdict(totalChgDLDict, startLineCSVfilename, endLineCSVfilename)

  return totalChgDLDict

# IDEAS FROM: 'graph-weights::printCollectedDict(...)'
def printWeightChgDict(chgDLDict, nodeNumber=10):
  print()

  for key in chgDLDict.keys():
    print(f"'chgDLDict' epoch: {key}")
    print("----------")
    
    elemIdx = 0
    # 'chgDLDict[key]' is the LIST of (start/end) logit lines
    for logitLineDict in chgDLDict[key]:
      print(f"{weightMapDict[elemIdx]} logits")
      elemIdx += 1
      
      valsStr = ""
      for nodeKey in logitLineDict:
        if nodeNumber > 10: # ie pruned nodes so not want only default of first 10
          valsStr += f"{nodeKey}: {round(logitLineDict[nodeKey],3)}, "

        elif nodeKey < nodeNumber:
          valsStr += f"{round(logitLineDict[nodeKey],3)}, "

      lastIdx = list(logitLineDict)[-1]
      print(f"  {valsStr}...(Idx:{lastIdx})")
    print("----------")
    print()

# 25/7/24 DH: https://matplotlib.org/stable/gallery/mplot3d/index.html#d-plotting
#             https://matplotlib.org/stable/gallery/mplot3d/surface3d.html#sphx-glr-gallery-mplot3d-surface3d-py
def graphChosenNodes(chgDLDict, startNodes, endNodes):
  plt.figure()
  plt.title("Total 'rolling' weight over training epochs for selected nodes")
  plt.xlabel("Epoch (soon to be z-axis of 3D graph)")
  plt.ylabel("Weight")
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)
  
  # Start Logits / End Logits
  #  xVals = nodeIdx key (initially just biggest 10)
  #  yVals = node value (at KEY xVals)
  #  zVals = epochs (therefore "z, y" graphs located at KEY xVals, later changing to sphere with ALL xVals)

  zVals = list(chgDLDict.keys())

  for node in startNodes:
    # Access DLD data structure for:
    #   1) Node 'node', 2) in End Logits, 3) for each epoch
    yVals = [chgDLDict[epoch][gStartIdx][node] for epoch in zVals]

    plt.plot(zVals, yVals, label=f"Start: node {node}")
  # END: --- "for node in startNodes" ---

  for node in endNodes:
    # Access DLD data structure for:
    #   1) Node 'node', 2) in End Logits, 3) for each epoch
    yVals = [chgDLDict[epoch][gEndIdx][node] for epoch in zVals]

    plt.plot(zVals, yVals, label=f"  End: node {node}", linestyle='dashed')
  # END: --- "for node in endNodes" ---
  
  if graph_weights.gShowFlag:
    #plt.legend(loc="best")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show(block=False)

  """ FROM: https://matplotlib.org/stable/gallery/mplot3d/2dcollections3d.html#sphx-glr-gallery-mplot3d-2dcollections3d-py
  #ax = plt.figure().add_subplot(projection='3d')
  ax = plt.figure().add_subplot()

  # Plot a sin curve using the x and y axes.
  x = np.linspace(0, 1, 100)
  y = np.sin(x * 2 * np.pi) / 2 + 0.5
  #print(f"x: {x}")
  #print(f"y: {y}")
  
  #ax.plot(x, y, zs=10, zdir='z', label='curve in (x, y)')
  # Use defaults for z axis
  ax.plot(x, y, label='curve in (x, y)')

  # Make legend, set axes limits and labels
  ax.legend()
  #ax.set_xlim(0, 1)
  #ax.set_ylim(0, 1)
  #ax.set_zlim(0, 1)
  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  #ax.set_zlabel('Z')

  # Customize the view angle so it's easier to see that the scatter points lie
  # on the plane y=0
  #ax.view_init(elev=20., azim=-35, roll=0)

  plt.show(block=False)
  """

def graphNode(chgDLDict, typeIdx, nodeNum):
  plt.figure()
  plt.title(f"Total 'rolling' weight over training epochs for node {nodeNum}")
  plt.xlabel("Epoch (soon to be z-axis of 3D graph)")
  plt.ylabel("Weight")
  
  zVals = list(chgDLDict.keys())
  #yVals = [round(chgDLDict[epoch][typeIdx][nodeNum]) for epoch in zVals]
  yVals = [chgDLDict[epoch][typeIdx][nodeNum] for epoch in zVals]

  plt.plot(zVals, yVals, label=f"{weightMapDict[typeIdx]}: node {nodeNum}")

  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)
  plt.axvline(x=1, color='red', linestyle='dashed', linewidth=0.5)
  plt.axvline(x=2, color='red', linestyle='dashed', linewidth=0.5)
  plt.axvline(x=3, color='red', linestyle='dashed', linewidth=0.5)

  if graph_weights.gShowFlag:
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show(block=False)

# 30/7/24 DH:
def getInitialWeights(fullweightDictListDict, startNodes, endNodes):
  initWeightsDictList = []

  startEpoch = '0'

  initWeightsDict = {}
  for node in startNodes:
    initWeightsDict[node] = fullweightDictListDict[startEpoch][gStartIdx][node]
    
  initWeightsDictList.append(initWeightsDict)
  
  initWeightsDict = {}
  for node in endNodes:
    initWeightsDict[node] = fullweightDictListDict[startEpoch][gEndIdx][node]
  
  initWeightsDictList.append(initWeightsDict)
  
  return initWeightsDictList

def graphGradient(nodeWeights, xLabel, xVals, yLabel, yVals):
  plt.figure()
  plt.title(f"Gradient of the change in node weight over epoch")
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  
  """
  print()
  print("  Node values")
  print("  -----------")
  print(f"  {nodeWeights}")
  print()

  print("  Gradient values")
  print("  ---------------")
  print(f"  xVals: {xVals}")
  print(f"  yVals: {yVals}")
  print()
  """

  plt.plot(xVals, yVals)
  
  if xLabel == "Epoch":
    plt.xticks(xVals)
    
    plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)
    plt.axvline(x=1, color='red', linestyle='dashed', linewidth=0.5)
    plt.axvline(x=2, color='red', linestyle='dashed', linewidth=0.5)
    plt.axvline(x=3, color='red', linestyle='dashed', linewidth=0.5)
  else:
    plt.yticks(yVals) # 'yVals' are "x values"
    
    plt.axvline(x=0, color='green', linestyle='dashed', linewidth=0.5)
    plt.axvline(x=-0.00010, color='green', linestyle='dashed', linewidth=0.5)
    plt.axhline(y=1, color='red', linestyle='dashed', linewidth=0.5)
    plt.axhline(y=2, color='red', linestyle='dashed', linewidth=0.5)
    plt.axhline(y=3, color='red', linestyle='dashed', linewidth=0.5)

  if graph_weights.gShowFlag:
    plt.tight_layout()
    plt.show(block=False)
  
# 31/7/24 DH:
def getInterpolatedVals(xVals, yVals):
  print()
  print("'getTrend()'")
  print("------------")
  print()
  # 31/7/24 DH: Interpolate data (https://docs.scipy.org/doc/scipy/tutorial/interpolate.html)
  #             (Conceptual issue here, since graphs are drawn purely for interpolation (heuristically or data extraction) 
  #              but 'matplotlib' just displays data [Matplotlib = Model -> View -> NOT Enhanced Model] )

  # DEFAULT: Just replicating 'np.gradient()'
  funcInput = xVals
  print(f"DEFAULT: Func input ({funcInput.__class__}):  {funcInput}")

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
  # "This class is considered legacy and will no longer receive updates."
  #interp1d_func = interp1d(xVals, yVals)
  #interpOutput = interp1d_func(funcInput)
  #print(f"'scipy.interp1d()' output ({interpOutput.__class__}): {interpOutput}")

  # https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#legacy-interface-for-1-d-interpolation-interp1d
  # "The ‘cubic’ kind of 'interp1d()' is equivalent to 'make_interp_spline()', and the ‘linear’ kind is equivalent to 'np.interp()'"
  interpOutput = np.interp(funcInput, xVals, yVals)
  print(f"'np.interp()' output ({interpOutput.__class__}): {interpOutput}")

def displayTrends(trendDict):
  # ID: "ZERO" -> "POSITIVE"
  # ID: "ZERO" -> "-ve"
  print(f"  TREND IS: {trendDict['trend'].lower()}")

  trendDict['toPos'] = []
  trendDict['toNeg'] = []
  for zero in trendDict['zero']:
    if trendDict[zero+1] == "POSITIVE":
      trendDict['toPos'].append(zero)
    if trendDict[zero+1] == "-ve":
      trendDict['toNeg'].append(zero)

  if len(trendDict['toPos']) > 0:
    print(f"    ZERO to POSITIVE nodes: {trendDict['toPos']}")
  
  if len(trendDict['toNeg']) > 0:
    print(f"    ZERO to -ve nodes: {trendDict['toNeg']}")

# 31/7/24 DH:
def getTrend(gradPts, gradList):
  trendDict = {}
  trendDict['plusCnt'] = 0
  trendDict['negCnt'] = 0
  trendDict['zero'] = []

  for idx in range(gradPts):
    currVal = gradList[idx]

    if currVal <= 0:
      trendDict['negCnt'] += 1
      trendDict[idx] = "-ve"
    else:
      trendDict['plusCnt'] += 1
      trendDict[idx] = "POSITIVE"
    
    if abs(currVal) < 0.00002:
      trendDict[idx] = "ZERO"
      trendDict['zero'].append(idx)
    
    #print(f"  +ve: {trendDict['plusCnt']}, -ve: {trendDict['negCnt']}) {idx}: {currVal} ({trendDict[idx]})")
    
  # END: --- "for idx in range(gradPts)" ---

  if trendDict['plusCnt'] > trendDict['negCnt']:
    trendDict['trend'] = "POSITIVE"
  else:
    trendDict['trend'] = "-ve"

  displayTrends(trendDict)
  

# 30/7/24 DH:
def getInflectionPoints(rollingWeightsFromStartDLD, startNodes, endNodes):
  print()
  print("'getInflectionPoints()'")
  print("-----------------------")

  epochs = list(rollingWeightsFromStartDLD.keys())
  firstEpoch = epochs[0]
  
  print("'startNodes':")
  print("-------------")
  # ---------------------------------------------------------------------------------------
  for node in startNodes:
    print(f"{node}")
 
    nodeWeights = [rollingWeightsFromStartDLD[epoch][gStartIdx][node] for epoch in epochs]

    npWeights = np.array(nodeWeights)
    gradList = np.gradient(npWeights, edge_order=1)
    gradPts = gradList.shape[0]

    getTrend(gradPts, gradList)

  print()
  print("'endNodes':")
  print("-----------")
  # ---------------------------------------------------------------------------------------
  for node in endNodes:
    print(f"{node}")

    nodeWeights = [rollingWeightsFromStartDLD[epoch][gEndIdx][node] for epoch in epochs]

    npWeights = np.array(nodeWeights)
    gradList = np.gradient(npWeights, edge_order=1)
    gradPts = gradList.shape[0]

    getTrend(gradPts, gradList)

    """ GRAPHING GRADIENT
    xLabel = "Epoch"
    xVals = np.arange(0, gradPts, step=1)
    yLabel = "Gradient"
    yVals = gradList
    
    graphGradient(nodeWeights, xLabel, xVals, yLabel, yVals)
    #getInterpolatedVals(xVals, yVals)
    
    # Switch axis
    funcInput = [-0.0001, -0.00005, 0, 0.00005]
    #graphGradient(nodeWeights, yLabel, yVals, xLabel, xVals, funcInput)
    """
    

if __name__ == "__main__":
  if len(sys.argv) > 1:
    (weightsLog, fullweightsLog, roundedLog) = assignPaths(sys.argv[1])

    print(f"'{weightsLog}' is PERCENT CHG weights (from 'huggin_utils::checkWeightsForAllSets():' weightStats = getWeightStats(idx) )")

    # 23/7/24 DH: Full weights are ONLY TAKEN at the start + end epochs NOT EVERY EPOCH (which is weight diffs)

  else:
    print(f"You need to provide an '\"output_dir\"/weights' path")
    exit(0)
  
  # 29/7/24 DH:
  #weightsDictListDict = graph_weights.collectWeights(weightsLog)
  weightsDictListDict = graph_weights.collectWeights(roundedLog)
  
  """ DEV
  print()
  print("Individual weight by epoch (using 'graph-weights::printCollectedDict(...)')")
  print("-------------------------------")
  graph_weights.printCollectedDict(weightsDictListDict)
  """

  # DEV shortcut
  graph_weights.gShowFlag = True

  #                      GET 10 BIGGEST CHANGING NODES
  # =============================================================================
  # 24/7/24 DH: Get values in 'graph_weights.getLargestValues(...)' via:
  fullweightDictListDict = graph_weights.collectWeights(fullweightsLog)
  
  showGraph = True
  (startLineIdxDict, endLineIdxDict) = graph_weights.calcAndGraphTrgDiffs(fullweightDictListDict, lastGraph=False, showGraph=showGraph)
  startNodes = list(startLineIdxDict.keys())
  endNodes = list(endLineIdxDict.keys())
  # -----------------------------------------------------------------------------

  # 30/7/24 DH:
  epoch0weights = getInitialWeights(fullweightDictListDict, startNodes, endNodes)

  #                     THEN GET ROLLING WEIGHT CHANGES
  # =============================================================================
  # DEV: Prev this was done with "weights-full.log" using 'calcAndGraphTrgDiffs(...)' (and we are USING "weights.log" ie ONLY DIFFS)

  rollingWeightsChgDLDict = getRollingWeightChgs(weightsDictListDict)

  # 24/7/24 DH: Having got total adjacent values for each of 768 nodes for start + end logits we now select epochs for DEV CHECK
  #epochsWanted = [2, 10, 19]
  epochsWanted = None # ie all epochs wanted (just pruning unwanted nodes from 10 largest changing ones)
  pruneWeights(rollingWeightsChgDLDict, epochsWanted, startNodes, endNodes)

  """ DEV
  print()
  print("Rolling total weight chgs by epoch")
  print("----------------------------------")
  printWeightChgDict(rollingWeightChgDLDict, nodeNumber=768)
  """

  # 30/7/24 DH: Need to add at key start (since 'Dict.keys()' is insertion order)
  rollingWeightsFromStartDLD = {}
  rollingWeightsFromStartDLD[0] = epoch0weights
  rollingWeightsFromStartDLD.update(rollingWeightsChgDLDict)

  # 24/7/24 DH: Graph chosen nodes over all epochs (incl Epoch 0)
  graphChosenNodes(rollingWeightsFromStartDLD, startNodes, endNodes)
  
  # 30/7/24 DH: Points of inflection curves
  # ---------------------------------------
  """
  for node in startNodes:
    graphNode(rollingWeightsFromStartDLD, gStartIdx, node)

  for node in endNodes:
    graphNode(rollingWeightsFromStartDLD, gEndIdx, node)
  """
  
  getInflectionPoints(rollingWeightsFromStartDLD, startNodes, endNodes)

  print()
  print("NEED:")
  print("  1) Absolute chg + epoch 0 of all nodes")
  # https://huggingface.co/brunokreiner
  # https://github.com/huggingface/transformers/issues/11047
  print("  2) Weights from non-Pretrained Bert")

  if showGraph:
    print()
    print("PRESS RETURN TO FINISH", end='')
    response = input()
  
  