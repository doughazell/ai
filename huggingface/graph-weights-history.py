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
  OPTION 1: Keep running weight value from start
  OPTION 2: Have rolling weight chg [CHOSEN]

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
  plt.title("Total 'rolling' weight chg over training epochs for selected nodes")
  plt.xlabel("Epoch (soon to be z-axis of 3D graph)")
  plt.ylabel("Weight chg")
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)
  
  # Start Logits / End Logits
  #  xVals = nodeIdx key (initially just biggest 10)
  #  yVals = node value (at KEY xVals)
  #  zVals = epochs (therefore "z, y" graphs located at KEY xVals, later changing to sphere with ALL xVals)

  zVals = list(chgDLDict.keys())

  print()
  print(f"Start nodes: {startNodes}")
  for node in startNodes:
    # Access DLD data structure for:
    #   1) Node 'node', 2) in End Logits, 3) for each epoch
    yVals = [round(chgDLDict[epoch][gStartIdx][node]) for epoch in zVals]
    print(f"  {node}: {yVals}")

    plt.plot(zVals, yVals, label=f"Start: node {node}")
  # END: --- "for node in startNodes" ---

  print()
  print(f"End nodes: {endNodes}")
  for node in endNodes:
    # Access DLD data structure for:
    #   1) Node 'node', 2) in End Logits, 3) for each epoch
    yVals = [round(chgDLDict[epoch][gEndIdx][node]) for epoch in zVals]
    print(f"  {node}: {yVals}")

    plt.plot(zVals, yVals, label=f"  End: node {node}", linestyle='dashed')
  # END: --- "for node in endNodes" ---
  
  if graph_weights.gShowFlag:
    #plt.legend(loc="best")
    plt.legend(loc="lower right")
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
  plt.title(f"Total 'rolling' weight chg over training epochs for node {nodeNum}")
  plt.xlabel("Epoch (soon to be z-axis of 3D graph)")
  plt.ylabel("Weight chg")
  
  zVals = list(chgDLDict.keys())
  yVals = [round(chgDLDict[epoch][typeIdx][nodeNum]) for epoch in zVals]

  plt.plot(zVals, yVals, label=f"{weightMapDict[typeIdx]}: node {nodeNum}")

  if graph_weights.gShowFlag:
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show(block=False)
  

if __name__ == "__main__":
  if len(sys.argv) > 1:
    # 19/6/24 DH: 'output_dir' now is 'previous_output_dir-Google-BERT/weights' (FROM: checkpointing.py::weightPath = f"{logPath}/weights")
    #             GIVING: '~/weights/weights-graphs'
    output_dir = os.path.abspath(sys.argv[1])

    #graph_weights.gTrainer_log = "weights.log"
    weightsLog = os.path.join(output_dir, gTrainer_log)
    print(f"'{gTrainer_log}' is PERCENT CHG weights (from 'huggin_utils::checkWeightsForAllSets():' weightStats = getWeightStats(idx) )")

    # 23/7/24 DH: Full weights are ONLY TAKEN at the start + end epochs NOT EVERY EPOCH (which is weight diffs)

    graph_weights.gWeightsGraphDir = os.path.join(output_dir, "weights-graphs")
    Path(graph_weights.gWeightsGraphDir).mkdir(parents=True, exist_ok=True)
  else:
    print(f"You need to provide an '\"output_dir\"/weights' path")
    exit(0)
  
  weightsDictListDict = graph_weights.collectWeights(weightsLog)
  
  """ DEV
  print()
  print("Individual weight chgs by epoch (using 'graph-weights::printCollectedDict(...)')")
  print("-------------------------------")
  graph_weights.printCollectedDict(weightsDictListDict)
  """

  # DEV shortcut
  graph_weights.gShowFlag = True

  # 24/7/24 DH: Prune weights for 10 BIGGEST CHANGING nodes
  #   ==> Get values in 'graph_weights.getLargestValues(...)' via:
  (weightsLog, fullweightsLog) = graph_weights.assignPaths(sys.argv[1])
  fullweightDictListDict = graph_weights.collectWeights(fullweightsLog)
  
  showGraph = True
  (startLineIdxDict, endLineIdxDict) = graph_weights.calcAndGraphTrgDiffs(fullweightDictListDict, lastGraph=False, showGraph=showGraph)
  startNodes = list(startLineIdxDict.keys())
  endNodes = list(endLineIdxDict.keys())

  # DEV: Prev this was done with "weights-full.log" using 'calcAndGraphTrgDiffs(...)' (and we are USING "weights.log" ie ONLY DIFFS)
  rollingWeightChgDLDict = getRollingWeightChgs(weightsDictListDict)

  # 24/7/24 DH: Having got total adjacent values for each of 768 nodes for start + end logits we now select epochs for DEV CHECK
  #epochsWanted = [2, 10, 19]
  epochsWanted = None # ie all epochs wanted (just pruning unwanted nodes from 10 largest changing ones)
  pruneWeights(rollingWeightChgDLDict, epochsWanted, startNodes, endNodes)

  """ DEV
  print()
  print("Rolling total weight chgs by epoch")
  print("----------------------------------")
  printWeightChgDict(rollingWeightChgDLDict, nodeNumber=768)
  """

  # 24/7/24 DH: Graph chosen nodes over all epochs
  graphChosenNodes(rollingWeightChgDLDict, startNodes, endNodes)
  graphNode(rollingWeightChgDLDict, gEndIdx, 287)

  if showGraph:
    print()
    print("PRESS RETURN TO FINISH", end='')
    response = input()
  
  