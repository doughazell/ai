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

def pruneWeights(weightsDictListDict, epochsWanted, startNodes, endNodes):
  deleteEpochs = []
  deleteNodes = []

  for epoch in weightsDictListDict:
    # Test against list FROM: 'graph-node-logits::graphLogitsByLayer(...)' (see comment re "__eq__" vs "__contains__")
    if not any(map(int(epoch).__eq__, epochsWanted)):
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
      prevChgDLDict = {}
      prevChgDLDict['epoch'] = epoch
      prevChgDLDict[epoch] = chgDictListDict[epoch]
    
  # END: --- "for epoch in chgDictListDict" ---

  return totalChgDLDict

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
  
  print()
  print("Individual weight chgs by epoch (using 'graph-weights::printCollectedDict(...)')")
  print("-------------------------------")
  graph_weights.printCollectedDict(weightsDictListDict)

  # DEV shortcut
  graph_weights.gShowFlag = True

  # 24/7/24 DH: Prune weights for 10 BIGGEST CHANGING nodes
  #   ==> Get values in 'graph_weights.getLargestValues(...)' via:
  (weightsLog, fullweightsLog) = graph_weights.assignPaths(sys.argv[1])
  weightDictListDict = graph_weights.collectWeights(fullweightsLog)
  (startLineIdxDict, endLineIdxDict) = graph_weights.calcAndGraphTrgDiffs(weightDictListDict, lastGraph=False)

  startNodes = list(startLineIdxDict.keys())
  endNodes = list(endLineIdxDict.keys())
  print(f"Start nodes: {startNodes}")
  print(f"End nodes: {endNodes}")

  # DEV: Prev this was done with "weights-full.log" using 'calcAndGraphTrgDiffs(...)' (and we are USING "weights.log" ie ONLY DIFFS)
  rollingWeightChgDLDict = getRollingWeightChgs(weightsDictListDict)

  # 24/7/24 DH: Having got total adjacent values for each of 768 nodes for start + end logits we now select epochs for DEV CHECK
  epochsWanted = [2, 10, 19]
  pruneWeights(rollingWeightChgDLDict, epochsWanted, startNodes, endNodes)

  print()
  print("Rolling total weight chgs by epoch")
  print("----------------------------------")
  printWeightChgDict(rollingWeightChgDLDict, nodeNumber=768)

  # 24/7/24 DH: TODO: Graph chosen nodes over all epochs

  print()
  print("PRESS RETURN TO FINISH", end='')
  response = input()
  
  