# 27/7/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

"""
Steps necessary to change default "weights.log"
-----------------------------------------------
eg 
"1-Start: [0.13, -0.168, 0.278, 0.119, ...]"
"1-End: [-0.726, 0.081, 0.171, -0.175, ...]"

1) Remove: "[]"
2) Create 2 csv's for for "Start", another for "End" :
   "weights-start.csv" (OpenOffice Spreadsheet needs ".csv" to prevent loading as doc)
   "weights-end.csv"
3) Change "1-Start:" to "1," when populating correct 'csv'
"""

# 23/7/24 DH: NEEDED: $ cd ~/huggingface; ln -s graph-weights.py graph_weights.py
from graph_weights import *
import graph_weights

def printHeadings(csvFile, lastNodeKey):
  csvFile.write(f"Epoch,")
  for cnt in range(lastNodeKey+1):
    csvFile.write(f"{cnt},")
  
  csvFile.write(f"\n")

def printAdaptedLine(csvFile, key, logitLineDict):
  valsStr = ""
  for nodeKey in logitLineDict:
    valsStr += f"{round(logitLineDict[nodeKey],3)}, "

  csvFile.write(f"{key}, {valsStr}\n")

# IDEAS FROM: 'graph-weights-history::printWeightChgDict(...)'
def printCSVdict(weightsDictListDict):
  startLineCSVfile = open("weights-start.csv", "w")
  endLineCSVfile = open("weights-end.csv", "w")

  firstEpochKey = list(weightsDictListDict)[0]
  firstNodeDict = weightsDictListDict[firstEpochKey][gStartIdx]
  lastNodeKey = list(firstNodeDict)[-1]

  printHeadings(startLineCSVfile, lastNodeKey)
  printHeadings(endLineCSVfile, lastNodeKey)

  for key in weightsDictListDict.keys():
    
    elemIdx = 0
    # 'chgDLDict[key]' is the LIST of (start/end) logit lines
    for logitLineDict in weightsDictListDict[key]:

      if weightMapDict[elemIdx] == "Start":
        printAdaptedLine(startLineCSVfile, key, logitLineDict)
      if weightMapDict[elemIdx] == "End":
        printAdaptedLine(endLineCSVfile, key, logitLineDict)

      elemIdx += 1

if __name__ == "__main__":
  if len(sys.argv) > 1:
    # 19/6/24 DH: 'output_dir' now is 'previous_output_dir-Google-BERT/weights' (FROM: checkpointing.py::weightPath = f"{logPath}/weights")
    #             GIVING: '~/weights/weights-graphs'
    output_dir = os.path.abspath(sys.argv[1])

    #graph_weights.gTrainer_log = "weights.log"
    weightsLog = os.path.join(output_dir, gTrainer_log)
    print(f"'{gTrainer_log}' is PERCENT CHG weights (from 'huggin_utils::checkWeightsForAllSets():' weightStats = getWeightStats(idx) )")

    # 23/7/24 DH: Full weights are ONLY TAKEN at the start + end epochs NOT EVERY EPOCH (which is weight diffs)

  else:
    print(f"You need to provide an '\"output_dir\"/weights' path")
    exit(0)
  
  weightsDictListDict = graph_weights.collectWeights(weightsLog)

  printCSVdict(weightsDictListDict)
  print()
