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
  
  with open(logitsLog) as source :
    print()
    print("Filename: ", logitsLog)
    print("---------")

    textLines = [line.strip() for line in source.readlines() if line.strip()]
  
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

  return recordsDict

def displayLogits(recordsDict):
  print()
  print(f"recordsDict")
  print( "-----------")
  for key in recordsDict:
    if "logits" in key:
      print(f"{key}:")
      for logitsKey in recordsDict[key]:
        #logitsVals = recordsDict[key][logitsKey]
        logitsVals = ""
        for val in recordsDict[key][logitsKey]:
          logitsVals += f"{val}, "

        print(f"  {logitsKey}: {logitsVals}")
    else:
      print(f"{key}: {recordsDict[key]}")

# Taken from: 'ai/huggingface/stop_trainer_utils.py'
def graphLogits(recordsDict):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  plt.title("Token values from different training epochs")
  plt.xlabel("Token ID")
  plt.ylabel("Logit value")

  xValNum = recordsDict['input_ids'].shape[0]
  xVals = range(xValNum + 2)
  yVals = recordsDict['start_logits']['1']

  plt.plot(xVals, yVals, label="Token values")
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  #plt.draw()
  plt.show()

if __name__ == "__main__":
  recordsDict = collectLogits()
  displayLogits(recordsDict)
  graphLogits(recordsDict)