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
from pathlib import Path

# 8/6/24 DH: Hard-coded to prevent needing to add: "HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))" code
# (ORIG LOG USED IN 'graph-logits.py': trainer_log = "seq2seq_qa_INtrainer.log")
gTrainer_log = "weights/node287-logits.log"
gOutputDir = None
# 9/7/24 DH: Path refactor so can be run from any dir
gCWD = Path.cwd()

# 12/6/24 DH: Future-proofing the model output log (when splitting on "-")
# 'huggin_utils.py::logSelectedNodeLogits(...)' :
#   Training: "{epochNum}-{bert_cnt}-{bertLayerName}"
#   Non-training:       "-{bert_cnt}-{bertLayerName}-{embedTokens}"
gEpochIdx = 0
gLayerIdx = 1
gNameIdx  = 2
gTokenIdx = 3

# 16/6/24 DH: Now wanting to just save graphs (rather than display them as default)
gShowFlag = False

def collectLogits():
  recordsDict = {}
  tokenLens = []

  # 16/6/24 DH: Changed from "if len(sys.argv) == 2" due to having cmd line arg to "show" graphs (in order to use the saved versions)
  if len(sys.argv) > 1:
    global gOutputDir
    gOutputDir = os.path.abspath(sys.argv[1])
    logitsLog = os.path.join(gOutputDir, gTrainer_log)
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

    # TRAINING EXAMPLE:     "0-12: [-0.8225892782211304, -0.24359458684921265, ...]"
    # NON-TRAINING EXAMPLE:  "-1-154: [0.0373508557677269, 0.10611902177333832, ...]"
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
        # 20/6/24 DH: 'epochLayerCnt' now contains Token Len for non-training run so "Epoch-LayerCnt-TokenLen"
        epochLayerCnt = lineSplit[0]
        linePart = lineSplit[1]
        
        # 10/6/24 DH: For a non-training run there are no epochs so multiple runs have the same key 
        #             (with 'recordsDict' just holding the last run if not checked for existing key entry)
        # 11/6/24 DH: Prob no longer necessary with addition of Token Length in 'huggin_utils.py::logSelectedNodeLogits(...)'
        # 20/6/24 DH: There are some SQuAD entries with same token len (eg 127)
        if epochLayerCnt in recordsDict:
          print(f"{epochLayerCnt} already added")
          # Repeated key clashes will result in "token len.2.2" (which may then cause probs below with 'float()')
          epochLayerCnt = f"{epochLayerCnt}.2"
          print(f"  so adding: {epochLayerCnt}")
  
        recordsDict[epochLayerCnt] = np.array( literal_eval(linePart) )

        # 11/6/24 DH: Get number of Token Lengths used (ie from "-1-self-120")
        keySplit = epochLayerCnt.split("-")
        if len(keySplit) > 2:
          tokenLenStr = keySplit[gTokenIdx]
          try:
            tokenLen = int(tokenLenStr)
          except ValueError:
            # 20/6/24 DH: For when ".2" is appended to 'epochLayerCnt' (above)
            tokenLen = float(tokenLenStr)

          # Check if Token Length already added for different BertLayer
          if tokenLen not in tokenLens:
            tokenLens.append(tokenLen)

  return (recordsDict, tokenLens)

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

# Taken from: 'ai/huggingface/graph-logits.py'
def graphLogitsByLayer(recordsDict, layerNum, wantedTokenLen=None, lastGraph=False):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  fig = plt.figure()

  # 15/6/24 DH: https://matplotlib.org/stable/api/figure_api.html#figure-geometry
  # This just the canvas size (rather than scale the image like "Preview_Tools_Adjust size...")
  #(figWidth, figHeight) = fig.get_size_inches()
  #
  # Setting 'fig.dpi' adjusts image size
  #fig.set_dpi(fig.dpi * 0.7)

  # 12/6/24 DH: 'recordsDict' TRAINING keys are: "{epochNum}-{bert_cnt}-{bertLayerName}"
  epochList = [key.split("-")[gEpochIdx] for key in list(recordsDict.keys())]
  firstEpoch = epochList[0]
  lastEpoch = epochList[-1]
  
  #selectedLogits = 100
  # Used to print out all logits for a non-training run (which are dependent on Q+Context token length)
  selectedLogits = None

  # 11/6/24 DH:
  # --------------------------------------------------------------------------------------------------------------
  def plotAllLayersLines(allLayerLinesList):
    for lineDict in allLayerLinesList:
      tokenLen = lineDict['tokenLen']
      layer = lineDict['layer']
      compName = lineDict['compName']
      xVals = lineDict['xVals']
      yVals = lineDict['yVals']

      lwVal = 0.25
      if int(layer) == 12:
        # Now taking from multiple parts of Bert Layer 12
        if "self" in compName:
          lwVal = 1.0
        elif "out" in compName:
          lwVal = 1.5

      # Add line to 'plt.figure()' declared at top of 'graphLogitsByLayer(...)'
      # ...however we now want to have separate "all layers" graph for each token length (ie "Q+Context" output)
      
      # remove guide line in due course...
      #plotLine(lwVal=lwVal, labelStr=f"{tokenLen}-Layer{layer}")
      plt.plot(xVals[:selectedLogits], yVals[:selectedLogits], label=f"{tokenLen}-Layer{layer}-{compName}", linewidth=lwVal)

    # Used in graph title
    layerName = "all layers"
    return layerName
  # --------------------------------------------------------------------------------------------------------------

  # Used for DEV training run OF CUSTOM JSON with 20 epochs (ie 'run_qa.py' rather than 'test-qa-efficacy.py') ???
  epochsWanted = [0, 18, 19]

  # Used for "all layers" graph (currently only populated for NON-TRAINING run log lines)
  allLayerLinesList = []
  # 9/6/24 DH: Key is now "{epoch}-{layer num}-{token length}"
  for key in recordsDict:
    # 10/6/24 DH: For NON-TRAINING run 'keySplit[gTokenIdx]' will be token length, eg "-1-out-154"
    """ Key sections defined above
    gEpochIdx = 0
    gLayerIdx = 1
    gNameIdx  = 2
    gTokenIdx = 3
    """
    keySplit = key.split("-")
    epoch = keySplit[gEpochIdx]
    layer = keySplit[gLayerIdx]
    compName = keySplit[gNameIdx]

    # Hopefully, https://en.wikipedia.org/wiki/Short-circuit_evaluation ...spookily connected with AI...
    # epochsWanted = ['0', '19']
    #
    # "__contains__" means that '0' will match '0' and '10'
    #if any(map(key.__contains__, epochsWanted)) and int(key.split("-")[1]) == layerNum:

    # --------------------------------------------------------------------------------------------------
    def plotInplaceLine(lwVal, labelStr):
      xVals = range(len(recordsDict[key]))
      # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
      yVals = recordsDict[key].tolist()

      # Label with epoch or token length
      plt.plot(xVals[:selectedLogits], yVals[:selectedLogits], label=f"{labelStr}", linewidth=lwVal)
    # --------------------------------------------------------------------------------------------------

    try: # Training run only
      if any(map(int(epoch).__eq__, epochsWanted)) and int(layer) == layerNum:
        # 23/7/24 DH: Highlighting probable dev using custom JSON training with 20 epochs
        print()
        print(f"  *** MAYBE DEV: {epoch} from '{epochsWanted}' at layer: {layerNum} using 'plotInplaceLine()' ***")
        print()

        lwVal = (int(epoch)+4) / (int(lastEpoch))
        plotInplaceLine(lwVal, labelStr=epoch)
        # Used in graph title (+ cmd line feedback)
        lineType = "epoch"
        # Used in graph title
        layerName = f"layer {layerNum}"

    # This will catch ALL LINES from non-training run (due to no epoch)
    # EXAMPLE: "-1-out-154: [0.0373508557677269, 0.10611902177333832, ...]"
    except ValueError: # "invalid literal for int() with base 10: ''" (ie non-training run)
      if len(keySplit) > gTokenIdx:
        tokenLen = keySplit[gTokenIdx]
      else:
        print(f"NON-training run key '{key}' did not contain a token length")

      # Check for non-training run KEY DUP layer eg "1,2" (prob no longer needed with "-1-out-154", ie with token length appended)
      # ---------------------------------------------------------------------------------------------------------------------
      layerSplit = layer.split(",")
      if len(layerSplit) > 1:
        layer = layerSplit[0]
      # ---------------------------------------------------------------------------------------------------------------------

      # https://www.statology.org/matplotlib-line-thickness/ 
      #   "By default, the line width is 1.5 but you can adjust this to any value greater than 0."
      if int(layer) == layerNum:
        plotInplaceLine(lwVal=0.5, labelStr=tokenLen)
        # Used in graph title
        layerName = f"layer {layerNum}"

      # Debug of graph line shapes by adding all lines from same Token Length together
      if int(layerNum) == -1:
        # 20/6/24 DH: Need to handle multiple token len keys since SQuAD contains multiple 127 entries (+ others prob)
        wanted = False
        try:
          if int(tokenLen) == wantedTokenLen:
            wanted = True
            tokenLenNum = int(tokenLen)
        except ValueError: # eg "invalid literal for int() with base 10: '127.2'"
          if float(tokenLen) == wantedTokenLen:
            wanted = True
            tokenLenNum = float(tokenLen)

        if wanted:
          # Added for unpacking in 'plotAllLayersLines(...)'
          lineDict = {'tokenLen': tokenLenNum, 'layer': layer, 'compName': compName,'xVals': range(len(recordsDict[key])), 'yVals': recordsDict[key].tolist()}
          allLayerLinesList.append(lineDict)

      # Used in graph title (+ cmd line feedback)
      lineType = "token length"
    # END: ------ "except ValueError" ------
  # END: ------ "for key in recordsDict" ------

  # 11/6/24 DH: Now plotting "all layers" graph by Token Length (with lines collected in "for key in recordsDict")
  #             (currently only populated for NON-TRAINING run log lines, so SHOULD BE OK for TRAINING run log lines)
  layerName = plotAllLayersLines(allLayerLinesList)

  # 12/5/24 DH: Providing more feedback to output stage
  # 10/6/24 DH: Change title based on epoch for training OR token length for non-training
  titleStr = f"Logits from Bert Model, {layerName}, node 287 by {lineType}"

  plt.title(titleStr)
  plt.xlabel("Logit/Token ID")
  plt.ylabel("Logit value")

  print(f"\"{titleStr}\"")
  
  plt.legend(loc="best")

  #legendStr = f"Start logits: Solid line\nEnd logits:   Dotted line"
  #plt.figtext(0.15, 0.2, legendStr)
  
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  # 'tokenLen' at this point (ie after "for key in recordsDict" will ALWAYS CONTAIN THE LAST RUN TOKEN LEN)
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
  if lastGraph: # have mechanism to change saved image size (for later graphviz'ing)
    figDpi = fig.dpi * 0.5
  else:
    figDpi = fig.dpi

  # 7/7/24 DH: Tracking down error when running from any dir (ie '~/ai/huggingface' rather than '~/ai/t5')
  # 9/7/24 DH: Path refactor work when 'gv-graphs' is CWD subdir
  #allLayersImgName = f"{gOutputDir}/gv-graphs/all_layers-287-{wantedTokenLen}.png"

  allLayersImgName = f"{gCWD}/gv-graphs/all_layers-287-{wantedTokenLen}.png"
  print(f"   SAVING: {allLayersImgName}")
  print()
  plt.savefig(allLayersImgName, dpi=figDpi)

  if gShowFlag:
    #plt.draw()
    plt.show(block=lastGraph)

# -------------------------------------------------END: GRAPHING ---------------------------------------------

if __name__ == "__main__":
  # 'gTrainer_log = "weights/node287-logits.log"' centric
  (recordsDict, tokenLens) = collectLogits()

  displayLogits(recordsDict)

  # Debug of graph line shapes by adding all lines together
  tokenLensNum = len(tokenLens)
  cnt = 0
  lastGraph = False

  for tokenLen in tokenLens:
    cnt += 1
    if cnt == tokenLensNum:
      lastGraph = True

    # 16/6/24 DH:
    if len(sys.argv) > 2 and "show" in sys.argv[2]:
      gShowFlag = True

    graphLogitsByLayer(recordsDict, layerNum=-1, wantedTokenLen=tokenLen, lastGraph=lastGraph)
  
  # ...amusingly now we only want the combined lines graph...
  #graphLogitsByLayer(recordsDict, layerNum=1)
  #graphLogitsByLayer(recordsDict, layerNum=12, lastGraph=True)

  if not gShowFlag:
    print()
    print("NOT SHOWING images (please add 'show' to cmd line args if images wanted)")
    print()

  

