# 23/3/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

# --------------------------- FUNCTIONS -------------------------
"""
niceWorkGoodjob()

Q&A Logits
----------
getIDsAndLogits(batchIdx, input_ids, start_logits, end_logits, startDelta)
logLogits(input_ids, start_logits, end_logits, start_loss=None, end_loss=None)

JSON Arrow data
---------------
stripListLayer(examples)

Optimizer Param Groups
----------------------
getBiasMapping()
printParamSummary(i, param, step_size, lr, at_start=True)
printGroupParams(i, group)
print_decay_parameter_names(opt_model, decay_parameters)

Storing layer weighting for graphing
------------------------------------
graphWeights(percentChgDictList, saveVals=True)
getWeightStats(weightsListIdx)
checkWeightsForAllSets()
logWeightings(weight_tensor)


"""
# ---------------------------------------------------------------

import sys
import matplotlib.pyplot as plt
from checkpointing import *
import checkpointing_config
# 24/5/24 DH: Needed to access 'Trainer.should_save' for end full weights saving
from transformers import Trainer

# 30/3/24 DH: ;)
def niceWorkGoodjob():
  print()
  print(f"  niceWorkGoodjob")
  print()

# 30/3/24 DH:
gInputIdsWritten = False
gEpochNum = 0
# 19/7/24 DH:
gInput_idsEntry = None

# ---------------------------------------------------------------
# Q&A Logits
# ---------------------------------------------------------------

def getIDsAndLogits(batchIdx, input_ids, start_logits, end_logits, extraDelta):
  
  try:
    inputIdsEnd = (input_ids[batchIdx] == 0).nonzero()[0]
  except IndexError:
    inputIdsEnd = input_ids[batchIdx].shape[0]

  idxEnd = inputIdsEnd + extraDelta

  startLogitsList = (start_logits[batchIdx][0:idxEnd]).tolist()
  startLogitsList = [round(value,2) for value in startLogitsList]

  endLogitsList = (end_logits[batchIdx][0:idxEnd]).tolist()
  endLogitsList = [round(value,2) for value in endLogitsList]

  return (inputIdsEnd, startLogitsList, endLogitsList)

# 19/7/24 DH: Now searching for randomized first entry of first batch
def findSampleInBatch(input_ids):
  unfoundIdx = -1

  # Previously assigned to first entry of first batch
  global gInput_idsEntry

  batchNum = input_ids.shape[0]
  for num in range(batchNum):
    # Firstly need to remove "zero padding" (like 'getIDsAndLogits(...)')
    try:
      inputIdEnd = (input_ids[num] == 0).nonzero()[0]
      inputId = input_ids[num][0:inputIdEnd].tolist()
      if inputId == gInput_idsEntry:
        return num
    except IndexError:
      return unfoundIdx

  return unfoundIdx

#################################################################################################################
# API CALLED FROM '*ForQuestionAnswering.forward()':
#   import huggin_utils
#   huggin_utils.logLogits(...) 
#
# EXAMPLE USAGE: https://github.com/doughazell/ai/commit/9b9d90037f38a1715fc6ead00c5363af9aff1608
#
# 30/3/24 DH: The same function can be used for a training run + model run
# 8/6/24 DH: Now defunct after overwriting Transformer/Torch with DeepPavlov reinstall
#            It was used to create 'seq2seq_qa_INtrainer.log' data for '$ python graph-logits.py'
#            (now only https://github.com/doughazell/ai/commits/main/t5/graphs/seq2seq_qa_INtrainer.log remains)
#
#            Superceded by 'logWeightings()' (and an artifact of HashTagger echo)
#################################################################################################################
def logLogits(tokenizer, input_ids, start_logits, end_logits, start_loss=None, end_loss=None):
  # 14/5/24 DH: PREV: Pass 'epochNum' from 'forward()' (so correlated with checkpoint number rather than just a delta)
  # 19/7/24 DH: NOW: 'epochNum = Trainer.stateAPI.global_step'
  global gEpochNum
  global gInputIdsWritten
  global gInput_idsEntry

  # Append logits after the token logits (to measure the shift in the Q+Context field)
  extraDelta = 2

  print()
  print( "  Logging logits")
  print( "  --------------")
  print()
  
  """ 19/7/24 DH: Maybe no longer necessary
  # Model run only gets called once (for non-training run ???)
  # -------------------------------
  if start_loss == None:
    import logging
    sigLogger = logging.getLogger("trainer_signaller")

    sigLogger.info(f"input_ids[0]: {input_ids[0][0:inputIdsEnd]}")

    sigLogger.info(f"startLogitsList (+{extraDelta}): {startLogitsList}")
    sigLogger.info(f"endLogitsList (+{extraDelta}): {endLogitsList}")

  # -------------------------------
  """
  if start_loss == None:
    print()
    print("REMOVED CODE THOUGHT UNNECESSARY in 'huggin_utils::logLogits(...)'")
    print()

  # Access custom additional API assigned in 'Trainer.compute_loss(...)' SO ONLY AVAILABLE IN TRAINING
  from transformers import Trainer
  epochNum = Trainer.stateAPI.global_step
  
  # 19/7/24 DH: Use first entry of first batch + search for it in later epochs
  # 12/8/24 DH: This works for Custom JSON (with 10 samples) but not for SQUAD (with 88524 samples)
  if Trainer.stateAPI.epoch == 0:
    idx = 0
    (inputIdsEnd, startLogitsList, endLogitsList) = getIDsAndLogits(idx, input_ids, start_logits, end_logits, extraDelta)
    gInput_idsEntry = input_ids[idx][0:inputIdsEnd].tolist()
  else:
    idx = findSampleInBatch(input_ids)
    
  if idx != -1:
  
    gEpochNum += 1
    
    # 28/3/24 DH: Use Python logging rather than 'transformers.utils.logging'
    import logging
    sigLogger = logging.getLogger("trainer_signaller")

    (inputIdsEnd, startLogitsList, endLogitsList) = getIDsAndLogits(idx, input_ids, start_logits, end_logits, extraDelta)

    # Need to add in 'input_ids[batchIdx]' on first time using global 'inputIdsWritten'
    # 30/3/24 DH: However it looks like there is randomisation of the dataset between epochs...!
    if not gInputIdsWritten:
      #gInputIdsWritten = True
      sigLogger.info(f"input_ids[{idx}]: {gInput_idsEntry}")

      # 18/7/24 DH: Detokenize 'input_ids' (like 'qa_lime.py::getTokStrings(...)')
      sigLogger.info(f"  {tokenizer.decode(gInput_idsEntry)}")

    # DEDUG OF REMAINING (384-TOKEN LOGITS)
    # -------------------------------------
    """
    endDelta = 10
    endIdx = inputIdsEnd + endDelta

    postStartList = (start_logits[batchIdx][inputIdsEnd:endIdx]).tolist()
    postStartList = [round(value,2) for value in postStartList]
    postStartEndList = (start_logits[batchIdx][-endDelta:]).tolist()
    postStartEndList = [round(value,2) for value in postStartEndList]
    print()
    print(f"  POST start ({inputIdsEnd}-{endIdx}): {postStartList}")
    print(f"             (-{endDelta}): {postStartEndList}")

    postEndList = (end_logits[batchIdx][inputIdsEnd:endIdx]).tolist()
    postEndList = [round(value,2) for value in postEndList]
    postEndEndList = (end_logits[batchIdx][-endDelta:]).tolist()
    postEndEndList = [round(value,2) for value in postEndEndList]
    print(f"  POST end ({inputIdsEnd}-{endIdx}): {postEndList}")
    print(f"            (-{endDelta}): {postEndEndList}")
    print()
    """
    
    sigLogger.info(f"{epochNum}) startLogitsList (+{extraDelta}): {startLogitsList}")
    sigLogger.info(f"  {epochNum}) start_loss: {start_loss}")

    sigLogger.info(f"{epochNum}) endLogitsList (+{extraDelta}): {endLogitsList}")
    sigLogger.info(f"  {epochNum}) end_loss: {end_loss}")
  # END: --- "if idx != -1" ---

# ---------------------------------------------------------------
# JSON Arrow data
# ---------------------------------------------------------------
    
# https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering#fine-tuning-bert-on-squad10
# "You might need to tweak the data processing inside the script if your data is structured differently."
def stripListLayer(examples):
  
  print("-----------------------------------")
  print(f"  examples: {examples.__class__}")
  print(f"  Stripping a list from:")
  print( "  ======================")
  
  # https://docs.python.org/3/library/collections.abc.html
  # examples.keys() => <class 'collections.abc.KeysView'>
  for key in examples.keys():
    print(f"  {key}: {examples[key][0].__class__}")

    if isinstance(examples[key][0], list):
      examples[key] = examples[key][0]
    
    print(f"    => {key}: {examples[key][0].__class__}")
    print()

  print("-----------------------------------")

# ---------------------------------------------------------------
# Optimizer Param Groups
# ---------------------------------------------------------------

# 25/4/24 DH: Create global for population in 'AdamW.__init__()'
gBiasMapping = []

# 25/4/24 DH: Mapping comes from 'Trainer.create_optimizer()' so written out to file for dynamic data passing
def getBiasMapping():
  global gBiasMapping
  try:
    gp1_filename = "gp1-params.txt"
    with open(gp1_filename) as fp:
      gBiasMapping = [line.strip() for line in fp.readlines() if line.strip()]
  except IOError as e:
    print()
    print(e)
    print()

def printParamSummary(i, param, step_size, lr, at_start=True):
  if at_start:
    if i == 0:
      print(f"    step_size: {step_size}, lr: {lr}")

    if i < 5:
      # 23/4/24 DH: Debug of 'AdamW.step()' 'param_groups'
      # 25/4/24 DH: Added global lookup table before '_single_tensor_adamw()' of layer names for Group 1 elem 'param_groups'
      
      if len(param.shape) == 1:
        try:
          print(f"    {i}) {param.shape}, '{gBiasMapping[i]}'")
        except IndexError:
          print(f"    {i}) {param.shape}, 'NO BIAS MAPPING'")

        print(f"      param[:5] => {param[:5]}")
      else:
        print(f"    {i}) {param.shape}")
        print(f"      param[:5] => {param[:5]}")

  else: # at_end
    print( "    ...")
    print(f"    {i}) {param.shape}")
    # Closure line from AdamW.step()
    print( "  --------------------------------------")
    print()

# 25/4/24 DH:
def printGroupParams(i, group):
  print( "  --------------------------------------")
  segment = ""
  if i == 0:
    segment = "ie 'weight'"
  if i == 1:
    segment = "ie Feedforward 'LayerNorm + bias'"
  print(f"  AdamW.step(): [GROUP: {i}, {segment}] lr: {group['lr']}, betas: {group['betas']}, eps: {group['eps']}, initial_lr: {group['initial_lr']}")
  print( "  ------------")

# 26/4/24 DH:
def print_decay_parameter_names(opt_model, all_layernorm_layers, decay_parameters):

  # 25/4/24 DH: Adapted from 'Trainer.create_optimizer()' BUT USING 'n' versus 'p'
  all_parameters = [n for n, p in opt_model.named_parameters()]
  bias_parameters = [name for name in all_layernorm_layers if "weight" not in name]

  non_decay_parameters = [n for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)]
  extraNonDecayParams = [n for n in non_decay_parameters if (n not in bias_parameters)]

  print()

  # 26/4/24 DH: All params
  all_filename = "all-params.txt"
  gp_name = "'all'"
  print(f"Saving {gp_name:<15} param names to '{all_filename}'")
  with open(all_filename, 'w') as sys.stdout:
    for name in all_parameters:
      print(name)
  # ...reset 'sys.stdout'
  sys.stdout = sys.__stdout__

  # 26/4/24 DH: Group 0 params
  gp0_filename = "gp0-params.txt"
  gp_name = "'Group 0'"
  print(f"Saving {gp_name:<15} param names to '{gp0_filename}'")
  with open(gp0_filename, 'w') as sys.stdout:
    for name in decay_parameters:
      print(f"  {name}")
  # ...reset 'sys.stdout'
  sys.stdout = sys.__stdout__

  # 25/4/24 DH: Group 1 params
  gp1_filename = "gp1-params.txt"
  gp_name = "'Group 1'"
  print(f"Saving {gp_name:<15} param names to '{gp1_filename}'")
  with open(gp1_filename, 'w') as sys.stdout:
    for name in non_decay_parameters:
      print(name)
  # ...reset 'sys.stdout'
  sys.stdout = sys.__stdout__

  # 26/4/24 DH: Group 1 Extra params
  gp1_extra_filename = "gp1-extra-params.txt"
  gp_name = "'Group 1 Extra'"
  print(f"Saving {gp_name:<15} param names to '{gp1_extra_filename}'")
  with open(gp1_extra_filename, 'w') as sys.stdout:
    for name in extraNonDecayParams:
      print(f"  {name}")
  # ...reset 'sys.stdout'
  sys.stdout = sys.__stdout__  

# ---------------------------------------------------------------
# Storing layer weighting for graphing
# ---------------------------------------------------------------

# 12/5/24 DH: MUTABLE variables (don't need to be accessed with 'global' to prevent local scope overlay)
weightMapDict = {
  0: "Start",
  "Start": 767,
  1: "End",
  "End": 10,
}
# 13/5/24 DH: "transformers$ grep -r '\[\[\]\]' *"
weightValMatrix = [[],[]]

# 14/5/24 DH: IMMUTABLE variables (so need to be accessed via 'global')
# Start from elem 0 after first toggle
gValMatrixIdx = 1

# 21/5/24 DH: If still set to '-1' then first entry for full values graphing (for later comparison with last values graph)
gFirstEpoch = -1

# Taken from: 'ai/huggingface/graph-logits.py'
def graphWeights(percentChgDictList, saveVals=True, roundedDictList=None):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  # 14/5/24 DH: Access custom additional API
  from transformers import Trainer
  epochNum = Trainer.stateAPI.global_step

  # 12/5/24 DH: Providing more feedback to output stage
  # 21/5/24 DH: CURRENTLY the only time we set 'saveVals=False' is with full value graphs
  if saveVals:
    titleStr = f"Weight change by node from 'qa_outputs' layer for epoch {epochNum}"
    #plt.ylim(top=20, bottom=-15)
  else:
    titleStr = f"Full weight by node from 'qa_outputs' layer for epoch {epochNum}"

  plt.title(titleStr)
  plt.xlabel("Node number")
  plt.ylabel("Weight")
  
  print(f"  \"{titleStr}\" (USING: 'abs(prevWeight)')")

  listLen = len(percentChgDictList)
  idx = 0
  for iDict in percentChgDictList:
    xVals = percentChgDictList[idx].keys()
    yVals = [percentChgDictList[idx][key] for key in percentChgDictList[idx]]

    if saveVals: # NOTE: 'gWeightsFile'
      # Opened in 'checkpointing.py::createLoggers(training_args)' called from 'run_qa.py'
      checkpointing_config.gWeightsFile.write(f"{epochNum}-{weightMapDict[idx]}: {yVals}\n")

      # 29/7/24 DH: MAYBE ADD ROUNDED WEIGHTS EXTRA LINE HERE for each "epoch-lineType"
      #   Option 1: Add 2 extra lines per epoch to "weights.log"
      #   Option 2: Populate lines per epoch to "weights-rounded.log" [CHOSEN]
      # 29/7/24 DH:
      if roundedDictList:
        yRoundedVals = [roundedDictList[idx][key] for key in roundedDictList[idx]]
        checkpointing_config.gRoundedWeightsFile.write(f"{epochNum}-{weightMapDict[idx]}: {yRoundedVals}\n")

      if idx == 1: # space after end line (ie idx 1)
        checkpointing_config.gWeightsFile.write("\n")
        # 29/7/24 DH:
        if roundedDictList:
          checkpointing_config.gRoundedWeightsFile.write("\n")
    
    else: # NOTE: 'gFullWeightsFile'
      checkpointing_config.gFullWeightsFile.write(f"{epochNum}-{weightMapDict[idx]}: {yVals}\n")
      if idx == 1: # space after end line (ie idx 1)
        checkpointing_config.gFullWeightsFile.write("\n")
      
    lwVal = (idx + 1) / listLen

    plt.plot(xVals, yVals, label=f"{weightMapDict[idx]}", linewidth=lwVal)

    idx += 1
  
  plt.legend(loc="best")

  #legendStr = f"Start logits: Solid line\nEnd logits:   Dotted line"
  #plt.figtext(0.15, 0.2, legendStr)
  
  #plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  #plt.draw()

  # 24/5/24 DH: When called from '*QuestionAnswering.forward()' we just 'savefig()' (rather than 'show()')
  #             which means that we can 'close()' after saving (rather than waiting to display at end of training)
  #plt.show(block=False)

  # 21/5/24 DH: CURRENTLY the only time we set 'saveVals=False' is with full value graphs
  if saveVals:
    graphFilename = f"{epochNum}-percentDiffs"
  else:
    graphFilename = f"{epochNum}-fullValues"
  plt.savefig(f"{checkpointing_config.gGraphDir}/{graphFilename}.png")

  # 24/5/24 DH: "RuntimeWarning: More than 20 figures have been opened. Figures created through ... (`matplotlib.pyplot.figure`) are retained 
  #              until explicitly closed and may consume too much memory.  Consider using `matplotlib.pyplot.close()"
  plt.close()

# 14/5/24 DH: Convert full weight values to rounded weight diff
def getWeightStats(weightsListIdx):
  global gValMatrixIdx
  
  percentChgDict = {}
  # 29/7/24 DH:
  roundedWeightsDict = {}

  currLen = len(weightValMatrix[gValMatrixIdx][weightsListIdx])
  try:
    # NOTE: "1 - gValMatrixIdx"
    prevLen = len(weightValMatrix[1 - gValMatrixIdx][weightsListIdx])
  except IndexError:
    prevLen = "NONE"

  if prevLen != "NONE":
    for idx in range(currLen):
      currWeight = weightValMatrix[gValMatrixIdx][weightsListIdx][idx]
      prevWeight = weightValMatrix[1 - gValMatrixIdx][weightsListIdx][idx]
      #print(f"    {idx}: {currWeight} from {prevWeight}")

      # Need percent change from previous
      diff = currWeight - prevWeight
      # 22/5/24 DH: 'diff/prevWeight' needs 'abs(prevWeight)'
      percentChgFromPrev = round(diff / abs(prevWeight) * 100, 3)
      #mTxt = "Diff:"
      #print(f"{mTxt:>17} {diff}, Percent from prev: {percentChgFromPrev}%")

      percentChgDict[idx] = percentChgFromPrev
      # 29/7/24 DH:  Needed to round to 6dp (in order to record ANY CHANGE) from 2dp (acceptable for % chg values per epoch) 
      roundedWeightsDict[idx] = round(currWeight, 6)

    # END: --- "for idx in range(currLen)" ---
  
  return (percentChgDict, roundedWeightsDict)

# 21/5/24 DH:
def getFullvalsDictList(weightValMatrix):
  fullvalsDictList = []

  valMatrixLen = len(weightValMatrix)
  # 'qa_outputs' full values list for all 767 nodes
  for idx in range(valMatrixLen):
    #print(f"  'getFullvalsDictList()' - {weightMapDict[idx]}")
    fullvalsDict = {}

    fullvalsListLen = len(weightValMatrix[idx])
    for valIdx in range(fullvalsListLen):
      fullvalsDict[valIdx] = weightValMatrix[idx][valIdx]

    fullvalsDictList.append(fullvalsDict)

  return fullvalsDictList

# 14/5/24 DH: WRAPPER to convert full weight values to weight diff
def checkWeightsForAllSets():
  global gValMatrixIdx
  global gFirstEpoch
  percentChgDictList = []
  # 29/7/24 DH:
  roundedDictList = []

  idx = 0
  # Full values for 'qa_outputs' CURRENT values
  # ----------------------------------------
  # weightValMatrix[gValMatrixIdx][0]: "[0.0387488454580307, 0.030061528086662292, 0.018482686951756477, ..., 0.012040662579238415]"
  # weightValMatrix[gValMatrixIdx][1]: "[0.007029589265584946, -0.0615961030125618, -0.028790319338440895, ..., 0.026614349335432053]"
  for iList in weightValMatrix[gValMatrixIdx]:
    (weightStats, roundedWeights) = getWeightStats(idx)
    
    # Account for first entry when 'prevLen == "NONE"'
    if len(weightStats) > 0:
      percentChgDictList.append(weightStats)
      # 29/7/24 DH:
      roundedDictList.append(roundedWeights)

    idx += 1

  if len(percentChgDictList) > 0:
    # Rounded, percentage diff for 'qa_outputs' values FROM PREV values
    # --------------------------------------------------------------
    # percentChgDictList[0]: "{0: 0.012, 1: 0.011, 2: 0.025, ..., 767: 0.042}"
    # percentChgDictList[1]: "{0: 0.046, 1: -0.002, 2: -0.008, ..., 767: 0.017}"

    # 29/7/24 DH: (See 'graph-weights-history.py' detailed comment re graphing weight chg over training epoch)
    #             We also need to RECORD ROUNDED WEIGHT VALUE (as well as percentage diff) in 'getWeightStats(idx)' (in order to graph over training)
    graphWeights(percentChgDictList, roundedDictList=roundedDictList)
  
  # 21/5/24 DH: Graph final full weights (rather than just percentage diffs)

  # Access custom additional API
  from transformers import Trainer
  epochNum = Trainer.stateAPI.global_step
  maxEpochs = Trainer.stateAPI.max_steps

  print()
  print(f"Epoch: {epochNum} of {maxEpochs}")
  print()
  if gFirstEpoch < 0:
    print("  ...that'll do donkey, that'll do...")
    fullvalsDictList = getFullvalsDictList(weightValMatrix[gValMatrixIdx])
    graphWeights(fullvalsDictList, saveVals=False)

    # Only used for the first entry
    gFirstEpoch = epochNum

  # 24/5/24 DH: If training is stopped with'Ctrl-C' then need to access 'Trainer.should_save' flag (as well as last epoch)
  if epochNum == maxEpochs - 1 or Trainer.should_save:
    print("  Yup last one")
    fullvalsDictList = getFullvalsDictList(weightValMatrix[gValMatrixIdx])
    graphWeights(fullvalsDictList, saveVals=False)


############################################################################################
# API CALLED FROM '*ForQuestionAnswering.forward()':
#   import huggin_utils
#   huggin_utils.logWeightings(...) 
############################################################################################

def logWeightings(weight_tensor):
  global gValMatrixIdx
  # Toggle idx between [0,1]
  gValMatrixIdx += 1
  gValMatrixIdx = gValMatrixIdx % 2

  print()
  print( "  Logging weightings")
  print( "  ------------------")
  print()

  # For 'qa_outputs.weight' it is [2,768]
  weightsDim = list(weight_tensor.shape)

  for idx in range(weightsDim[0]):

    # Add weights with full precision (prior to comparing with next 'weight_tensor' in order to find diffs prior to rounding)

    # 13/5/24 DH: If previous weights present (as determined by list len) then they need removing prior to adding next set
    newWeights = weight_tensor[idx].tolist()
    newWeightsLen = len(newWeights)
    try:
      currWeightValMatrixLen = len(weightValMatrix[gValMatrixIdx][idx])
    except IndexError:
      currWeightValMatrixLen = 0

    # STORING weights (currently with full precision)
    # ---------------
    if newWeightsLen != currWeightValMatrixLen:
      weightValMatrix[gValMatrixIdx].append(newWeights)
    else:
      # Remove previous weights arrays for (start+end)
      weightValMatrix[gValMatrixIdx].pop()
      weightValMatrix[gValMatrixIdx].pop()
      weightValMatrix[gValMatrixIdx].append(newWeights)
    
  # END: --- "for idx in range(weightsDim[0])" ---

  # Debug
  """
  print("  ########## DEBUG POPULATED ELEMS ###########")
  # in both start + end weight arrays
  for idx in range(weightsDim[0]):
    valIdx = weightMapDict[weightMapDict[idx]]
    print(f"  {weightMapDict[idx]} idx {valIdx}: {weightValMatrix[gValMatrixIdx][idx][valIdx]}")
    iLast = weightsDim[1] - 1
  print("  ############################################")
  """

  # 21/5/24 DH: Also calls: graphWeights(percentChgDictList)
  checkWeightsForAllSets()

############################################################################################
# API CALLED FROM 'BertSelfAttention.forward()':
#   import huggin_utils
#   huggin_utils.logSelectedNodeLogits(...) 
############################################################################################

# 8/6/24 DH:
def logSelectedNodeLogits(nodeForeachLogit, bert_cnt, bertLayerName, embedTokens=384):
  # Copied from: 'graphWeights(percentChgDictList, saveVals=True)'
  """
  checkpointing_config.gFullWeightsFile.write(f"{epochNum}-{weightMapDict[idx]}: {yVals}\n")
  if idx == 1: # space after end line (ie idx 1)
    checkpointing_config.gFullWeightsFile.write("\n")
  """

  # Access custom additional API (need to prevent: "'NoneType' object has no attribute 'global_step'")
  from transformers import Trainer
  if Trainer.stateAPI:
    epochNum = Trainer.stateAPI.global_step
    
    checkpointing_config.gSelectedNodeFile.write(f"{epochNum}-{bert_cnt}-{bertLayerName}: {nodeForeachLogit}\n")
    checkpointing_config.gSelectedNodeFile.write("\n")

  else:
    # 9/6/24 DH: Save logits for non-training run
    #   'createLoggers(training_args, overwrite=True)' to open 'gSelectedNodeFilename' for non-training run
    # 12/9/24 DH: Chg 'gSelectedNodeFilename' to 'gSelectedNodeFile' (as distinct from the filename)
    checkpointing_config.gSelectedNodeFile.write(f"-{bert_cnt}-{bertLayerName}-{embedTokens}: {nodeForeachLogit}\n")
    checkpointing_config.gSelectedNodeFile.write("\n")
    print(f"    Saved '{bert_cnt}-{embedTokens}' to 'gSelectedNodeFile'")

  # 11/6/24 DH: ..."do you flush?" (like 'Trainer.training_step()::checkpointing_config.gLossFile.flush()')
  checkpointing_config.gSelectedNodeFile.flush()

# 13/9/24 DH: ASCII-art time...
"""                                       - [checkpointing_config.py] -
modeling_bert.py <-> huggin_utils.py <-> |                             | <- checkpointing.py
"""
def getTrackedNode():
  return checkpointing_config.gTrackedNode
