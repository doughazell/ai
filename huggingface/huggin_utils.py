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

"""
# ---------------------------------------------------------------

import sys

# 30/3/24 DH:
def niceWorkGoodjob():
  print()
  print(f"  niceWorkGoodjob")
  print()

# 30/3/24 DH:
inputIdsWritten = False
epochNum = 0

# ---------------------------------------------------------------
# Q&A Logits
# ---------------------------------------------------------------

def getIDsAndLogits(batchIdx, input_ids, start_logits, end_logits, startDelta):
  
  try:
    inputIdsEnd = (input_ids[batchIdx] == 0).nonzero()[0]
  except IndexError:
    inputIdsEnd = input_ids[batchIdx].shape[0]

  idxEnd = inputIdsEnd + startDelta

  startLogitsList = (start_logits[batchIdx][0:idxEnd]).tolist()
  startLogitsList = [round(value,2) for value in startLogitsList]

  endLogitsList = (end_logits[batchIdx][0:idxEnd]).tolist()
  endLogitsList = [round(value,2) for value in endLogitsList]

  return (inputIdsEnd, startLogitsList, endLogitsList)

############################################################################################
# API CALLED FROM '*ForQuestionAnswering.forward()':
#   import huggin_utils
#   huggin_utils.logLogits(...) 
#
# 30/3/24 DH: The same function can be used for a training run + model run
############################################################################################
def logLogits(input_ids, start_logits, end_logits, start_loss=None, end_loss=None):
  global epochNum
  global inputIdsWritten

  # Append logits after the token logits (to measure the shift in the Q+Context field)
  extraDelta = 2

  # Take the first item in the batch (which currently is the 2 overflow batch)
  batchIdx = 0
  
  (inputIdsEnd, startLogitsList, endLogitsList) = getIDsAndLogits(batchIdx, input_ids, start_logits, end_logits, extraDelta)

  # Model run only gets called once
  # -------------------------------
  if start_loss == None:
    import logging
    sigLogger = logging.getLogger("trainer_signaller")

    sigLogger.info(f"input_ids[{batchIdx}]: {input_ids[batchIdx][0:inputIdsEnd]}")
    sigLogger.info(f"startLogitsList (+{extraDelta}): {startLogitsList}")
    sigLogger.info(f"endLogitsList (+{extraDelta}): {endLogitsList}")

  # -------------------------------

  # 28/3/24 DH: Use Python logging rather than 'transformers.utils.logging'
  #
  # (DURING TRAINING: Take first entry of overflow batch from Dataset #10 with batch size #8
  #  THEREFORE the model has already been trained by batch size (8) reps over un-trained model that only "knows language")
  if start_logits.shape[0] == 2:
  
    epochNum += 1
    
    import logging
    sigLogger = logging.getLogger("trainer_signaller")

    # Need to add in 'input_ids[batchIdx]' on first time using global 'inputIdsWritten'
    # 30/3/24 DH: However it looks like there is randomisation of the dataset between epochs...!
    if not inputIdsWritten:
      inputIdsWritten = True
      sigLogger.info(f"input_ids[{batchIdx}]: {input_ids[batchIdx][0:inputIdsEnd]}")

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
  # END: ------ "if start_logits.shape[0] == 2" ------

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

