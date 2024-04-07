# 23/3/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

# 30/3/24 DH:
def niceWorkGoodjob():
  print()
  print(f"  niceWorkGoodjob")
  print()

# 30/3/24 DH:
inputIdsWritten = False
epochNum = 0

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