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

def logLogits(input_ids, start_logits, end_logits, start_loss, end_loss):
  global epochNum
  global inputIdsWritten

  # 28/3/24 DH: Use Python logging rather than 'transformers.utils.logging'
  #             (Take first entry of overflow batch from Dataset #10 with batch size #8)
  if start_logits.shape[0] == 2:
  
    epochNum += 1
    
    import logging
    sigLogger = logging.getLogger("trainer_signaller")

    inputIdsEnd = (input_ids[0] == 0).nonzero()[0]
    # Need to add in 'input_ids[0]' on first time using global 'inputIdsWritten'
    # 30/3/24 DH: However it looks like there is randomisation of the dataset between epochs...!
    if not inputIdsWritten:
      inputIdsWritten = True
      sigLogger.info(f"input_ids[0]: {input_ids[0][0:inputIdsEnd]}")

    endDelta = 10
    endIdx = inputIdsEnd + endDelta

    postStartList = (start_logits[0][inputIdsEnd:endIdx]).tolist()
    postStartList = [round(value,2) for value in postStartList]
    postStartEndList = (start_logits[0][-endDelta:]).tolist()
    postStartEndList = [round(value,2) for value in postStartEndList]
    print()
    print(f"  POST start ({inputIdsEnd}-{endIdx}): {postStartList}")
    print(f"             (-{endDelta}): {postStartEndList}")

    postEndList = (end_logits[0][inputIdsEnd:endIdx]).tolist()
    postEndList = [round(value,2) for value in postEndList]
    postEndEndList = (end_logits[0][-endDelta:]).tolist()
    postEndEndList = [round(value,2) for value in postEndEndList]
    print(f"  POST end ({inputIdsEnd}-{endIdx}): {postEndList}")
    print(f"            (-{endDelta}): {postEndEndList}")
    print()

    startDelta = 2
    idxEnd = inputIdsEnd + startDelta

    startLogitsList = (start_logits[0][0:idxEnd]).tolist()
    startLogitsList = [round(value,2) for value in startLogitsList]
    sigLogger.info(f"{epochNum}) startLogitsList (+{startDelta}): {startLogitsList}")
    sigLogger.info(f"  {epochNum}) start_loss: {start_loss}")

    endLogitsList = (end_logits[0][0:idxEnd]).tolist()
    endLogitsList = [round(value,2) for value in endLogitsList]
    sigLogger.info(f"{epochNum}) endLogitsList (+{startDelta}): {endLogitsList}")
    sigLogger.info(f"  {epochNum}) end_loss: {end_loss}")


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