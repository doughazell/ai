# 15/8/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys, os, random, time
from ast import literal_eval

print("Importing 'AutoTokenizer'")
from transformers import AutoTokenizer

print("Importing 'datasets'...")
import datasets
from datasets import load_dataset

import db_utils

gDBDIR            = "/Users/doug/src/ai/bert"
gDBNAME           = "stack_trace.db"
gTABLENAMEindices = "sample_indices"

# Used to store a DictList of records from 'db_utils.iterateRecords(...)'
gRecordDictList = []

# ================================= SQUAD ===================================
def getSQUADindex(dataset, idx, numSamples, tokenizer):
  data = dataset[idx]

  question = data['question']
  context  = data['context']
  answer   = data['answers']
  expAnswer      = answer['text'][0]
  expAnswerStart = answer['answer_start'][0]

  encoding = tokenizer(question, context, return_tensors="pt")
  all_tokens = tokenizer.convert_ids_to_tokens( encoding["input_ids"][0].tolist() )
  tokenLen = len(all_tokens)

  print(f"INDEX: {idx} (of {numSamples})")
  print(f"QUESTION: {question}")
  print(f"CONTEXT: {context}")
  print(f"EXPECTED ANSWER: {expAnswer}")
  print(f"EXPECTED ANSWER START IDX: {expAnswerStart}")
  print(f"TOKEN LEN (Bert vocab): {tokenLen}")

def getIndexFromUser():
  print("What index do you want? (just press return for random index) ", end='')
  response = input()
  if len(response) == 0:
    # https://en.wikipedia.org/wiki/Division_(mathematics)#Of_integers
    # "Give the integer quotient as the answer, so 26 / 11 = 2. It is sometimes called 'integer division', and denoted by '//'."
    datasetIdx = int( (random.random() * numSamples * numSamples) // numSamples )
    print(f"Getting 'random' index: {datasetIdx}")
    print()
  else:
    try:
      datasetIdx = int(response)
      print()
    except ValueError:
      return False
  
  return datasetIdx

# =============================== CHECK DUPLICATE SEQ_IDS ===================================
# 15/8/24 DH: Passed to 'db_utils.iterateRecords(...)' and called for every record in 'stack_trace.db::sample_indices'
def checkDuplicatesColHandler(record, recordNum):
  # Specific to 'stack_trace.db::sample_indices' schema
  id                = record[0]
  model_efficacy_id = record[1]
  seq_num           = record[2]
  seq_idsStr        = record[3]
  print(f"{id}")
  print(f"  model_efficacy_id: {model_efficacy_id}")
  print(f"  seq_num: {seq_num}")
  print(f"  seq_ids: {seq_idsStr}")

  # https://docs.python.org/3/library/ast.html#ast.literal_eval
  seq_idsList = literal_eval(seq_idsStr)

  # https://docs.python.org/3/tutorial/datastructures.html#sets "eliminating duplicate entries."
  seq_idsSet = set(seq_idsList)

  seq_idsSetLen = len(seq_idsSet)
  if int(seq_num) != seq_idsSetLen:
    print(f"We have a winner: 'seq_idsSet' is {seq_idsSetLen}")
  
  print()

# 16/8/24 DH:
def checkAdjacentIdsInSet(recordDictList, startIdx, endIdx):
  print()
  print(f"Start last index: {startIdx}, model_efficacy_id: {recordDictList[startIdx]['model_efficacy_id']}")
  print(f"End last index: {endIdx}, model_efficacy_id: {recordDictList[endIdx]['model_efficacy_id']}")
  print()
  print("*** NOTE: This only checks ADJACENT RECORDS for duplicates ***")
  print()

  prevIdx = None
  for idx in range(endIdx + 1 - startIdx):
    listIdx = startIdx + idx

    if prevIdx != None:
      # https://docs.python.org/3/tutorial/datastructures.html#sets
      #   "a & b"  # letters in both a and b
      idsInBothRecords = recordDictList[prevIdx]['seq_idsSet'] & recordDictList[listIdx]['seq_idsSet']
      print(f"  Ids in record {listIdx} & {prevIdx}: '{idsInBothRecords}'")

      # Now double-check "a & b"
      # ------------------------
      print(f"  Double checking values between index {listIdx} and {prevIdx}:")
      uniqueIDs = True
      for prevID in recordDictList[prevIdx]['seq_idsSet']:
        for listID in recordDictList[listIdx]['seq_idsSet']:
          if listID == prevID:
            print(f"    ...we have a winner: prevID: {prevID}, listID: {listID}")
            uniqueIDs = False
      if uniqueIDs:
        print("    no repeats found")
      # ------------------------

      prevIdx = listIdx
      print()
    # END: --- "if prevIdx != None" ---

    else:
      prevIdx = listIdx
  # END: --- "for idx in range(endIdx + 1 - startIdx)" ---

# 17/8/24 DH:
def checkIdsInSet(recordDictList, startIdx, endIdx):
  print(f"SET: {recordDictList[startIdx]['model_efficacy_id']} - startIdx: {startIdx}, endIdx: {endIdx}")
  print()
  for idx1 in range(endIdx + 1 - startIdx):
    listIdx1 = startIdx + idx1

    for idx2 in range(endIdx + 1 - startIdx):
      listIdx2 = startIdx + idx2
      if idx1 != idx2:
        # https://docs.python.org/3/tutorial/datastructures.html#sets
        #   "a & b"  # letters in both a and b
        idsInBothRecordsSet = recordDictList[listIdx1]['seq_idsSet'] & recordDictList[listIdx2]['seq_idsSet']
        if len(idsInBothRecordsSet) == 0:
          idsInBothRecordsStr = ""
        else:
          idsInBothRecordsStr =  list(idsInBothRecordsSet)
        print(f"  Ids in record {listIdx1} & {listIdx2}: '{idsInBothRecordsStr}'")
      # END: --- "if idx1 != idx2" ---
    # END: --- "for idx2 in range(endIdx + 1 - startIdx)" ---
  # END: --- "for idx1 in range(endIdx + 1 - startIdx)" ---
  print()

# 17/8/24 DH:
def populateDictListHandler(record, recordNum):
  global gRecordDictList
  recordDict = {}

  # Specific to 'stack_trace.db::sample_indices' schema
  id                = record[0]
  model_efficacy_id = record[1]
  seq_num           = record[2]
  seq_idsStr        = record[3]
  
  # https://docs.python.org/3/library/ast.html#ast.literal_eval
  seq_idsList = literal_eval(seq_idsStr)

  # https://docs.python.org/3/tutorial/datastructures.html#sets "eliminating duplicate entries."
  seq_idsSet = set(seq_idsList)

  recordDict['id'] = int(id)
  recordDict['model_efficacy_id'] = int(model_efficacy_id)
  recordDict['seq_num'] = int(seq_num)
  recordDict['seq_idsList'] = seq_idsList
  recordDict['seq_idsSet'] = seq_idsSet

  gRecordDictList.append(recordDict)

  # We have a list of ALL RECORDS
  if len(gRecordDictList) == recordNum:
    print()

# 16/8/24 DH: Passed to 'db_utils.iterateRecords(...)' and called for every record in 'stack_trace.db::sample_indices'
def checkDupsByModelIDHandler(record, recordNum):
  populateDictListHandler(record, recordNum)

  # We have a list of ALL RECORDS so now cross correlate 'seq_ids' from same 'model_efficacy_id'
  if len(gRecordDictList) == recordNum:
    startIdxList = [0, 0]
    startIdxTop = 0
    
    currModelId = 0
    dlIdx = 0
    for dict in gRecordDictList:

      if dict['model_efficacy_id'] != currModelId:
        currModelId = dict['model_efficacy_id']
        
        if dlIdx != 0: # ie we have moved onto next set of 'model_efficacy_id' BUT NOT FIRST ENTRY
          startLastSetIdx = startIdxList[startIdxTop]
          endLastSetIdx = dlIdx - 1
          #checkAdjacentIdsInSet(gRecordDictList, startLastSetIdx, endLastSetIdx)
          checkIdsInSet(gRecordDictList, startLastSetIdx, endLastSetIdx)

          # Toggle top to prevent overwriting the last start index (which initially is '0')
          print(f"TOGGLING TO SET: {currModelId} at index {dlIdx}")
          startIdxTop = (startIdxTop + 1) % len(startIdxList)
          startIdxList[startIdxTop] = dlIdx
      # END --- "if dict['model_efficacy_id'] != currModelId" ---

      dlIdx += 1
    # END: --- "for dict in gRecordDictList" ---

    # Now test last 'model_efficacy_id' in set
    startLastSetIdx = startIdxList[startIdxTop]
    endLastSetIdx = dlIdx - 1
    #checkAdjacentIdsInSet(gRecordDictList, startLastSetIdx, endLastSetIdx)
    checkIdsInSet(gRecordDictList, startLastSetIdx, endLastSetIdx)

  # END: --- "if len(gRecordDictList) == recordNum" ---

# 16/8/24 DH:
def checkForSeqIDs(dupSeqIDs):
  # Not necessary for just reading (but would create a local variable when writing without "global")
  global gRecordDictList 

  for seq_id in dupSeqIDs:
    print(f"CHECKING in 'gRecordDictList': {seq_id}")
    for dict in gRecordDictList:
      if seq_id in dict['seq_idsSet']:
        print(f"  FOUND {seq_id} in ID: {dict['id']}")

  print()

# ======================================== MAIN =========================================

# Connect Python/Huggingface with Ionic/Angular: https://github.com/DavidPineda/zmq-socket-angular
#                                                https://zeromq.org/languages/nodejs/
#                                                https://zeromq.org/languages/python/
#                                                https://groovetechnology.com/blog/how-to-use-python-in-node-js/
#
# Put Ionic on GitHub: https://stackoverflow.com/questions/53036381/how-to-deploy-ionic-4-app-to-github-pages
if __name__ == "__main__":
  print()
  print("...to infinity and beyond...may they never meet")
  print()

  # FROM: 'test-qa-efficacy.py::main()'
  datasetName = 'squad'
  raw_datasets = load_dataset(datasetName)
  numSamples           = raw_datasets['train'].num_rows
  numValidationSamples = raw_datasets['validation'].num_rows
  
  print(f"{numSamples} samples + {numValidationSamples} validation samples = {numSamples + numValidationSamples} total in '{datasetName}'")
  print()

  # Re-randomize seeded generator
  random.seed(time.time())

  print("Do you want to check for duplicate 'seq_ids'? [Y/n] ", end='')
  response = input()
  if response.lower() == "y" or len(response) == 0:
    statsDB = db_utils.getDBConnection(f"{gDBDIR}/{gDBNAME}")
    print()

    """ DEBUG
    print(f"RAW DATA FROM '{gDBNAME}::{gTABLENAMEindices}'")
    print( "----------------------------------------------")
    db_utils.iterateRecords(statsDB, gTABLENAMEindices, handlerFunc=checkDuplicatesColHandler)
    """
    
    print("CHECKING FOR DUPLICATE 'seq_ids'")
    print("--------------------------------")
    db_utils.iterateRecords(statsDB, gTABLENAMEindices, handlerFunc=checkDupsByModelIDHandler)
    #db_utils.iterateRecords(statsDB, gTABLENAMEindices, handlerFunc=populateDictListHandler)

    """
    # Find [23581, 21533, 46079] in mid-sequence for 'model_efficacy_id' '8' (and HENCE NOT CORRECT) in order to FIND WHICH ONE IS CORRECT
    RESULT: Only found as final sequence in ID's 7,8 so UNABLE TO TELL WHICH ONE IS CORRECT FROM THAT
            'qca.log' shows '2-65' (21533) is correct for 'test-qa-efficacy.py::runRandSamples(...)' HACK:

                for idx in range(iterations):
                  ...
                  indicesCheck = [23581, 21533, 46079]
                  datasetsIdx = indicesCheck[idx]
    #dupSeqIDs = [23581, 21533, 46079]
    #checkForSeqIDs(dupSeqIDs)
    """

  datasetIdx = getIndexFromUser()
  while datasetIdx is False:
    datasetIdx = getIndexFromUser()

  dataset = raw_datasets["train"]
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  getSQUADindex(dataset, datasetIdx, numSamples, tokenizer)

  