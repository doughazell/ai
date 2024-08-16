# 15/8/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys, os, random, time
from ast import literal_eval

print("Importing 'datasets'...")
import datasets
from datasets import load_dataset

import db_utils

gDBDIR            = "/Users/doug/src/ai/bert"
gDBNAME           = "stack_trace.db"
gTABLENAMEindices = "sample_indices"

def getSQUADindex(dataset, idx, numSamples):
  data = dataset[idx]

  question = data['question']
  context  = data['context']
  answer   = data['answers']
  expAnswer      = answer['text'][0]
  expAnswerStart = answer['answer_start'][0]

  print(f"INDEX: {idx} (of {numSamples})")
  print(f"QUESTION: {question}")
  print(f"CONTEXT: {context}")
  print(f"EXPECTED ANSWER: {expAnswer}")
  print(f"EXPECTED ANSWER START IDX: {expAnswerStart}")

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
def checkIdsInSet(recordDictList, startIdx, endIdx):
  print()
  print(f"Start last index: {startIdx}, model_efficacy_id: {recordDictList[startIdx]['model_efficacy_id']}")
  print(f"End last index: {endIdx}, model_efficacy_id: {recordDictList[endIdx]['model_efficacy_id']}")

  prevIdx = None
  for idx in range(endIdx + 1 - startIdx):
    listIdx = startIdx + idx

    if prevIdx != None:
      # https://docs.python.org/3/tutorial/datastructures.html#sets
      #   "a & b"  # letters in both a and b
      idsInBothRecords = recordDictList[prevIdx]['seq_idsSet'] & recordDictList[listIdx]['seq_idsSet']
      print()
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
    # END: --- "if prevIdx != None" ---

    else:
      prevIdx = listIdx
  # END: --- "for idx in range(endIdx + 1 - startIdx)" ---
  print()

gRecordDictList = []
# 16/8/24 DH: Passed to 'db_utils.iterateRecords(...)' and called for every record in 'stack_trace.db::sample_indices'
def checkDupsByModelIDHandler(record, recordNum):
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
    checkIdsInSet(gRecordDictList, startLastSetIdx, endLastSetIdx)

  # END: --- "if len(gRecordDictList) == recordNum" ---

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

# Connect Python/Huggingface with Ionic/Angular: https://github.com/DavidPineda/zmq-socket-angular
# https://stackoverflow.com/questions/53036381/how-to-deploy-ionic-4-app-to-github-pages
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
    print(f"RAW DATA FROM '{gDBNAME}::{gTABLENAMEindices}'")
    print( "----------------------------------------------")
    db_utils.iterateRecords(statsDB, gTABLENAMEindices, handlerFunc=checkDuplicatesColHandler)
    print("NOW CHECKING FOR DUPLICATE 'seq_ids'")
    print("------------------------------------")
    db_utils.iterateRecords(statsDB, gTABLENAMEindices, handlerFunc=checkDupsByModelIDHandler)

  datasetIdx = getIndexFromUser()
  while datasetIdx is False:
    datasetIdx = getIndexFromUser()

  dataset = raw_datasets["train"]
  getSQUADindex(dataset, datasetIdx, numSamples)

  
