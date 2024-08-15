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
def checkDuplicatesColHandler(record):
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

  # https://en.wikipedia.org/wiki/Division_(mathematics)#Of_integers
  # "Give the integer quotient as the answer, so 26 / 11 = 2. It is sometimes called 'integer division', and denoted by '//'."
  datasetIdx = int( (random.random() * numSamples * numSamples) // numSamples )

  dataset = raw_datasets["train"]
  getSQUADindex(dataset, datasetIdx, numSamples)

  statsDB = db_utils.getDBConnection(f"{gDBDIR}/{gDBNAME}")
  db_utils.iterateRecords(statsDB, gTABLENAMEindices, handlerFunc=checkDuplicatesColHandler)
  
