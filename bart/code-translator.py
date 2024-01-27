from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import numpy as np

# 25/1/24 DH:
from code_translator_trainer import trainSeqClassModel, trainSeq2SeqLM

# 14/1/24 DH:
print("-----------------------------------------------------------------------------------------")
print("Using Transformers BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')")
print("-----------------------------------------------------------------------------------------")

# --------------------------------------------------------------------------------------------------------------
# 22/1/24 DH:
def getTensorFromIntStrList(intStrList) -> torch.tensor:
  # 'input_ids' needs to be 2-D array (prob for obfuscation reasons...)
  int2DList = [[int(item) for item in intStrList]]
  input_ids = torch.tensor(int2DList)
  return input_ids

# 15/1/24 DH: TODO: This would probably be better with 'csv' (like 'num_beams-parser.py')
def paragraphDecode(tokenizer, filename) -> torch.tensor :
  intStrList = []
  with open(filename) as source :
    for line in source.readlines():
      # Remove newline character from each int printout line
      lineStrip = line.rstrip()

      for item in lineStrip.split(","):
        # Remove whitespace
        itemStrip = item.strip()
        # Guard against empty string after last ',' on line
        if itemStrip:
          intStrList.append(itemStrip)

  input_ids = getTensorFromIntStrList(intStrList)

  print("Got 'input_ids' from '{}'".format(filename) )
  print("  ", input_ids)
  print()
  decodedString = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

  print()
  print("tokenizer.batch_decode() from 'input_ids': ")
  print("                               ---------")
  print("  ", decodedString)

  return input_ids

def paragraphSummary(model, tokenizer, input_ids):
  summary_ids = model.generate(input_ids, num_beams=2, min_length=0, max_length=130)

  decodedString = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  
  print()
  print("tokenizer.batch_decode() from 'summary_ids': ")
  print("                               ===========")
  print("  ", decodedString)

# 24/1/24 DH:
def paragraphTextB4summary(filename):
  print()
  print("--------------- Orig text before summary --------------")
  print("                ++++++++++++++++++++++++")
  
  print("FILENAME: ", filename)
  print()
  with open(filename) as source :
    for line in source.readlines():
      # Remove newline character from each int printout line
      lineStrip = line.rstrip()

      print("LINE: '{}'".format(lineStrip))

  print("-------------------------------------------------------")

def removeBosEosTokens(idsTensor: torch.tensor) -> torch.tensor:
  # 22/1/24 DH: Taken from 'tokenization_roberta_fast.py::RobertaTokenizerFast(PreTrainedTokenizerFast)' :
  #               'bos' = 'beginning of sequence'
  #               'eos' = 'end of sequence'

  # Process: (Complex number domain)
  # --------
  #                   [1]          [2]                [3]                       [4]                               [5]
  # 2-D Tensor => 2-D Numpy => 1-D string => Remove unwanted chars => Split string into List => Convert int List back to 2-D Tensor

  idsNumpy2D = np.array(idsTensor)
  idsStr = str(idsNumpy2D[0])

  # 22/1/24 DH: Remove the '[]' (from Numpy conversion) then '0' (bos) and '2' (eos) from string
  idsStr = idsStr.strip('[]')
  idsStr = idsStr.strip('0 2')

  # 22/1/24 DH: Split on arbitrary sized whitespace
  idsStrList = idsStr.split()
  
  return getTensorFromIntStrList(idsStrList)

def tokenizeLine(tokenizer, line) -> dict:

  inputs = tokenizer(line, max_length=1024, return_tensors="pt")

  lineDict = {}
  lineDict['text'] = line

  input_ids = inputs["input_ids"]
  input_ids = removeBosEosTokens(input_ids)
  lineDict['input_ids'] = input_ids

  return lineDict

# 20/1/24 DH:
def paragraphText(model, tokenizer, filename, summaryWanted) -> dict :
  bartDictList = []

  with open(filename) as source :  
    print()
    print("Filename: ", filename)

    textLines = [line.strip() for line in source.readlines() if line.strip()]

    if len(textLines) > 1:
      print("text len: ", len(textLines))
      print()
      for line in textLines:
        #print("LINE: '{}'".format(line))

        bartDictList.append( tokenizeLine(tokenizer, line) )

    else:
      line = textLines[0]

      bartDictList.append( tokenizeLine(tokenizer, line) )

    print()
    print("------------------------------------------------------")
    num = 0
    for dict in bartDictList:
      line = dict['text']
      input_ids = dict['input_ids']

      # 24/1/24 DH: The number of the paragraph in the sequence
      num += 1
      dict['num'] = num

      print("{}) Contents: '{}'".format(num, line))
      print("'input_ids':", input_ids)
      print()
      if summaryWanted:
        paragraphSummary(model, tokenizer, input_ids)
        print("############################################")
        print()
    print("------------------------------------------------------")

    # Prev return: 'input["input_ids"]' (which has 'torch.tensor' type)

    return bartDictList

# 23/1/24 DH:
def calcTokenTotals(tokenizer, bartDictList: dict) -> dict:
  totalsDict = {}
  
  print("")
  for line in bartDictList:
    print("{}) {}".format(line['num'], line['input_ids']))

    # 'input_ids' eg: tensor([[50265,     4,  1437, 50266,  1437, 50267,     4,  1437, 50268]])
    idsNumpy2D = np.array( line['input_ids'] )
    for token in idsNumpy2D[0]:
      if token in totalsDict:
        totalsDict[token] += 1
      else:
        totalsDict[token] = 1
      
      # 23/1/24 DH: ffs...https://wiki.python.org/moin/Py3kStringFormatting
      #print("  {0:<5} = {1}".format(token, totalsDict[token]))
  
  print()
  print("Token histogram")
  print("---------------")
  # Keys are listed in the order in which they are allocated
  tokenDict = {}
  for token in totalsDict:
    
    keyFromVal = tokenizer.batch_decode([token], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # 24/1/24 DH: Entries are keyed by {'tokenizer.json' / 'added_tokens.json'} allocated token ID number and contain:
    #               "number": the number of times a token is used
    #               "key":    the text string that is allocated the token ID number
    tokenDict[token] = {"number": totalsDict[token], "key": keyFromVal}
    
    #print("{0:<5} = {1}, '{2}'".format(token, totalsDict[token], keyFromVal))
    # 24/1/24 DH: 'tokenDict' is a dict of dict's (NOT A 2-D ARRAY!)
    print("{0:<5} = {1}, '{2}'".format(token, tokenDict[token]['number'], tokenDict[token]['key']))

  # ------------------------- DEBUG ------------------------------
  print()
  print("Token map from 'tokenizer': ", tokenizer.__class__)
  
  cnt = 0
  print("('tokenizer.vocab' order changes every time, but 'totalsDict' keys are in order of creation...WHY...???)")
  for token in tokenizer.vocab:
    cutoff = 3
    if cnt < cutoff:
      print("  '{}' : '{}'".format(token, tokenizer.vocab[token]))
    if cnt == cutoff:
      print()
      print("    breaking out of 'tokenizer.vocab' key loop...")
      print()
      print("  (https://peps.python.org/pep-0234/#dictionary-iterators)")
      break

    cnt += 1
  # --------------------------------------------------------------

  return tokenDict

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
# 20/1/24 DH: EXAMPLE FOUND IN: 'models/bart/modeling_bart.py:571'
#   14/1/24 DH: "models/bart/modeling_bart.py: 1294: class BartForConditionalGeneration(BartPretrainedModel):"
# -------------------------------------------------------------------------------------------------------------
def runNewVocabTest(summaryWanted = False):
  tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

  model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
  print()
  print("model.generate: ", model.generate.__qualname__)

  """
  """
  print("========================================================================")
  filename = "bart-new-vocab-codes.txt"
  # 20/1/24 DH: 'orig' codes work with, or without, 'added_tokens.json'
  #filename = "bart-new-vocab-orig.txt"
  input_ids = paragraphDecode(tokenizer, filename)
  print("========================================================================")


  """
  """
  print()
  print("------------------- SUMMARY TEST USING OLD CODES WITH NEW VOCAB -------------------")
  if summaryWanted:
    filename = "new-vocab-test.txt"
    paragraphText(model, tokenizer, filename, summaryWanted=summaryWanted)
  print("-----------------------------------------------------------------------------------")

  filename = "xl-bully-ban-short2.txt"
  paragraphTextB4summary(filename)

  filename = "num_beams-BARTinput.txt"
  bartDictList = paragraphText(model, tokenizer, filename, summaryWanted=summaryWanted)
  hist = calcTokenTotals(tokenizer, bartDictList)
  # -------------------------------------------------------------------------------------------------------------

# 25/1/24 DH:
if __name__ == '__main__':
  #runNewVocabTest(summaryWanted = True)
  
  trainSeqClassModel()
  #trainSeq2SeqLM()
