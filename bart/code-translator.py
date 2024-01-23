# 14/1/24 DH:
from transformers.utils import is_torch_available

if is_torch_available():
  from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, AutoModelForSeq2SeqLM

from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
import numpy as np

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

def tokenizeLine(line) -> dict:

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

        bartDictList.append( tokenizeLine(line) )

    else:
      line = textLines[0]

      bartDictList.append( tokenizeLine(line) )

    print()
    print("------------------------------------------------------")
    num = 0
    for dict in bartDictList:
      line = dict['text']
      input_ids = dict['input_ids']

      num += 1
      print("{}) Contents: '{}'".format(num, line))
      print("'input_ids':", input_ids)
      print()
      if summaryWanted:
        paragraphSummary(model, tokenizer, input_ids)
        print("#########################")
        print()
    print("------------------------------------------------------")

    # Prev return: 'input["input_ids"]' (which has 'torch.tensor' type)

    return bartDictList

# --------------------------------------------
# 17/1/24 DH: Globals for 'paragraphSummary()'
# --------------------------------------------
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# 16/1/24 DH:
import inspect

print()
print("model.generate: ", model.generate.__qualname__)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

"""
filename = "bart-double-codes1.txt"
filename = "bart-double-codes2.txt"
"""

"""
"""
filename = "bart-new-vocab-codes.txt"
# 20/1/24 DH: 'orig' codes work with, or without, 'added_tokens.json'
#filename = "bart-new-vocab-orig.txt"
input_ids = paragraphDecode(tokenizer, filename)
print("========================================================================")

"""
"""
#filename = "new-vocab-test.txt"
filename = "num_beams-BARTinput.txt"
bartDictList = paragraphText(model, tokenizer, filename, summaryWanted=False)

# --------------------------------------------------------------------------------------------------------------

""" 20/1/24 DH: Taken from 'models/bart/modeling_bart.py:571'
ARTICLE_TO_SUMMARIZE = (
  "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
  "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
  "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"""

#MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
#autoSeq2Seq = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
# 14/1/24 DH: "models/bart/modeling_bart.py:class BartForConditionalGeneration(BartPretrainedModel):"

#print("AutoModelForSeq2SeqLM: ",AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn") )
# OUTPUT: 
"""
BartForConditionalGeneration(
  (model): BartModel(
    (shared): Embedding(50264, 1024, padding_idx=1)
    (encoder): BartEncoder(
    )
    (decoder): BartDecoder(
    )
  )
  (lm_head): Linear(in_features=1024, out_features=50264, bias=False)
)
"""



