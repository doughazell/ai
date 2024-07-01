# 25/6/24 DH:

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn.modules.linear
from datetime import datetime

# https://machinelearning.apple.com/research/overton

# 25/6/24 DH: """...""" adds "\n" to string at end of line

abstract = """We describe a system called Overton, whose main design goal is to support engineers in building, monitoring, and improving production 
machine learning systems. Key challenges engineers face are monitoring fine-grained quality, diagnosing errors in sophisticated applications, and 
handling contradictory or incomplete supervision data. Overton automates the life cycle of model construction, deployment, and monitoring by providing a 
set of novel high-level, declarative abstractions. Overton's vision is to shift developers to these higher-level tasks instead of lower-level machine learning tasks. 
In fact, using Overton, engineers can build deep-learning-based applications without writing any code in frameworks like TensorFlow. For over a year, 
Overton has been used in production to support multiple applications in both near-real-time applications and back-of-house processing. In that time, 
Overton-based applications have answered billions of queries in multiple languages and processed trillions of records reducing errors 1.7-2.9 times versus production systems.
"""

gFilename = "../bart/xl-bully-ban-short.txt"

# Get contents of file with added newlines OR the abstract (above)
def getFileContents(filename=False, lineWanted=False):
  if filename:
    with open(filename) as source :  
      print()
      print("Filename: ", filename)
      print("---------")

      txtLines = [line.strip() for line in source.readlines() if line.strip()]

      if lineWanted:
        # Adjust line number to index number
        txt = txtLines[lineWanted-1]
      else:
        txt = "\n".join(txtLines)
  else:
    txt = abstract

  print(f"{txt}")
  print()

  return txt

def printGeneratedIDs(tokenizer, generated_ids):
  #PREV: decodedString = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
  cnt = 0
  for pred in preds:
    cnt += 1
    print(f"{cnt}) len: {len(pred)}")
    print(f"  '{pred}'")
  
  print()

# Taken from 'ai/bart/code-translator.py'
def bartSummary(model_name, txtInclNewlines):
  # Create: BartForConditionalGeneration
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  print(f"FINE-TUNED MODEL: '{model_name}'")
  print(f"BART Summary ('{model.__class__}')")
  print( "------------")

  # Resetting cnt for next run (only necessary for multiple runs)
  torch.nn.modules.linear.Linear.fwdCnt = 0
  # Prevent Torch debug
  torch.nn.modules.linear.Linear.showDebug = False

  #PREV: inputs = tokenizer(line, max_length=1024, return_tensors="pt")
  input_ids = tokenizer.encode(txtInclNewlines, return_tensors="pt", add_special_tokens=True)
  startIds = [elem.item() for elem in input_ids[0][:5]]
  endIds = [elem.item() for elem in input_ids[0][-5:]]
  print(f"'input_ids': len = {len(input_ids[0])}, tokens = {startIds}...{endIds}")

  # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1612
  generated_ids = model.generate(input_ids, num_beams=10, min_length=0, max_length=500)
  printGeneratedIDs(tokenizer, generated_ids)

  print("No 'min_length=0'")
  generated_ids = model.generate(input_ids, num_beams=10, max_length=500)
  printGeneratedIDs(tokenizer, generated_ids)

def t5Summary(model_name, txtInclNewlines):
  # Create: T5ForConditionalGeneration
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  print(f"FINE-TUNED MODEL: '{model_name}'")
  print(f"T5 Summary ('{model.__class__}')")
  print( "----------")
  
  # Resetting cnt for next run (only necessary for multiple runs)
  torch.nn.modules.linear.Linear.fwdCnt = 0
  # Prevent Torch debug
  #print("NOT turning off 'torch.Linear' debug")
  torch.nn.modules.linear.Linear.showDebug = False

  # 25/6/24 DH: Works same without "summarize: "
  # Pre-trained/Fine-tuned obfuscation from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py#L363
  #input_ids = tokenizer.encode("summarize: " + abstract, return_tensors="pt", add_special_tokens=True)

  input_ids = tokenizer.encode(txtInclNewlines, return_tensors="pt", add_special_tokens=True)
  startIds = [elem.item() for elem in input_ids[0][:5]]
  endIds = [elem.item() for elem in input_ids[0][-5:]]
  print(f"'input_ids': len = {len(input_ids[0])}, tokens = {startIds}...{endIds}")

  # "kwargs": https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1615

  # FROM: https://huggingface.co/snrspeaks/t5-one-line-summary
  #generated_ids = model.generate(input_ids=input_ids, num_beams=10, max_length=500, repetition_penalty=2.5,
  #                              length_penalty=1, early_stopping=True, num_return_sequences=1)

  generated_ids = model.generate(input_ids=input_ids, num_beams=10, max_length=500)
  printGeneratedIDs(tokenizer, generated_ids)

  return

  print("Only 'num_breams=10', 'max_length=23', 'early_stopping=True'")
  generated_ids = model.generate(input_ids=input_ids, num_beams=10, max_length=23, early_stopping=True)
  printGeneratedIDs(tokenizer, generated_ids)

  print("Only 'num_breams=10', 'max_length=24', 'early_stopping=True'")
  generated_ids = model.generate(input_ids=input_ids, num_beams=10, max_length=24, early_stopping=True)
  printGeneratedIDs(tokenizer, generated_ids)

if __name__ == '__main__':
  
  txtInclNewlines = getFileContents(gFilename)
  #txtInclNewlines = getFileContents()
  
  # https://strftime.org/
  #nowStr = datetime.today().strftime('%-H:%-M:%-S')

  """
  # 26/6/24 DH: Now timing diff models
  before = datetime.now()
  # Prev summary model in 'ai/bart/code-translator.py'
  model_name = "facebook/bart-large-cnn"
  bartSummary(model_name, txtInclNewlines)
  after = datetime.now()

  timeDiff = round((after - before).total_seconds(), 1)
  print(f"  {timeDiff} secs")
  print()
  """

  before = datetime.now()
  # https://huggingface.co/snrspeaks/t5-one-line-summary
  model_name = "snrspeaks/t5-one-line-summary"
  t5Summary(model_name, txtInclNewlines)
  after = datetime.now()

  timeDiff = round((after - before).total_seconds(), 1)
  print(f"  {timeDiff} secs")


  
  
  


