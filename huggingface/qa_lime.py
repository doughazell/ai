# 19/2/24 DH:
import torch
import numpy as np
from transformers import T5ForQuestionAnswering, T5ForConditionalGeneration, BertForQuestionAnswering, AutoTokenizer, BartForQuestionAnswering
import matplotlib.pyplot as plt

def stripListLayer(_question, _context, _answer):
  if isinstance(_question, list):
    _question = _question[0]
  
  if isinstance(_context, list):
    _context = _context[0]
  
  if isinstance(_answer, list):
    _answer = _answer[0]

  return (_question, _context, _answer)

def printDatasetInfo(raw_datasets):
  
  raw_data = raw_datasets["train"][0]
  
  # ----------------------------------------------------------
  # 'datasets.arrow_dataset.py' line 2349: def __len__(self):
  #   Number of rows in the dataset.  
  #   return self.num_rows
  # ----------------------------------------------------------

  print()
  print("------ printDatasetInfo() ------")
  print(f"raw_datasets['train'] : {raw_datasets['train'].__class__}, num_rows: {raw_datasets['train'].num_rows}")
  print(f"raw_data: {raw_data.__class__}, {raw_data.keys()}")
  print()
  for key in raw_data:
    if isinstance(raw_data[key], list):
      raw_data_key = raw_data[key][0]
    else:
      raw_data_key = raw_data[key]

    print(f"{key}) {raw_data_key}")
    print("  ...")
    print("---")

# 19/2/24 DH: Taken from 'real_bday_paradox.py'
def graphTokenVals(startVals, endVals):
  
  # from 'lime_utils.py::displayCoefficients()'
  #plt.axhline(y=0, color='green', linestyle='-')

  plt.plot(range(len(startVals)), startVals, label="Start logits")
  plt.plot(range(len(endVals)), endVals, label="End logits")
  plt.legend(loc="upper left")
  plt.ylim(ymin=-10, ymax=10)
  plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  plt.title("Logits by token ID")
  plt.xlabel("Token ID")
  plt.ylabel("Logit value")

  plt.draw()
  plt.show()

def transformerLIMEing(output, tokenizer, all_tokens):
  max_start_logits_idx = torch.argmax(output["start_logits"])
  max_end_logits_idx = torch.argmax(output["end_logits"])
  answer_tokens = all_tokens[max_start_logits_idx : max_end_logits_idx + 1]

  # 6/2/24 DH: "RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
  #             FROM: start_logits_dump = np.array(output['start_logits'])
  start_logits_dumpORIG = start_logits_dump = output['start_logits'].detach().numpy()
  end_logits_dumpORIG = end_logits_dump = output['end_logits'].detach().numpy()
  
  tokNum = 10
  
  # 6/2/24 DH: [-5:][::-1] MEANS...take an array from end-5, then take an array with reverse order
  start_logits_dump = np.sort(start_logits_dump[0])[-tokNum:][::-1]
  start_logits_dump = [np.round(value) for value in start_logits_dump]

  end_logits_dump = np.sort(end_logits_dump[0])[-tokNum:][::-1]
  end_logits_dump = [np.round(value) for value in end_logits_dump]

  print()
  print(f"Transformer LIME'ing (printing {tokNum} from {start_logits_dumpORIG.shape})")
  print( "--------------------")
  maxStartLogit = start_logits_dumpORIG[0][max_start_logits_idx]
  maxEndLogit = end_logits_dumpORIG[0][max_end_logits_idx]

  print(f"start_logits max: {max_start_logits_idx}, value: {np.round(maxStartLogit)}")
  print(f"end_logits max: {max_end_logits_idx}, value: {np.round(maxEndLogit)}")
  print()
  print(f"output['start_logits']: {len(output['start_logits'][0])}, {start_logits_dump}")
  print(f"output['end_logits']:   {len(output['end_logits'][0])}, {end_logits_dump}")
  print()
  print(f"all_tokens: {len(all_tokens)}, {all_tokens.__class__}")
  print(tokenizer.decode(tokenizer.convert_tokens_to_ids(all_tokens)))

  startVals = []
  endVals = []
  for i in range(len(all_tokens)):
    # 2 layers of decoding: tok -> id -> syllable/word
    tokWord = tokenizer.decode( tokenizer.convert_tokens_to_ids(all_tokens[i]) )
    startTokVal = np.round(start_logits_dumpORIG[0][i])
    endTokVal = np.round(end_logits_dumpORIG[0][i])

    startVals.append(startTokVal)
    endVals.append(endTokVal)

    print(f"{i:<3} {tokWord:10} {startTokVal:>5} (start) {endTokVal:>10} (end)")
  
  print()
  print(f"answer_tokens: {len(answer_tokens)}")
  print(f"  '{answer_tokens}'")
  print("")
  print(f"  answer_tokens ids: '{tokenizer.convert_tokens_to_ids(answer_tokens)}'")
  print("  ------------------")
  print(f"  tokenizer.decode() ids: '{tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))}'")
  print()

  answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
  print("ANSWER: ", answer)
  print("------")

  graphTokenVals(startVals, endVals)

# 24/3/24 DH:
def getCorrectModelAndTokenizer(model_name, model_args, origTokenizer):
  print( "###################################")
  print(f"LOADING: {model_name}")
  print( "###################################")

  # Needed for HuggingFace Hub names like, "sjrhuschlee/flan-t5-base-squad2"
  if "t5" in model_name.lower():
    # 4/2/24 DH: See comment below re 'T5ForConditionalGeneration' vs 'T5ForQuestionAnswering'
    #model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = T5ForQuestionAnswering.from_pretrained(model_name)
  
  # https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForQuestionAnswering
  elif "bart" in model_args.model_name_or_path.lower():
    model = BartForQuestionAnswering.from_pretrained(model_name)
  
  elif "t5" in model_args.model_name_or_path.lower():
    model = T5ForQuestionAnswering.from_pretrained(model_name)

  else:
    print()
    print( "  *****")
    print(f"  USING: BertForQuestionAnswering.from_pretrained({model_name})")
    print( "  *****")
    print()
    model = BertForQuestionAnswering.from_pretrained(model_name)

  # 27/2/24 DH: Need to change the tokenizer as well...!!!
  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=model_args.cache_dir,
    use_fast=True,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
  )

  print()
  print("Changing Tokenizer")
  print(f"  from: {origTokenizer.__class__}")
  print(f"  to:   {tokenizer.__class__}")
  print()

  return (model, tokenizer)


def getModelOutput(raw_data, tokenizer, data_args, model_args):

  # Model needs to set here (so can easily use HuggingFace Hub model or local directory specified by 'model_name_or_path')
  # -----------------------
  #model_name = "sjrhuschlee/flan-t5-base-squad2"

  #model_name = "previous_output_dir-Google-BERT"
  #model_name = "previous_output_dir-Google-BERT/checkpoint-14216"
  
  # 23/3/24 DH:
  #model_name = "previous_output_dir-Google-T5"
  model_name = "previous_output_dir-Fairseq-BART"
  
  # ----------------------------------------------------------------------------
  (model, tokenizer) = getCorrectModelAndTokenizer(model_name, model_args, tokenizer)

  print()
  print("Model type in Q&A: ", model.__class__)
  print()
  # ----------------------------------------------------------------------------
  
  #tokenizer.cls_token = '<cls>'
  #question = f"{tokenizer.cls_token}{raw_data['question']}"
  question = raw_data['question']
  context = raw_data['context']
  answer = raw_data['answers']
  #context = context.replace('as a child, and rose to fame in the late 1990s ', '')

  # 14/2/24 DH:
  (question, context, answer) = stripListLayer(question, context, answer)
  #answer = raw_data['answers']['text'][0]
  answer = answer['text'][0]
  """
  question = "When did Beyonce become famous?"
  context = "Beyonce started singing as a child but became famous in the 1990s"
  #context = "Beyonce became famous in 1990s and then went onto selling many records"
  answer = ""
  """

  print()
  if answer == "":
    print("Params from script")
  else:
    print(f"Params from '{data_args.dataset_name if data_args.dataset_name else data_args.train_file}'")
  print("-------------------------")
  print("QUESTION: ", question)
  print("CONTEXT: ", context)
  print("ANSWER: ", answer)

  """
  
  # DEBUG FOR 'encoding' SENT TO "model(encoding['input_ids']")
  # -----------------------------------------------------------
  questionEncoding = tokenizer(question, return_tensors="pt")
  contextEncoding = tokenizer(context, return_tensors="pt")
  print("question encoding: ", questionEncoding['input_ids'])
  print("-------")
  print("context encoding: ", contextEncoding['input_ids'])
  """
  
  # 'transformers/tokenization_utils_base.py(2731)__call__()'
  encoding = tokenizer(question, context, return_tensors="pt")

  """
  
  print("Just passing 'encoding['input_ids'] to model(): ")
  print(encoding['input_ids'])
  """
  
  # T5ForConditionalGeneration results in: "ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds"
  output = model(
    encoding["input_ids"],
    #attention_mask=encoding["attention_mask"]
  )

  all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
  transformerLIMEing(output, tokenizer, all_tokens)

  
  