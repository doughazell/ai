# 19/2/24 DH:
import torch
import numpy as np
from transformers import (
  T5ForQuestionAnswering, 
  T5ForConditionalGeneration, 
  BertForQuestionAnswering, 
  AutoTokenizer, 
  BartForQuestionAnswering, 
  RobertaForQuestionAnswering
)
import matplotlib.pyplot as plt

def getListElem(_questions, _contexts, _answers, index):
  if isinstance(_questions, list):
    _question = _questions[index]
  
  # 7/4/24 DH: Designed to cope with JSON data plus Arrow added list layer + normal Arrow Datasets
  else:
    _question = _questions
  
  if isinstance(_contexts, list):
    _context = _contexts[index]
  
  # 7/4/24 DH:
  else:
    _context = _contexts
  
  if isinstance(_answers, list):
    _answer = _answers[index]
  
  # 7/4/24 DH:
  else:
    _answer = _answers

  return (_question, _context, _answer)

def printDatasetInfo(raw_datasets, datasetsIdx):
  
  raw_data = raw_datasets["train"][datasetsIdx]
  
  # ----------------------------------------------------------
  # 'datasets.arrow_dataset.py' line 2349: def __len__(self):
  #   Number of rows in the dataset.  
  #   return self.num_rows
  # ----------------------------------------------------------

  print()
  print("------ printDatasetInfo() ------")
  print(f"raw_datasets['train'] : {raw_datasets['train'].__class__}, num_rows: {raw_datasets['train'].num_rows}")
  print(f"USING idx: '{datasetsIdx}' from 'raw_datasets = load_dataset()'")
  print(f"  raw_data : {raw_data.__class__}, {raw_data.keys()}")
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
# 27/5/24 DH: TODO: Decide whether to keep 'lastGraph' cascade from: 
#                     'test-qa-efficacy.py::runRandSamples(...)'
#                     'getModelOutput(...)'
#                     'transformerLIMEing(...)'
#                     'graphTokenVals(...)'
#  (since we want to display graphs + 'answerDictDict' AT SAME TIME)
def graphTokenVals(startVals, endVals, tokWordStr, tokIDStr, lastGraph=False):
  # 27/5/24 DH: Create new canvas to prevent 'plt.show(block=False)' being drawn onto initial graph
  plt.figure()

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

  # From 'stop_trainer_utils.py::displayIntervals(intervalLog)'
  #legendStr = f"{mTxt:>24} {mRound}\n{bTxt:>22} {bRound}\n{rvTxt:>23} {rvalue}\n{sdTxt:>20} {stderr}\n{inTxt} {intercept_stderr}"
  #plt.figtext(0.2, 0.2, legendStr)
  legendStr = f"{tokWordStr}\n{tokIDStr}"
  plt.figtext(0.15, 0.2, legendStr)

  # 27/5/24 DH: Unable to change window location with 'macos' (incl "matplotlib.use('QtAgg')", "matplotlib.use('TkAgg')")
  #             https://matplotlib.org/stable/users/explain/figure/backends.html#selecting-a-backend)

  #    matplotlib.use('TkAgg')
  #    plt.get_current_fig_manager().window.wm_geometry("+600+400") # move the window
  #      RESULT: "Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'macosx' is currently running"

  #plt.draw()
  plt.show(block=lastGraph)

# 7/4/24 DH: Used with custom JSON cut-down data to learn about BERT (not SQuAD)
def getTokStrings(all_tokens, tokenIDs, tokenizer, printOut=False):
  
  tokWordStr = ""
  tokIDStr = ""
  tokIDStrPLT = ""
  
  for i in tokenIDs:
    tokWord = tokenizer.decode( tokenizer.convert_tokens_to_ids(all_tokens[i]) )
    tokWordStr += f"{tokWord},"

    tokWordLen = len(tokWord)
    tokWordLenPLT = tokWordLen
    if tokWordLen > 3:
      tokWordLenPLT += 1

    tokWordSpace = tokWordLen - len(str(i))
    sp = ""
    # This renders different on cmd line and 'plt.draw()'
    tokIDStr += f"{sp:>{tokWordSpace}}{i},"
    tokIDStrPLT += f"{sp:>{tokWordLenPLT}}{i},"

  if printOut:
    print(tokWordStr)
    print(tokIDStr)
    print()

  return (tokWordStr, tokIDStrPLT)

def transformerLIMEing(output, tokenizer, all_tokens, printOut=False, lastGraph=False):
  max_start_logits_idx = torch.argmax(output["start_logits"])
  max_end_logits_idx = torch.argmax(output["end_logits"])
  answer_tokens = all_tokens[max_start_logits_idx : max_end_logits_idx + 1]

  # 6/2/24 DH: "RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
  #             FROM: start_logits_dump = np.array(output['start_logits'])
  start_logits_dumpORIG = start_logits_dump = output['start_logits'].detach().numpy()
  end_logits_dumpORIG = end_logits_dump = output['end_logits'].detach().numpy()
  
  tokNum = 10
  
  # 6/2/24 DH: "[-5:][::-1]" MEANS take an array from end-5, then take an array with reverse order
  start_logits_dump = np.sort(start_logits_dump[0])[-tokNum:][::-1]
  start_logits_dump = [np.round(value) for value in start_logits_dump]

  end_logits_dump = np.sort(end_logits_dump[0])[-tokNum:][::-1]
  end_logits_dump = [np.round(value) for value in end_logits_dump]

  if printOut:
    print()
    print(f"Transformer LIME'ing (printing {tokNum} from {start_logits_dumpORIG.shape})")
    print( "--------------------")

  maxStartLogit = start_logits_dumpORIG[0][max_start_logits_idx]
  maxEndLogit = end_logits_dumpORIG[0][max_end_logits_idx]

  if printOut:
    print(f"start_logits max: {max_start_logits_idx}, value: {np.round(maxStartLogit)}")
    print(f"end_logits max: {max_end_logits_idx}, value: {np.round(maxEndLogit)}")
    print()
    print(f"output['start_logits']: {len(output['start_logits'][0])}, {start_logits_dump}")
    print(f"output['end_logits']:   {len(output['end_logits'][0])}, {end_logits_dump}")
    print()

  all_tokens_len = len(all_tokens)
  
  if printOut:
    print(f"all_tokens: {all_tokens_len}, {all_tokens.__class__}")
  
  tokenIDs = range(all_tokens_len)

  # 7/4/24 DH:
  tokDebugLen = 20
  if all_tokens_len < tokDebugLen:
    (tokWordStr, tokIDStrPLT) = getTokStrings(all_tokens, tokenIDs, tokenizer, printOut)
    print(f"  (adding 'word-token' key to graph since 'all_tokens' < {tokDebugLen})")
    print()
  else:
    tokWordStr = "" 
    tokIDStrPLT = ""

  startVals = []
  endVals = []
  for i in tokenIDs:
    # 2 layers of decoding: tok -> id -> syllable/word
    tokWord = tokenizer.decode( tokenizer.convert_tokens_to_ids(all_tokens[i]) )
    startTokVal = np.round(start_logits_dumpORIG[0][i])
    endTokVal = np.round(end_logits_dumpORIG[0][i])

    # Used for 'graphTokenVals(startVals, endVals)'
    startVals.append(startTokVal)
    endVals.append(endTokVal)

    if printOut:
      print(f"{i:<3} {tokWord:10} {startTokVal:>5} (start) {endTokVal:>10} (end)")
  
  if printOut:
    print()
    print(f"answer_tokens: {len(answer_tokens)}")
    print(f"  '{answer_tokens}'")
    print("")
    print(f"  ANSWER from token '{max_start_logits_idx}' to '{max_end_logits_idx}'")
    print(f"  answer_tokens ids: '{tokenizer.convert_tokens_to_ids(answer_tokens)}'")
    print("  ------------------")
    print(f"  tokenizer.decode() ids: '{tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))}'")
    print()

  answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
  print("ANSWER: ", answer)
  print("------")

  #print("NOT CALLING: 'graphTokenVals()'")
  graphTokenVals(startVals, endVals, tokWordStr, tokIDStrPLT, lastGraph)

  return (answer, max_start_logits_idx, max_end_logits_idx)

# 24/3/24 DH:
def getCorrectModelAndTokenizer(model_name, model_args):
  print( "###################################")
  print(f"LOADING: {model_name}")
  print( "###################################")

  # Needed for HuggingFace Hub names like, "sjrhuschlee/flan-t5-base-squad2"
  if "t5" in model_name.lower():
    # 4/2/24 DH: See comment below re 'T5ForConditionalGeneration' vs 'T5ForQuestionAnswering'
    #model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = T5ForQuestionAnswering.from_pretrained(model_name)

  # 9/4/24 DH:
  elif "roberta" in model_name.lower():
    model = RobertaForQuestionAnswering.from_pretrained(model_name)

  # The default model uses: "model_name = training_args.output_dir", so need to use 'model_args'
  # --------------------------------------------------------------------------------------------

  # https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForQuestionAnswering
  elif "bart" in model_args.model_name_or_path.lower():
    model = BartForQuestionAnswering.from_pretrained(model_name)
  
  elif "t5" in model_args.model_name_or_path.lower():
    model = T5ForQuestionAnswering.from_pretrained(model_name)
  # --------------------------------------------------------------------------------------------
  
  else:
    """
    print()
    print( "  *****")
    print(f"  USING: BertForQuestionAnswering.from_pretrained({model_name})")
    print( "  *****")
    print()
    """
    # 1/6/24 DH: Bert is Pre-trained with: '"type_vocab_size": 2' BUT CAN RUN Q&A WITH: '"type_vocab_size": 1'
    model = BertForQuestionAnswering.from_pretrained(model_name, ignore_mismatched_sizes=True)

  # 27/2/24 DH: Need to change the tokenizer as well...!!!
  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=model_args.cache_dir,
    use_fast=True,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
  )

  return (model, tokenizer)


def getModelOutput(raw_data, data_args, model_args, printOut=False, lastGraph=False):
  print()
  print("------ Now [re]running the trained model for Q&A ------")

  # Model needs to set here (so can easily use HuggingFace Hub model or local directory specified by 'output_dir')
  # -----------------------
  #model_name = "sjrhuschlee/flan-t5-base-squad2"
  
  # 8/4/24 DH: Downloading to prevent weeks of fine-tuning work on SQuAD2
  #         (as used by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L73)
  #model_name = "deepset/bert-base-cased-squad2"
  #model_name = "deepset/bert-base-uncased-squad2"
  #model_name = "deepset/roberta-base-squad2"

  # Initial training of BERT/SQuAD
  # (from: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering#fine-tuning-bert-on-squad10)
  model_name = "previous_output_dir-Google-BERT"
  #model_name = "previous_output_dir-Google-BERT/checkpoint-14216"

  #model_name = training_args.output_dir

  # 26/3/24 DH: BERT/BART trained batch size (default 8) sets in 1st epoch 'forward()' hook so graphing untrained models
  """
  print("    ---------------------")
  print("*** USING UNTRAINED MODEL ***")
  print("    ---------------------")
  """
  
  #model_name = "google-bert/bert-base-uncased"
  #model_name = "facebook/bart-base"
  
  # ----------------------------------------------------------------------------
  (model, tokenizer) = getCorrectModelAndTokenizer(model_name, model_args)

  print()
  print("Model type in Q&A: ", model.__class__)
  print()
  # ----------------------------------------------------------------------------
  
  #tokenizer.cls_token = '<cls>'
  #question = f"{tokenizer.cls_token}{raw_data['question']}"
  questions = raw_data['question']
  contexts = raw_data['context']
  answers = raw_data['answers']
  #context = context.replace('as a child, and rose to fame in the late 1990s ', '')

  # 14/2/24 DH: Fixed when using JSON lists
  index = 2 #index = (#item - 1)

  # 27/5/24 DH: Now randomising non-training run sample

  # 7/4/24 DH: Designed to cope with JSON data plus Arrow added list layer (using index) + normal Arrow Datasets
  (question, context, answer) = getListElem(questions, contexts, answers, index)
  expAnswer = answer['text'][0]

  if answer == "":
    print("Params from script")
  else:
    print(f"Params from '{data_args.dataset_name if data_args.dataset_name else data_args.train_file}'")
  print("-------------------------")
  if data_args.train_file:
    print(f"JSON IDX: {index}")
  print("QUESTION: ", question)
  print("CONTEXT: ", context)
  print("EXPECTED ANSWER: ", expAnswer)

  # --- DEBUG ---
  # 27/3/24 DH: Taken from "(Pdb) input_ids[0]" ('BertForQuestionAnswering.forward()' in 'transformers/models/bert/modeling_bert.py')

  # FIRST ENTRY 2nd time
  #debug_tokens = [101, 2043, 2003, 6350, 1029,  102, 6350, 2003, 2012, 1021, 3286,  102]
  #debug_txt = tokenizer.decode(debug_tokens)
  #print(f"debug_txt: {debug_txt}")
  # -------------

  # 'transformers/tokenization_utils_base.py(2731)__call__()'
  encoding = tokenizer(question, context, return_tensors="pt")
  
  if printOut:
    print()
    print("Just passing 'encoding['input_ids'] to model(): ")
    print(f"  {encoding['input_ids'][0].tolist()}")
    print()
  
  print()
  # 28/5/24 DH: NO NEWLINE after heading since 'Linear.forward()' printout is designed to follow TQDM lines (that remain on printout)
  #print("======================== HUGGINGFACE NON-TRAINING RUN ==========================", end='')
  print("======================== HUGGINGFACE NON-TRAINING RUN ==========================")
  # T5ForConditionalGeneration results in: "ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds"
  
  # 31/5/24 DH: Handled by 'BertForQuestionAnswering.forward()' (via usual Torch hooks)
  print()
  # 31/5/24 DH: NO NEWLINE after "aid-memoire" since 'Linear.forward()' printout is designed to follow TQDM lines (that remain on printout)
  print("  CALLING: 'model(encoding['input_ids'])' which gets HANDLED BY 'BertForQuestionAnswering.forward()'", end='')

  output = model(
    encoding["input_ids"],
    #attention_mask=encoding["attention_mask"]
  )
  print("------------------------ END: HUGGINGFACE NON-TRAINING RUN ---------------------")

  # 24/4/24 DH: "pip install torchinfo" (https://github.com/TylerYep/torchinfo) "(formerly torch-summary)"
  print()
  print("-----------------------------------------")
  print("NO LONGER CALLING: torchinfo.summary()")
  print("                   torchview.draw_graph()")
  print("-----------------------------------------")
  print()
  """
  from torchinfo import summary
  with open('bert_qa.summary', 'w') as sys.stdout:
    summary(model, depth=5, verbose=1)
  # ...reset 'sys.stdout'
  sys.stdout = sys.__stdout__

  # --------------------------------------------------
  
  # 26/4/24 DH: Also try: https://github.com/mert-kurttutan/torchview, "pip install torchview; pip install graphviz"
  from torchview import draw_graph
  model_graph = draw_graph(model, input_data=encoding, depth=5, expand_nested=True, hide_inner_tensors=False)
  # 29/4/24 DH: Now added to 'draw_graph()' since adding legend
  #model_graph.visual_graph.render()
  """

  all_tokens = tokenizer.convert_ids_to_tokens( encoding["input_ids"][0].tolist() )
  tokenLen = len(all_tokens)
  (answer, startIdx, endIdx) = transformerLIMEing(output, tokenizer, all_tokens, printOut, lastGraph)

  return (tokenLen, question, expAnswer, answer, startIdx, endIdx)

  
  