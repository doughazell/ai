# ------------------------------------------------------------------------------------
# 25/1/24 DH: Model fine tuning
#             -----------------
# https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt
# https://huggingface.co/docs/transformers/training#train-with-pytorch-trainer
# ------------------------------------------------------------------------------------

from transformers.utils import is_torch_available
if is_torch_available():
  from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, AutoModelForSeq2SeqLM

from transformers import AutoTokenizer, DataCollatorWithPadding, EvalPrediction

# 25/1/24 DH: https://huggingface.co/docs/transformers/training#train-with-pytorch-trainer

from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

# 26/1/24 DH:
def printPubMedDataset(tokenized_datasets, train_dataset):
  # 25/1/24 DH: https://arrow.apache.org/docs/python/index.html
  #             "It contains a standardized column-oriented memory format that is able to represent flat and hierarchical data for 
  #              efficient analytic operations"

  print(tokenized_datasets)
  print()
  print("PubMed train_dataset: " )
  print()
  print("ARTICLE: ")
  txtList = train_dataset[0]["article"].split("\n")
  txtStrip = [line.lstrip() for line in txtList]
  print("".join(txtStrip))
  print("--------------")

  print("ABSTRACT: ")
  txtList = train_dataset[0]["abstract"].split("\n")
  txtStrip = [line.lstrip() for line in txtList]
  print("".join(txtStrip))
  print("--------------")

  # 'datasets/arrow_dataset.py':1732 def data = "The Apache Arrow table backing the dataset."
  #print("small_eval_dataset: ", small_eval_dataset.data)
  print()

def trainSeqClassModel():
  training_args = TrainingArguments(output_dir="test_trainer_SeqClass", evaluation_strategy="epoch")
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  model2Train = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

  metric = evaluate.load("accuracy")

  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

  # 25/1/24 DH: https://huggingface.co/datasets/yelp_review_full#data-fields
  #             'text': The review texts
  #             'label': Corresponds to the score associated with the review (between 1 and 5).

  # 25/1/24 DH: I prob need to load a dataset file that contains text of 'added_tokens.json' tokens
  dataset = load_dataset("yelp_review_full")
  #print(dataset["train"][101])
  #print(dataset["train"]["label"])

  def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1))
  small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1))

  # DATASET PRINTOUT REMOVED

  trainer = Trainer(
      model=model2Train,
      args=training_args,
      train_dataset=small_train_dataset,
      eval_dataset=small_eval_dataset,
      compute_metrics=compute_metrics,
  )
  trainer.train()

###############################################################################################################
# 27/1/24 DH: RESULTS IN: 
#             "ValueError: The model did not return a loss from the inputs, only the following keys: 
#              logits,past_key_values,encoder_last_hidden_state. 
#              For reference, the inputs it received are input_ids,attention_mask."
#
# Why did I want to finetune with new vocab in the first place (leading to this finetuning with orig vocab)?
# I wanted to tokenize on sequences in order to ID diffs in 'num_beams' summaries,
# (which then prevented 'model.generate()' working with new vocab).
#
# Spiral-method-of-learning:
# 1) Tokenize on sequence vocab with 'AutoTokenizer.from_pretrained("facebook/bart-large-cnn")'     = WORKS
# 2) Finetune 'AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)' = WORKS
# 3) Finetune 'AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")'                    = NOT WORK
#
# What next ?
# Revert vocab to orig if want to generate...eventually find out how to train 'Seq2SeqLM:BART'
#
# DEBUG:
# transformers/models/bart/modeling_bart.py(1413)forward()
# transformers/trainer.py(2779)compute_loss()
###############################################################################################################
def trainSeq2SeqLM():
  training_args = TrainingArguments(output_dir="test_trainer_Seq2Seq", evaluation_strategy="epoch")
  tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
  model2Train = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

  datasetName = "ccdv/pubmed-summarization"
  print()
  print("Loading: ", datasetName)
  print()
  dataset = load_dataset(datasetName)

  def tokenize_function(examples):
    return tokenizer(examples["article"], examples["abstract"], padding="max_length", truncation=True)

  print()
  print("Mapping: '{}'".format(dataset))
  tokenized_datasets = dataset.map(tokenize_function, batched=True)
  
  small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1))
  small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1))
  
  printPubMedDataset(tokenized_datasets, small_train_dataset)

  metric = evaluate.load("accuracy")
  # 26/1/24 DH: Based on 'AutoModelForSequenceClassification' for 'yelp_review_full' matching text with score 1-5
  def compute_metrics(eval_pred: EvalPrediction):
    # 26/1/24 DH: 'print()' or 'breakpoint()' here doesn't work (prob due to be accessed via 'compute_metrics' func pointer)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  
  # 26/1/24 DH:
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # 27/1/24 DH: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
  trainer = Trainer(
    model=model2Train,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,

    compute_metrics=compute_metrics,

    data_collator=data_collator,
    tokenizer=tokenizer,
  )
  print()
  print("Training: ", trainer.__class__)
  print()
  trainer.train()


# ---------------------------------------------------------------------------------------------------------
# ------------------------------------------------- DEBUG -------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

"""
##############################################
# 'transformers/models/auto/modeling_auto.py' :
##############################################
619: MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
      [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "BartForConditionalGeneration"),
        ...
      ]

1057: MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
        CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
      )

1131: class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
        _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


print("AutoModelForSeq2SeqLM: ",AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn") )
# OUTPUT: 
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



