#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
# 12/2/24 DH:
import json

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer

import transformers
from transformers import (
  AutoConfig,
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  DataCollatorForSeq2Seq,
  HfArgumentParser,
  Seq2SeqTrainingArguments,
  set_seed,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """

  model_name_or_path: str = field(
    metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  config_name: Optional[str] = field(
    default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
    default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
    default=None,
    metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
  )
  use_fast_tokenizer: bool = field(
    default=True,
    metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
  )
  model_revision: str = field(
    default="main",
    metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
  )
  token: str = field(
    default=None,
    metadata={
        "help": (
          "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
          "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
        )
    },
  )
  use_auth_token: bool = field(
    default=None,
    metadata={
        "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
    },
  )
  trust_remote_code: bool = field(
    default=False,
    metadata={
      "help": (
        "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
        "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
        "execute code present on the Hub on your local machine."
      )
    },
  )


@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training and eval.
  """

  dataset_name: Optional[str] = field(
    default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
  )
  dataset_config_name: Optional[str] = field(
    default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
  )
  context_column: Optional[str] = field(
    default="context",
    metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
  )
  question_column: Optional[str] = field(
    default="question",
    metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
  )
  answer_column: Optional[str] = field(
    default="answers",
    metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
  )
  train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
  validation_file: Optional[str] = field(
    default=None,
    metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
  )
  test_file: Optional[str] = field(
    default=None,
    metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
  )
  overwrite_cache: bool = field(
    default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
  )
  preprocessing_num_workers: Optional[int] = field(
    default=None,
    metadata={"help": "The number of processes to use for the preprocessing."},
  )
  max_seq_length: int = field(
    default=384,
    metadata={
      "help": (
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded."
      )
    },
  )
  max_answer_length: int = field(
    default=30,
    metadata={
      "help": (
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another."
      )
    },
  )
  val_max_answer_length: Optional[int] = field(
    default=None,
    metadata={
      "help": (
        "The maximum total sequence length for validation target text after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded. Will default to `max_answer_length`. "
        "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
        "during ``evaluate`` and ``predict``."
      )
    },
  )
  pad_to_max_length: bool = field(
    default=True,
    metadata={
      "help": (
        "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
        " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
      )
    },
  )
  max_train_samples: Optional[int] = field(
    default=None,
    metadata={
      "help": (
        "For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set."
      )
    },
  )
  max_eval_samples: Optional[int] = field(
    default=None,
    metadata={
      "help": (
        "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set."
      )
    },
  )
  max_predict_samples: Optional[int] = field(
    default=None,
    metadata={
      "help": (
        "For debugging purposes or quicker training, truncate the number of prediction examples to this "
        "value if set."
      )
    },
  )
  version_2_with_negative: bool = field(
    default=False, metadata={"help": "If true, some of the examples do not have an answer."}
  )
  null_score_diff_threshold: float = field(
    default=0.0,
    metadata={
      "help": (
        "The threshold used to select the null answer: if the best answer has a score that is less than "
        "the score of the null answer minus this threshold, the null answer is selected for this example. "
        "Only useful when `version_2_with_negative=True`."
      )
    },
  )
  doc_stride: int = field(
    default=128,
    metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
  )
  n_best_size: int = field(
    default=20,
    metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
  )
  num_beams: Optional[int] = field(
    default=None,
    metadata={
      "help": (
        "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
        "which is used during ``evaluate`` and ``predict``."
      )
    },
  )
  ignore_pad_token_for_loss: bool = field(
    default=True,
    metadata={
      "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
    },
  )

  def __post_init__(self):
    if (
      self.dataset_name is None
      and self.train_file is None
      and self.validation_file is None
      and self.test_file is None
    ):
      raise ValueError("Need either a dataset name or a training/validation file/test_file.")
    else:
      if self.train_file is not None:
        extension = self.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
      if self.validation_file is not None:
        extension = self.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
      if self.test_file is not None:
        extension = self.test_file.split(".")[-1]
        assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
      if self.val_max_answer_length is None:
        self.val_max_answer_length = self.max_answer_length


question_answering_column_name_mapping = {
  "squad_v2": ("question", "context", "answer"),
}

# 12/2/24 DH:
# ------------------------------------------------------------------------------------------------

################################################
# 13/2/24 DH: This prob needs to done via map()
#
# https://www.linkedin.com/pulse/demystifying-pythons-magic-methods-oscar-alfonso-tello-brise%C3%B1o/
################################################
def prune2DLists(raw_datasets):
  toplevelDict = raw_datasets.column_names
  print()
  print(f"Top level raw_datasets: {toplevelDict.__class__}, KEYS: {list(toplevelDict.keys())}")

  for key in toplevelDict:
    print()
    print("------")
    print(f"{key}: {raw_datasets[key]}")
    column_names = raw_datasets[key].column_names
    data = raw_datasets[key]

    for column in column_names:

      try:
        colData = data[column][0]
        cnt = 1

        # 14/2/24 DH: If it is len 1 then it is still nested EXCEPT "1st time JSON" where it is a LIST of 1 LIST...__len__()
        #             (rather than str, or dict for answers)
        #   States: 1) Arrow data, 2) 1st time JSON, 3) subsequent JSON
        while len(colData) == 1:
          colData = colData[0]
          cnt += 1

        if cnt > 1 or isinstance(colData, list):
          colData = colData[0]

        print(f"{column}: {data[column].__class__}, loops to data: {cnt}, colData type: {colData.__class__}, {colData}")
        print()
      except KeyError as e:
        print(e)
        print(colData)

      


  print()
  return raw_datasets

def displayLoadedDataset(raw_datasets, selectNum = 10):
  print(raw_datasets)
  
  isJSON = False

  # 13/2/24 DH:
  if not raw_datasets['train'].num_rows > selectNum:
    raw_datasets = prune2DLists(raw_datasets)
    dataMemoryMappedTable = raw_datasets['train']
    isJSON = True
    
  else:
    # -----------------------------------------------------------------
    # 13/2/34 DH: SQuAD dataset
    # -----------------------------------------------------------------

    print(f"SQuAD dataset selecting {selectNum} samples")
    selected_datasets = raw_datasets['train'].select(range(selectNum))
    dataMemoryMappedTable = selected_datasets.data

  column_names = dataMemoryMappedTable.column_names
  print(f"COLUMNS: {column_names}")
  print()
  
  for column in column_names:
    print(f"{column} from {column_names}")
    if isJSON:
      data = dataMemoryMappedTable[column][0]
    else:
      data = dataMemoryMappedTable[column]
    
    if column == "question":
      continue
  
    elif column == "answers":
      print(f"{column} (len: {len(data)}) {data.__class__}")

      for i in range(len(data)):
        # 12/2/24 DH: Need to cast to str to prevent 
        #    "TypeError: unsupported format string passed to pyarrow.lib.StringScalar.__format__" in f"" width field
        text = str(data[i]['text'][0])
        textStr = "'" + text + "'"
        
        # 12/2/24 DH: Notice double cast to get an int that can be used for array indexing
        ans_start = int( str(data[i]['answer_start'][0]) )
        ans_end = ans_start+len(text)
        
        if isJSON:
          question = str(dataMemoryMappedTable['question'][0][i])
          context = str(dataMemoryMappedTable['context'][0][i])
        else:
          question = str(dataMemoryMappedTable['question'][i])
          context = str(dataMemoryMappedTable['context'][i])
        
        print(f"  Q: {question}")
        print(f"  A: Text: {textStr:30} Answer start: {ans_start:<10} Context {i} (from {ans_start} to {ans_end}): {context[ans_start : ans_end]}")
        print()

    else:
      # 12/2/24 DH: 'data' is <class 'pyarrow.lib.ChunkedArray'>
      print(f"{column} (len: {len(data)}) {data.__class__}")
      print(f"  {data[0]}")
      print(f"  ...")

    print()
    print(f"...ok, so where has {column_names[1:]} gone then...???")

    # 13/2/24 DH: The JSON-based 'raw_datasets' will have had a 2-D List pruning above
    return raw_datasets


# 12/2/24 DH:
def saveDatasetToJSON(raw_datasets, selectNum = 10):
  print()
  print("saveDatasetToJSON()")
  print("-------------------")

  isJSON = False

  # 13/2/24 DH:
  if not raw_datasets['train'].num_rows > selectNum:
    selected_datasets = raw_datasets['train']
    isJSON = True
  else:
    selected_datasets = raw_datasets['train'].select(range(10))
    print(f"selected_datasets: {selected_datasets.__class__}, {selected_datasets.shape}")

  #dataDict = dataMemoryMappedTable.to_pydict()
  dataDict = selected_datasets.to_dict()

  #jsonDumpStr = json.dumps(dataDict)
  
  # 13/2/24 DH: All in 1 line
  #with open('data.json', 'w') as f:
  #  json.dump(dataDict, f)
  
  # 13/2/24 DH: Spread over multiple lines with indentations
  with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(dataDict, f, ensure_ascii=False, indent=2)
# ------------------------------------------------------------------------------------------------

# 7/2/24 DH: Access 'training_args' assigned in 'main()' from 'signal_handler()'
training_args = None
gStoppingFlag = False
gCheckpointNum = 0

def main():
  # See all possible arguments in src/transformers/training_args.py
  # or by passing the --help flag to this script.
  # We now keep distinct sets of args, for a cleaner separation of concerns.

  # 7/2/24 DH:
  global training_args

  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
  else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  if model_args.use_auth_token is not None:
    warnings.warn(
        "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
        FutureWarning,
    )
    if model_args.token is not None:
        raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
    model_args.token = model_args.use_auth_token

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    )

  if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # 2/2/24 DH:
  transformers.utils.logging.set_verbosity_error()

  # Log on each process the small summary:
  logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
  )
  #logger.info(f"Training/evaluation parameters {training_args}")

  ###############################################################################
  # Detecting last checkpoint.
  ###############################################################################



  ####################################################################################################################
  # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
  # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
  # (the dataset will be downloaded automatically from the datasets Hub).
  #
  # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
  # 'text' is found. You can easily tweak this behavior (see below)
  #
  # In distributed training, the load_dataset function guarantee that only one local process can concurrently
  # download the dataset.
  ####################################################################################################################
  print()
  print(f"Loading dataset: {data_args.dataset_name}")
  print()
  
  if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
      data_args.dataset_name,
      data_args.dataset_config_name,
      cache_dir=model_args.cache_dir,
      token=model_args.token,
      )
  else:
    data_files = {}
    if data_args.train_file is not None:
      data_files["train"] = data_args.train_file
      extension = data_args.train_file.split(".")[-1]

      raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        # 13/2/24 DH: 'field' is a '**config_kwargs' (additional keyword arguments):
        #             Keyword arguments to be passed to the `BuilderConfig` and used in the [`DatasetBuilder`].
        #field="data",
        cache_dir=model_args.cache_dir,
        token=model_args.token,
      )
  # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
  # https://huggingface.co/docs/datasets/loading_datasets.
  
  """
  "train_file": "data.json",
  """

  raw_datasets = displayLoadedDataset(raw_datasets)
  #print("Not currently calling: saveDatasetToJSON()")
  saveDatasetToJSON(raw_datasets)
  sys.exit("Yup, starting to take shape...")

  ######################################################################################
  # Load pretrained model and tokenizer
  #
  # Distributed training:
  # The .from_pretrained methods guarantee that only one local process can concurrently
  # download model & vocab.
  ######################################################################################
  config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
  )
  tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
  )
  model = AutoModelForSeq2SeqLM.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
  )
  print()
  print("Config return from AutoConfig: ", config.__class__)
  print("Tokenizer return from AutoTokenizer: ", tokenizer.__class__)
  print("Model returned from 'AutoModelForSeq2SeqLM': ", model.__class__)
  print()

  # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
  # on a small vocab and want a smaller embedding size, remove this test.
  embedding_size = model.get_input_embeddings().weight.shape[0]
  if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

  if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

  ###############################################################################
  # Preprocessing the datasets.
  # We need to generate and tokenize inputs and targets.
  ###############################################################################

  # 'column_names'
  # --------------
  if training_args.do_train:
    column_names = raw_datasets["train"].column_names
  else:
    logger.info("There is nothing to do. Please pass `do_train`.")
    return

  # Get the column names for input/target.
  dataset_columns = question_answering_column_name_mapping.get(data_args.dataset_name, None)

  # 'question_column'
  # -----------------
  if data_args.question_column is None:
    question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
  else:
    question_column = data_args.question_column
    if question_column not in column_names:
      raise ValueError(
        f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
      )

  # 'context_column'
  # ----------------
  if data_args.context_column is None:
    context_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
  else:
    context_column = data_args.context_column
    if context_column not in column_names:
      raise ValueError(
        f"--context_column' value '{data_args.context_column}' needs to be one of: {', '.join(column_names)}"
      )

  # 'answer_column'
  # ---------------
  if data_args.answer_column is None:
    answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
  else:
    answer_column = data_args.answer_column
    if answer_column not in column_names:
      raise ValueError(
        f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
      )

  # Temporarily set max_answer_length for training.
  max_answer_length = data_args.max_answer_length
  padding = "max_length" if data_args.pad_to_max_length else False

  if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning(
      "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
      f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
    )

  if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
      f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
      f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
  max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

  ####################################################################
  # 2/2/24 DH: Preprocess data functions
  ####################################################################
  
  # [1/4]
  def createReadableQAContext(questions, contexts, answers):
    print()
    print("----------------------------------------------")
    print(f"Max samples: {data_args.max_train_samples}")
    print()
    for i in range(len(questions)):
      print(f"{i}) {questions[i]}")
      if len(answers[i]['text']) == 0:
        print(f"  BLANK ANSWER")

    print()
    for i in range(len(questions)):
      print(f"{i}) INPUT:")
      print(f"  a) Question: {questions[i]}")
      print(f"  b) Context: {contexts[i]}")
      print()
      print(f"  c) Answer: {answers[i]}")
      print()
        
    print("----------------------------------------------")
    print()

  # [2/4]
  def preprocess_squad_batch(
    examples,
    question_column: str,
    context_column: str,
    answer_column: str,
    ) -> Tuple[List[str], List[str]]:

    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]

    def generate_input(_question, _context):
      return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

    inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
    # 3/2/24 DH: Some SQuAD validation answers are blank, "as roughly half of its questions don’t contain an answer",
    # (https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15743593.pdf)
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]

    # 2/2/24 DH:
    createReadableQAContext(questions, contexts, answers)

    return inputs, targets

  # [3/4]
  def preprocess_function(examples):
    print()
    print()
    print("*** TRAINING DATASET ***")

    inputs, targets = preprocess_squad_batch(examples, question_column, context_column, answer_column)

    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
    # Tokenize targets with text_target=...
    labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
      labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
      ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

  # [4/4]
  # Validation preprocessing

  ################################################################
  # 4/2/24 DH: Map training data
  ################################################################
  if training_args.do_train:
    if "train" not in raw_datasets:
      raise ValueError("--do_train requires a train dataset")
      
    train_dataset = raw_datasets["train"]

    if data_args.max_train_samples is not None:
      # We will select sample from whole data if argument is specified
      max_train_samples = min(len(train_dataset), data_args.max_train_samples)
      train_dataset = train_dataset.select(range(max_train_samples))

    # Create train feature from dataset
    with training_args.main_process_first(desc="train dataset map pre-processing"):
      train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
      )
    if data_args.max_train_samples is not None:
      # Number of samples might increase during Feature Creation, We select only specified max samples
      max_train_samples = min(len(train_dataset), data_args.max_train_samples)
      train_dataset = train_dataset.select(range(max_train_samples))

  # ---------------------------------------------------------------------------------------
  
  # ---------------------------------------------------------------------------------------
  

  ##################################################################
  # Data collator
  ##################################################################
  


  ##################################################################
  # Initialize our Trainer
  ##################################################################
  
  

  ##############################################################################
  # 3/2/24 DH: Now run the trained model for Q&A
  ##############################################################################
  print()
  print("train_dataset:")
  print(train_dataset)

  #train_dataset = train_dataset.select(range(max_train_samples))
  raw_data = raw_datasets["train"][0]
  
  # ----------------------------------------------------------
  # 'datasets.arrow_dataset.py' line 2349: def __len__(self):
  #   Number of rows in the dataset.  
  #   return self.num_rows
  # ----------------------------------------------------------

  print()
  print(f"raw_datasets['train'] : {raw_datasets['train'].__class__}, num_rows: {raw_datasets['train'].num_rows}")
  print(f"raw_data: {raw_data.__class__}, {raw_data.keys()}")
  print()
  for key in raw_data:
    print(f"{key}) {raw_data[key]}")
    print("---")

  #print("Returning before 'HACK ZONE'...")
  #return

  # ================================= HACK ZONE ==================================
  #                                   ---------
  import torch

  from transformers import AutoModelForQuestionAnswering, T5ForQuestionAnswering, T5ForConditionalGeneration

  # https://huggingface.co/sjrhuschlee/flan-t5-base-squad2
  #
  # "The <cls> token must be manually added to the beginning of the question for this model to work properly. It uses the <cls> token to be
  #  able to make "no answer" predictions. The t5 tokenizer does not automatically add this special token which is why it is added manually."

  model_name = "sjrhuschlee/flan-t5-base-squad2"
  #model_name = "previous_output_dir"
  #model_name = "previous_output_dir/checkpoint-28782"

  #model = AutoModelForQuestionAnswering.from_pretrained(model_name)
  #tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  # 4/2/24 DH: See comment below re 'T5ForConditionalGeneration' vs 'T5ForQuestionAnswering'
  #model = T5ForConditionalGeneration.from_pretrained(model_name)
  model = T5ForQuestionAnswering.from_pretrained(model_name)

  print()
  print("Model type in Q&A: ", model.__class__)
  print()
  
  #question = f"{tokenizer.cls_token}Why did Putin invade Ukraine?"

  """
  #tokenizer.cls_token = '<cls>'
  #question = f"{tokenizer.cls_token}{raw_data['question']}"
  question = raw_data['question']
  context = raw_data['context']
  answer = raw_data['answers']['text'][0]
  #context = context.replace('as a child, and rose to fame in the late 1990s ', '')
  """
  question = "When did Beyonce become famous?"
  context = "Beyonce started singing as a child but became famous in the 1990s"
  #context = "Beyonce became famous in 1990s and then went onto selling many records"
  answer = ""
  
  print()
  print("QUESTION: ", question)
  print("CONTEXT: ", context)
  print("ANSWER: ", answer)

  """
  
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

  max_start_logits_idx = torch.argmax(output["start_logits"])
  max_end_logits_idx = torch.argmax(output["end_logits"])
  answer_tokens = all_tokens[max_start_logits_idx : max_end_logits_idx + 1]

  # 6/2/24 DH: "RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
  #             FROM: start_logits_dump = np.array(output['start_logits'])
  start_logits_dump = output['start_logits'].detach().numpy()
  # 6/2/24 DH: [-5:][::-1] MEANS...take an array from end-5, then take an array with reverse order
  start_logits_dump = np.sort(start_logits_dump[0])[-10:][::-1]
  start_logits_dump = [np.round(value, 3) for value in start_logits_dump]

  end_logits_dump = output['end_logits'].detach().numpy()
  end_logits_dump = np.sort(end_logits_dump[0])[-10:][::-1]
  end_logits_dump = [np.round(value, 3) for value in end_logits_dump]

  print()
  print("Transformer LIME'ing")
  print("--------------------")
  print(f"start_logits max: {max_start_logits_idx}, value: {output['start_logits'][0][max_start_logits_idx]}")
  print(f"end_logits max: {max_end_logits_idx}, value: {output['end_logits'][0][max_end_logits_idx]}")
  print()
  print(f"output['start_logits']: {len(output['start_logits'][0])}, {start_logits_dump}")
  print(f"output['end_logits']:   {len(output['end_logits'][0])}, {end_logits_dump}")
  print()
  print(f"all_tokens: {len(all_tokens)}, {all_tokens.__class__}")
  print(tokenizer.decode(tokenizer.convert_tokens_to_ids(all_tokens)))
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


def _mp_fn(index):
  # For xla_spawn (TPUs)
  main()


if __name__ == "__main__":
  main()
