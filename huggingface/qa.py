#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
from datasets import load_dataset
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# 27/2/24 DH:
from qa_lime import *

# 30/3/24 DH:
from checkpointing import *
import checkpointing_config

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
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
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


def main():
  # See all possible arguments in src/transformers/training_args.py
  # or by passing the --help flag to this script.
  # We now keep distinct sets of args, for a cleaner separation of concerns.

  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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

  # 2/3/24 DH:
  print()
  print("  Calling: 'transformers.utils.logging.set_verbosity_error()'")
  print()
  transformers.utils.logging.set_verbosity_error()

  

  # Log on each process the small summary:
  logger.warning(
      f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
      + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
  )

  print()
  print("  Removing: logger.info(f'Training/evaluation parameters {training_args}')")
  print()
  #logger.info(f"Training/evaluation parameters {training_args}")

  # Detecting last checkpoint.
  last_checkpoint = None
  if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
      last_checkpoint = get_last_checkpoint(training_args.output_dir)
      if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
          
          # 23/3/24 DH: Copying 'run_qa.py' alterations
          print(f"Output directory ({training_args.output_dir}) already exists and is not empty BUT NOT RAISING ValueError...!")
          """
          raise ValueError(
              f"Output directory ({training_args.output_dir}) already exists and is not empty. "
              "Use --overwrite_output_dir to overcome."
          )
          """

      elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
          logger.info(
              f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
              "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
          )

  # 3/3/24 DH:  PREVIOUSLY NO NEED TO CREATE LOGGERS FOR NON-TRAINING
  # 30/3/24 DH: NOW, THE INPUT_IDS & LOGITS ARE LOGGED
  # 30/3/24 DH: Needs to be after "Detecting last checkpoint" in order to create the first checkpoint with Ctrl-C 
  #             (and prevent the need for "run/remove/rerun" involving, ' "overwrite_output_dir": "True" ')
  # 25/5/24 DH: 'run_qa.py' now logs weights in 'weights-full.log' + 'weights.log' so need to prevent 'qa.py' from overwriting
  createLoggers(training_args, overwrite=False)

  # Set seed before initializing model.
  set_seed(training_args.seed)

  # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
  # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
  # (the dataset will be downloaded automatically from the datasets Hub).
  #
  # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
  # 'text' is found. You can easily tweak this behavior (see below).
  #
  # In distributed training, the load_dataset function guarantee that only one local process can concurrently
  # download the dataset.
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

      if data_args.validation_file is not None:
          data_files["validation"] = data_args.validation_file
          extension = data_args.validation_file.split(".")[-1]
      if data_args.test_file is not None:
          data_files["test"] = data_args.test_file
          extension = data_args.test_file.split(".")[-1]
      raw_datasets = load_dataset(
          extension,
          data_files=data_files,
          # 27/5/24 DH: Solving "KeyError: 'data'" for JSON datasets ONLY WHEN "Generating dataset json" (copy 'run_qa.py')
          #             (see "2024/b3-feb12")
          #field="data",
          cache_dir=model_args.cache_dir,
          token=model_args.token,
      )
  # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
  # https://huggingface.co/docs/datasets/loading_datasets.

  # Load pretrained model and tokenizer
  #
  # Distributed training:
  # The .from_pretrained methods guarantee that only one local process can concurrently
  # download model & vocab.
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
      use_fast=True,
      revision=model_args.model_revision,
      token=model_args.token,
      trust_remote_code=model_args.trust_remote_code,
  )
  model = AutoModelForQuestionAnswering.from_pretrained(
      model_args.model_name_or_path,
      from_tf=bool(".ckpt" in model_args.model_name_or_path),
      config=config,
      cache_dir=model_args.cache_dir,
      revision=model_args.model_revision,
      token=model_args.token,
      trust_remote_code=model_args.trust_remote_code,
  )

  # Preprocessing the datasets.
  # Preprocessing is slighlty different for training and evaluation.
  if training_args.do_train:
      column_names = raw_datasets["train"].column_names
  elif training_args.do_eval:
      column_names = raw_datasets["validation"].column_names
  else:
      column_names = raw_datasets["test"].column_names
  
  question_column_name = "question" if "question" in column_names else column_names[0]
  context_column_name = "context" if "context" in column_names else column_names[1]
  answer_column_name = "answers" if "answers" in column_names else column_names[2]

  # Padding side determines if we do (question|context) or (context|question).
  pad_on_right = tokenizer.padding_side == "right"

  # ------------------------------------------ HACK ZONE -----------------------------------------
  print()
  print("Config return from AutoConfig: ", config.__class__)
  print("Tokenizer return from AutoTokenizer: ", tokenizer.__class__)
  print("Model returned from AutoModelForQuestionAnswering: ", model.__class__)
  print()

  print()
  print("------ Now running the trained model for Q&A ------")

  # 27/5/24 DH: Copy changes made in 'test-qa-efficacy.py' refactor
  datasetsIdx = 0
  printDatasetInfo(raw_datasets, datasetsIdx)
  
  raw_data = raw_datasets["train"][datasetsIdx]
  getModelOutput(raw_data, data_args, model_args)


if __name__ == "__main__":

    main()
