# 15/5/25 DH: Refactor of arguments originally from 'run_qa.py' + 'run_seq2seq_qa.py'
#             that are NOT INCLUDED by 'transformers.training_args'
##############################################################################
#
#                                HugginAPI
#
##############################################################################

# --------------------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """
  # "Path to pretrained model or model identifier from huggingface.co/models"
  model_name_or_path: str = field()

  # "Path to directory to store the pretrained models downloaded from huggingface.co"
  cache_dir: Optional[str] = field(default=None,)

  # "The specific model version to use (can be a branch name, tag name or commit id)."
  model_revision: str = field(default="main",)

  # "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated 
  #  when running `huggingface-cli login` (stored in `~/.huggingface`)."
  token: str = field(default=None,)

  # "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should 
  #  only be set to `True` for repositories you trust and in which you have read the code, as it will execute code 
  #  present on the Hub on your local machine."
  trust_remote_code: bool = field(default=False,)

  # 11/8/24 DH: ALLOWING: "pretrained_model": false
  #   https://json-schema.org/understanding-json-schema/reference/boolean
  #   "Note that in JSON, true and false are lower case, whereas in Python they are capitalized (True and False)."
  pretrained_model: bool = field(default=True, metadata={"help": "Whether to use Pre-trained model for fine-tuning"})


@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training and eval.
  """
  # "The name of the dataset to use (via the datasets library)."
  dataset_name: Optional[str] = field(default=None,)

  # "The configuration name of the dataset to use (via the datasets library)."
  dataset_config_name: Optional[str] = field(default=None,)

  # "The input training data file (a text file)."
  train_file: Optional[str] = field(default=None,)
  
  max_seq_length: int = field(default=384,)
  doc_stride: int = field(default=128,)
# --------------------------------------------------------------------------------------------------------------
