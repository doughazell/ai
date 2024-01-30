##############################################################
# 30/1/24 DH: Class to store objects from retraining a model
##############################################################

from dataclasses import dataclass, field

import transformers
from transformers import (
    BartConfig,
    BartTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from code_translator_trainer_arguments import DataTrainingArguments, ModelArguments

@dataclass
class Seq2SeqModelData:
  # Name of model and dataset
  # -------------------------
  model_name_or_path: str = field(
    default=None
  )
  dataset_name: str = field(
    default=None
  )
  dataset_config: str = field(
    default=None
  )

  # Model, config, tokenizer, trainer (TODO: Make this a Union of supported model types)
  # ------------------------------------------------------------------------------------
  bart_config: BartConfig = field(
    default=None
  )
  bart_tokenizer_fast: BartTokenizerFast = field(
    default=None
  )
  bart_for_conditional_generation: BartForConditionalGeneration = field(
    default=None
  )
  data_collator_for_seq2seq: DataCollatorForSeq2Seq = field(
    default=None
  )
  seq2seq_trainer: Seq2SeqTrainer = field(
    default=None
  )

  # Training arguments
  # ------------------
  model_arguments: ModelArguments = field(
    default=None
  )
  data_training_arguments: DataTrainingArguments = field(
    default=None
  )
  seq2seq_training_arguments: Seq2SeqTrainingArguments = field(
    default=None
  )



