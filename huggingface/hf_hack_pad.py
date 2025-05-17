# 14/5/25 DH: Scratch-pad for testing HuggingFace technology
##############################################################################
#
#                                HugginAPI
#
##############################################################################
import sys, os

print("Importing 'transformers'")
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

from hf_arguments import *
from datasets import load_dataset

# 15/5/25 DH: CREATED WITH: "huggingface$ ln -s test-qa-efficacy.py test_qa_efficacy.py"
# (This may need to be reviewed in due course...:) )
print()
print("Importing 'test_qa_efficacy'")
print("----------------------------")
import test_qa_efficacy
print("<imported 'test_qa_efficacy'>")
print("-----------------------------")
print()
import checkpointing

# --------------------------------------------------------------------------------------------------------------
def initHF_env(jsonFile):
  print()
  print(f"Parsing: {jsonFile}")
  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
  model_args, data_args, training_args = parser.parse_json_file(json_file=jsonFile)

  # FROM 'test-qa-efficacy.py::main()'
  # ----------------------------------
  # 30/3/24 DH: Needs to be after "Detecting last checkpoint" in order to create the first checkpoint with Ctrl-C 
  #             (and prevent the need for "run/remove/rerun" involving, ' "overwrite_output_dir": "True" ')
  # 25/5/24 DH: 'run_qa.py' now logs weights in 'weights-full.log' + 'weights.log' so need to prevent 'qa.py' from overwriting
  #
  # ADDITION FOR 'hf_hack_pad.py'
  # -----------------------------
  # 16/5/25 DH: 'checkpointing_config.gGraphDir' is needed by 'qa_lime.py::graphTokenVals(...)' which is init'd to "None"
  checkpointing.createLoggers(training_args, overwrite=False)

  # Set seed before initializing model.
  # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L86
  print()
  print("************************************")
  print(f"CALLING: trainer_utils::set_seed({training_args.seed})")
  print("************************************")
  print()
  set_seed(training_args.seed)

  # 15/5/25 DH: FROM: 'test-qa-efficacy.py'
  if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    print()
    print(f"'load_dataset({data_args.dataset_name})'")
    print()
    raw_datasets = load_dataset(
      data_args.dataset_name, # eg 'squad'

      # 15/8/24 DH: All remaining args are BLANK
      # ----------------------------------------
      data_args.dataset_config_name,
      cache_dir=model_args.cache_dir,
      token=model_args.token,
    )
  
  return (model_args, data_args, training_args, raw_datasets)

def runRandSamples(model_args, training_args, data_args, raw_datasets):
  # 15/5/25 DH: FROM: 'test-qa-efficacy.py'
  if data_args.dataset_name is not None:
    iterations = 3

    # 15/5/25 DH: Adding 'try...except EnvironmentError'
    try:
      answerDictDict = test_qa_efficacy.runRandSamples(data_args.dataset_name, raw_datasets, data_args, model_args, iterations)
      someCorrectFlag = test_qa_efficacy.displayResults(answerDictDict, training_args, iterations)
    except EnvironmentError as e:
      print()
      print(f"ERROR: {e}")
      print()
      print("(Probably because you are running this script from a directory that does not contain a valid checkpoint/model directory)")
      print("(SEE: https://github.com/doughazell/ai/blob/main/README.md for more details)")
      print()

# 15/5/25 DH: This is a refactor of the code used to run a trained Q&A model that led onto 'get-model-output'
#   IT STARTED with https://github.com/doughazell/ai/blob/main/bart/dataset_qa.py#L780 to test simpler JSON Grammar
#     (COPY&PASTE TO BROWSER: file:///Users/doug/Desktop/devlogeD/2024/doc/b3-feb12.html)
#   THEN https://github.com/doughazell/ai/commits/main/bart/seq2seq_qa_utils.py 
#   TO 'qa_lime.py'
def runBertSQUAD():
  CFGDIR = "/Users/doug/src/ai/t5"
  JSONCFG = "qa_train-Google-BERT-SQUAD.json"

  print("USING:")
  print("------")
  print(f"  CFGDIR: {CFGDIR}")
  print(f"  JSONCFG: {JSONCFG}")

  # FROM: get-model-output::getCorrectAnswer
  #   python ${SCRIPTDIR}/test-qa-efficacy.py ${CFGDIR}/${JSONCFG}

  jsonFile = os.path.abspath(f"{CFGDIR}/{JSONCFG}")

  (model_args, data_args, training_args, raw_datasets) = initHF_env(jsonFile)
  print("------")
  
  runRandSamples(model_args, training_args, data_args, raw_datasets)


if __name__ == "__main__":
  runBertSQUAD()