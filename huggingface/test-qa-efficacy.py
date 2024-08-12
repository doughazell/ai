# 26/5/24 DH: Refactor of 'qa.py' to test several random SQuAD_1 entries
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import logging
logger = logging.getLogger(__name__)

import sys, os, random, time
from datetime import datetime
from pathlib import Path

print("Importing 'transformers'")
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
print("Importing 'datasets'")
import datasets
from datasets import load_dataset

# 27/2/24 DH:
from qa_lime import *
# 19/6/24 DH:
import qa_lime_config

# 30/3/24 DH:
from checkpointing import *
# 18/6/24 DH: Added to id file of 'createLoggers(...)'
import checkpointing

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

# 27/5/24 DH:
def runRandSamples(dataOrigin, raw_datasets, data_args, model_args, iterations=3):
  answerDictDict = {}

  # Re-randomize seeded generator (previously de-randomized in 'set_seed(training_args.seed)' above)
  random.seed(time.time())

  numSamples           = raw_datasets['train'].num_rows
  numValidationSamples = raw_datasets['validation'].num_rows

  print()
  print("                         RUNNING RANDOM SAMPLES")
  print("---------------------------------------------------------------------------")
  print(f"'{dataOrigin}' has '{numSamples}' samples")
  print(f"  (+ '{numValidationSamples}' validation samples giving total of '{numValidationSamples + numSamples}' in '{dataOrigin}')")
  print("---------------------------------------------------------------------------")

  # 12/8/24 DH: Reset file every time 'test-qa-efficacy.py' is called
  idxLogFilename = "dataset-idxs.log"
  idxLog = open(idxLogFilename, "w")

  # -------------------------------------------
  # Loop for specified model non-training runs
  # -------------------------------------------

  for idx in range(iterations):

    # https://en.wikipedia.org/wiki/Division_(mathematics)#Of_integers
    # "Give the integer quotient as the answer, so 26 / 11 = 2. It is sometimes called 'integer division', and denoted by '//'."
    datasetsIdx = int( (random.random() * numSamples * numSamples) // numSamples )

    raw_data = raw_datasets["train"][datasetsIdx]
    if idx + 1 == iterations:
      # 11/8/24 DH: Record 'datasetsIdx' in 'dataset-idxs.log' to be picked up by 'get-model-ouput' bash script to populate 'stack_trace.db'
      idxLog.write(f"{datasetsIdx}")

      print()
      # 11/6/24 DH: See 'qa_lime.py::graphTokenVals(...)' re NO LONGER SENDING 'lastGraph=True'
      #print(f"  Graph {idx+1} of {iterations} with IDX: '{datasetsIdx}' and last graph so sending 'True' to 'plt.show(block=True)'")
      print(f"*** Graph {idx+1} of {iterations} with IDX: '{datasetsIdx}' ***")

      ansDict = {}
      # 'lastGraph=True' legacy
      #(ansDict['question'], ansDict['expAnswer'], ansDict['answer']) = getModelOutput(raw_data, data_args, model_args, printOut=False, lastGraph=True)

      (ansDict['tokenLen'], ansDict['question'], ansDict['context'], ansDict['expAnswer'], 
       ansDict['answer'], ansDict['startIdx'], ansDict['endIdx']) = getModelOutput(raw_data, data_args, model_args, printOut=False)
      print(f"*** END Graph {idx+1} ***")
    else:
      # 11/8/24 DH: Record 'datasetsIdx' in 'dataset-idxs.log' to be picked up by 'get-model-ouput' bash script to populate 'stack_trace.db'
      idxLog.write(f"{datasetsIdx},")

      print()
      print(f"*** Graph {idx+1} of {iterations} with IDX: '{datasetsIdx}' ***")

      ansDict = {}
      #return (question, expAnswer, answer, startIdx, endIdx)
      (ansDict['tokenLen'], ansDict['question'], ansDict['context'], ansDict['expAnswer'],
       ansDict['answer'], ansDict['startIdx'], ansDict['endIdx']) = getModelOutput(raw_data, data_args, model_args, printOut=False)
      print(f"*** END Graph {idx+1} ***")
    
    answerDictDict[idx+1] = ansDict
  # END --- "for idx in range(iterations)" ---

  # 12/8/24 DH: Close file opened at start of function
  idxLog.close()

  return answerDictDict

# 2/8/24 DH: Create log for Custom JSON that does NOT provide correct answer (for testing QA fine-tuning non-Pretrained weights)
# https://discuss.huggingface.co/t/different-results-predicting-from-trainer-and-model/12922/5
def createErrorLogFile(answerDict, training_args):
  print("CREATING: '~/gv-graphs/qca.log' for incorrect Custom JSON data run")
  outFile = createLogFile(training_args)

  outFile.write(f"1-{answerDict['tokenLen']}: Q={answerDict['question']}\n")
  outFile.write(f"1-{answerDict['tokenLen']}: C={answerDict['context']}\n")
  outFile.write(f"1-{answerDict['tokenLen']}: A={answerDict['answer']}\n")
  outFile.write(f"1-{answerDict['tokenLen']}: START INDEX={answerDict['startIdx']}\n")
  outFile.write(f"1-{answerDict['tokenLen']}: END INDEX={answerDict['endIdx']}\n")
  outFile.write(f"1-{answerDict['tokenLen']}: TOKEN LEN={answerDict['tokenLen']}\n")
  outFile.write("\n")

  outFile.close()  

# 18/6/24 DH: Create (+ archive old versions) like 'checkpointing.py::archivePrevLogs(...)'
def createLogFile(training_args):

  # Question-Context-Answer
  """
  # 7/7/24 DH: Path refactor work :
  if qa_lime_config.gTLD:
    qcaFile = os.path.join(qa_lime_config.gTLD, "gv-graphs", "qca.log")
  else:
    qcaFile = os.path.join(training_args.output_dir, "gv-graphs", "qca.log")
  """
  # 8/7/24 DH: Get absolute path so that it can be run from any dir (not just dir containing Huggingface JSON)

  cwd = Path.cwd()

  qcaFile = os.path.join(cwd, "gv-graphs", "qca.log")
  print()
  print(f"  QCA file: '{qcaFile}'")

  print()
  if os.path.isfile(qcaFile):
    # https://strftime.org/
    today = datetime.today().strftime('%-d%b%-H%-M')
    qcaFileDated = f"{qcaFile}{today}"

    shutil.copy(qcaFile, qcaFileDated)

    print(f"  COPIED: '{qcaFile}' to '{qcaFileDated}'")
  else:
    print(f"  NOT COPIED: '{qcaFile}'")
  
  outFile = open(qcaFile, "w")
  return outFile

# 17/6/24 DH: Need to log correct actual answer sets by {key}-Token Len (in order to add details to 'qa-output-<date>.gv.pdf')
def displayResults(answerDictDict, training_args, iterations):
  qcaFileOpen = False
  # 5/8/24 DH: Getting stats of efficacy of current model state
  numCorrect = 0

  for key in answerDictDict:
    print()
    print(f"{key}) Question: {answerDictDict[key]['question']}")
    print(f"   Context: {answerDictDict[key]['context']}")
    print()
    print(f"   Expected answer: {answerDictDict[key]['expAnswer']}")
    print(f"   Actual answer: {answerDictDict[key]['answer']}")

    # Additional output for correct answers
    if answerDictDict[key]['expAnswer'].casefold() == answerDictDict[key]['answer'].casefold():
      print(f"   Token length: {answerDictDict[key]['tokenLen']}")
      descStr = "Answer start index"
      print(f"{descStr:>21}: {answerDictDict[key]['startIdx']}")
      descStr = "Answer end index"
      print(f"{descStr:>21}: {answerDictDict[key]['endIdx']}")

      # If it already exists from a previous key then just append it
      if not qcaFileOpen:
        outFile = createLogFile(training_args)
        qcaFileOpen = True
        
      outFile.write(f"{key}-{answerDictDict[key]['tokenLen']}: Q={answerDictDict[key]['question']}\n")
      outFile.write(f"{key}-{answerDictDict[key]['tokenLen']}: C={answerDictDict[key]['context']}\n")
      outFile.write(f"{key}-{answerDictDict[key]['tokenLen']}: A={answerDictDict[key]['answer']}\n")
      outFile.write(f"{key}-{answerDictDict[key]['tokenLen']}: START INDEX={answerDictDict[key]['startIdx']}\n")
      outFile.write(f"{key}-{answerDictDict[key]['tokenLen']}: END INDEX={answerDictDict[key]['endIdx']}\n")
      outFile.write(f"{key}-{answerDictDict[key]['tokenLen']}: TOKEN LEN={answerDictDict[key]['tokenLen']}\n")
      outFile.write("\n")

      numCorrect += 1
    # END --- <Expected answer == Actual answer> ---
    
    print()
  # END: ------ "for key in answerDictDict" ------

  try: # Handle case when not open due to no correct answers
    outFile.close() # since not using "with ..."

    # 5/8/24 DH: This may need a timestamp since it gets overwritten
    with open(gCorrectLogFilename, "w") as logFile:
      logLine1 = f"Number correct: {numCorrect}"
      logLine2 = f"From: {iterations}"

      logFile.write(logLine1 + "\n")
      logFile.write(logLine2 + "\n")

    return True
  except UnboundLocalError:
    print()
    print("No correct answers found")
    print()

    # 2/8/24 DH: This will leave previous "~/gv-graphs/qca.log" (and may cause INCORRECT 'qa-output-<date>.gv.pdf')
    ansKeyList = list(answerDictDict.keys())
    if len(ansKeyList) == 1: # 'main()::if data_args.train_file' ie Custom JSON
      # Have a no correct answers "qca.log" for testing QA fine-tuning of non-Pretrained weights
      createErrorLogFile(answerDictDict[ansKeyList[0]], training_args)
    
    return False

# 7/7/24 DH:
def getCmdArgs():
  if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    jsonFile = os.path.abspath(sys.argv[1])
  else:
    print("You need to provide a JSON config")
    exit(1)
  
  # 19/6/24 DH: "show" cmd line arg (like 'graph-weights.py') is needed in 'qa_lime.py::graphTokenVals(...)'
  if len(sys.argv) > 2 and "show" in sys.argv[2]:
    qa_lime_config.gShowFlag = True

  #if os.path.isfile("tld.txt"):
  try:
    tldFile = "tld.txt"
    with open(tldFile) as source:
      textLines = [line.strip() for line in source.readlines() if line.strip()]
      qa_lime_config.gTLD = textLines[0]
      print()
      print(f"TLD: {qa_lime_config.gTLD}")
  except FileNotFoundError as e:
    print()
    print(f"{tldFile} not found")

  return jsonFile

def main():
  # 7/7/24 DH:
  jsonFile = getCmdArgs()

  print()
  print(f"Parsing: {jsonFile}")
  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
  model_args, data_args, training_args = parser.parse_json_file(json_file=jsonFile)  

  # ----------------------------- LOGGING -------------------------------------------
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
  # ---------------------------------------------------------------------------------

  print("  'test-qa-efficacy::main()' - NOT USING CHECKPOINT")
  print()

  """
  # 29/5/24 DH: For a non-training run then we want to use the most trained version ie 'model.safetensors'
  #             (which means the training epochs used are: 'trainer_state.json::global_step')
  #
  # Detecting last checkpoint.
  last_checkpoint = None
  if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        
      # 23/3/24 DH: Copying 'run_qa.py' alterations
      print(f"Output directory ({training_args.output_dir}) already exists and is not empty BUT NOT RAISING ValueError...!")
        
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
      logger.info(
        f"Checkpoint detected, resuming training at {last_checkpoint}."
      )
  """

  # 3/3/24 DH:  PREVIOUSLY NO NEED TO CREATE LOGGERS FOR NON-TRAINING
  # 30/3/24 DH: NOW, THE INPUT_IDS & LOGITS ARE LOGGED
  # 30/3/24 DH: Needs to be after "Detecting last checkpoint" in order to create the first checkpoint with Ctrl-C 
  #             (and prevent the need for "run/remove/rerun" involving, ' "overwrite_output_dir": "True" ')
  # 25/5/24 DH: 'run_qa.py' now logs weights in 'weights-full.log' + 'weights.log' so need to prevent 'qa.py' from overwriting
  checkpointing.createLoggers(training_args, overwrite=False)

  # Set seed before initializing model.
  # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py#L86
  print()
  print("************************************")
  print(f"CALLING: trainer_utils::set_seed({training_args.seed})")
  print("************************************")
  print()
  set_seed(training_args.seed)

  # ------------------------------------- LOAD DATASETS ----------------------------------
  if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    print()
    print(f"'load_dataset({data_args.dataset_name})'")
    print()
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

    print()
    print(f"'load_dataset({data_args.train_file})'")
    print()
    raw_datasets = load_dataset(
      extension,
      data_files=data_files,
      # 27/5/24 DH: Solving "KeyError: 'data'" for JSON datasets ONLY WHEN "Generating dataset json" (copy 'run_qa.py')
      #             (see "2024/b3-feb12")
      #field="data",
      cache_dir=model_args.cache_dir,
      token=model_args.token,
    )
  # --------------------------------------------------------------------------------------

  #######################################################################################################
  # 28/5/24 DH: TODO: Correlate epoch training number, loss, actual-expected answer in non-training run
  #                   (add to SQLite DB: epoch, loss, SQuAD#, success/fail)
  #######################################################################################################

  # 27/5/24 DH: HARD-CODED to use FIRST SAMPLE of JSON datasets (which is the JSON list)
  if data_args.train_file:
    datasetsIdx = 0

    #printDatasetInfo(raw_datasets, datasetsIdx)
    raw_data = raw_datasets["train"][datasetsIdx]
    # 12/6/24 DH: Adding in user feedback for JSON data run
    ansDict = {}
    (ansDict['tokenLen'], ansDict['question'], ansDict['context'], ansDict['expAnswer'], 
     ansDict['answer'], ansDict['startIdx'], ansDict['endIdx']) = getModelOutput(raw_data, data_args, model_args, printOut=False)
    answerDictDict = {}
    answerDictDict[1] = ansDict

    # 6/8/24 DH: Update Custom JSON to use stats system developed for SQUAD
    iterations = 1

  # BUT random sample of Arrow Datasets (First entry is 0)
  elif data_args.dataset_name:
    iterations = 3
    answerDictDict = runRandSamples(data_args.dataset_name, raw_datasets, data_args, model_args, iterations)

  someCorrectFlag = displayResults(answerDictDict, training_args, iterations)
  
  if qa_lime_config.gShowFlag == True:
    print()
    print("PRESS RETURN TO FINISH", end='')
    response = input()
  
  return someCorrectFlag

if __name__ == "__main__":

  # main() :
  # "sys.argv[1]" used for JSON
  # "sys.argv[2]" used for 'show'
  # "tld.txt" stores path of HFDIR
  someCorrectFlag = main()
  if someCorrectFlag:
    # https://docs.python.org/3/library/sys.html#sys.exit
    # "zero is considered “successful termination” and any nonzero value is considered “abnormal termination” by shells"
    sys.exit(0)
  else:
    sys.exit(1)
