# 24/2/24 DH:

import sys, os, logging, shutil
from datetime import datetime
from pathlib import Path

print("'checkpointing.py' importing 'Trainer'...")
from transformers import Trainer
print()
print("...done")
from transformers.trainer_utils import get_last_checkpoint

import checkpointing_config

gStoppingFlag = False
gCheckpointNum = 0

# 6/6/24 DH:
gFileName = "weights"  
gFileNameFull = "weights-full"
gLossFilename = "loss-by-epochs"

# 8/6/24 DH:
#gSelectedNodeFilename = "node287-logits"
# 12/9/24 DH: Node to track gets selected by 'graph-weights.py' + IPC with "max-node.txt" via 'get-model-output' BASH script
gSelectedNodeFilename = "node-logits"
gMaxNodeFilename = "max-node.txt"

# 29/7/24 DH:
gFileNameRounded = "weights-rounded"
# 5/8/24 DH: Get % correct stats for 'test-qa-efficacy.py'
gCorrectLogFilename = "correct-answers.log"

# 6/6/24 DH:
def archivePrevLogs(weightPath, file=False):
  global gFileName
  global gFileNameFull
  global gLossFilename
  global gSelectedNodeFilename

  if file:
    files = [file]
  else:
    files = [gFileName, gFileNameFull, gLossFilename, gSelectedNodeFilename]

  print()
  print("Archiving previous logs")
  print("-----------------------")

  for logFile in files:
    logFile = f"{weightPath}/{logFile}.log"
    if os.path.isfile(logFile):
      # https://strftime.org/
      today = datetime.today().strftime('%-d%b%-H%-M')
      logFileDated = f"{logFile}{today}"

      shutil.copy(logFile, logFileDated)

      print(f"  COPIED: '{logFile}' to '{logFileDated}'")

      # 14/8/24 DH: 'graph-weights.py' copies "weights-full.log" to "weights-full-0.log" if it contains epoch 0
      #             This is then used for graphing % change during MID-SECTION TRAINING DUE TO CHECKPOINT

    else:
      print(f"  NOT COPIED: '{logFile}'")
    
  print()

# 12/9/24 DH:
def getTrackedNode():
  trackedNode = 0

  try:
    # FROM: 'create-gv-training.py'
    #lossesFilenameFile = f"{gCWD}/gv-graphs/losses_filename.txt"

    with open(gMaxNodeFilename) as inFile:
      trackedNode = inFile.readline()

  except FileNotFoundError:
    print()
    print(f"Filename: {gMaxNodeFilename} NOT FOUND")
    print("...exiting")
    print()
    sys.exit(69) # User defined return value

  return trackedNode

# 9/2/24 DH:
def createLoggers(training_args, overwrite=True):
  # 8/2/24 DH: https://docs.python.org/3/howto/logging.html
  #            https://docs.python.org/3/library/logging.html

  #sigLogger = logging.getLogger(__name__)
  
  # -----------------------------------------------------------------------------
  sigLogger = logging.getLogger("trainer_log")
  sigLogger.setLevel(logging.DEBUG)
  fileName = "seq2seq_qa_trainer"
  logPath = training_args.output_dir

  # 15/2/24 DH: Taken from 'trainer.py:Trainer._save()'
  os.makedirs(training_args.output_dir, exist_ok=True)

  # 25/5/24 DH: Opened in normal logging append mode (so NOT overwritten)
  fileHandler = logging.FileHandler(f"{logPath}/{fileName}.log")
  logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  fileHandler.setFormatter(logFormatter)
  # Need to remove default console handler setup in main() script, 'logging.basicConfig(..., handlers=[logging.StreamHandler(sys.stdout)] )
  sigLogger.addHandler(fileHandler)

  # -----------------------------------------------------------------------------
  sigLogger = logging.getLogger("trainer_signaller")
  sigLogger.setLevel(logging.DEBUG)
  fileName = "seq2seq_qa_INtrainer"
  logPath = training_args.output_dir

  # 25/5/24 DH: mode="w" so OVERWRITTEN
  fileHandler = logging.FileHandler(f"{logPath}/{fileName}.log", mode="w")
  logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  fileHandler.setFormatter(logFormatter)
  sigLogger.addHandler(fileHandler)
  sigLogger.info(f"PID: {os.getpid()}")

  # -----------------------------------------------------------------------------

  # 21/5/24 DH: Probably would have been easier to assign: 
  #             'checkpointing_config.logPath = training_args.output_dir'
  #   (and do everything else locally to where needed, since this is having feature-creep...!)
  logPath = training_args.output_dir
  graphDir = "graphs"
  checkpointing_config.gGraphDir = f"{logPath}/{graphDir}"
  # 8/6/24 DH:
  weightPath = f"{logPath}/weights"

  # 6/6/24 DH: Now global to archive existing logs with date
  global gFileName
  global gFileNameFull
  global gLossFilename
  # 8/6/24 DH:
  global gSelectedNodeFilename
  # 29/7/24 DH:
  global gFileNameRounded

  # 21/5/24 DH: https://docs.python.org/3/library/pathlib.html#concrete-paths
  Path(checkpointing_config.gGraphDir).mkdir(parents=True, exist_ok=True)

  cwd = Path.cwd()
  print(f"CREATED: '{checkpointing_config.gGraphDir}' IN: '{cwd}'")

  # 8/6/24 DH: Not needed outside this file so not in 'checkpointing_config' 
  #        (unlike 'gGraphDir': plt.savefig(f"{checkpointing_config.gGraphDir}/{graphFilename}.png"))
  Path(weightPath).mkdir(parents=True, exist_ok=True)
  print(f"CREATED: '{weightPath}' IN: '{cwd}'")

  # 12/9/24 DH: Track correct Node based on measured change in 'graph_weights.py::gNodeIDfilename = "max-node.txt"'
  trackedNode = getTrackedNode()
  checkpointing_config.gTrackedNode = int(trackedNode)

  # 25/5/24 DH: Prevent 'qa.py' overwriting log files from 'run_qa.py'
  if overwrite:
    # 5/6/24 DH: Save prev files + date before opening for writing (and hence overwriting the contents)
    archivePrevLogs(weightPath)

    # 9/6/24 DH: Changing path of ALL WEIGHT LOGS from 'logPath' to 'weightPath'
    checkpointing_config.gWeightsFile = open(f"{weightPath}/{gFileName}.log", 'w')
    # 21/5/24 DH:
    checkpointing_config.gWeightsFile.write(f"RECORDED WEIGHT PERCENTAGE DIFFS BY NUMBERED EPOCH\n")
    checkpointing_config.gWeightsFile.write(f"--------------------------------------------------\n")
    checkpointing_config.gWeightsFile.write(f"\n")

    # 21/5/24 DH:
    checkpointing_config.gFullWeightsFile = open(f"{weightPath}/{gFileNameFull}.log", 'w')
    checkpointing_config.gFullWeightsFile.write(f"RECORDED FULL WEIGHT BY NUMBERED EPOCH\n")
    checkpointing_config.gFullWeightsFile.write(f"--------------------------------------\n")
    checkpointing_config.gFullWeightsFile.write(f"\n")

    # 25/5/24 DH: This needs 'flush()' if ONLY "import checkpointing_config"
    checkpointing_config.gLossFile = open(f"{weightPath}/{gLossFilename}.log", 'w')
    checkpointing_config.gLossFile.write(f"LOSS BY NUMBERED EPOCH\n")
    checkpointing_config.gLossFile.write(f"----------------------\n")
    checkpointing_config.gLossFile.write(f"\n")

    # 8/6/24 DH: Open for both training + non-training runs
    # 12/9/24 DH: Chg 'gSelectedNodeFilename' to 'gSelectedNodeFile' (as distinct from the filename)
    checkpointing_config.gSelectedNodeFile = open(f"{weightPath}/{gSelectedNodeFilename}.log", 'w')
    checkpointing_config.gSelectedNodeFile.write(f"ALL LOGITS FROM NODE {trackedNode} IN A 'Bert' LAYER\n")
    checkpointing_config.gSelectedNodeFile.write(f"------------------------------------------\n")
    checkpointing_config.gSelectedNodeFile.write(f"\n")

    # 29/7/24 DH: CURRENTLY NOT adding to 'archivePrevLogs(weightPath)'
    checkpointing_config.gRoundedWeightsFile = open(f"{weightPath}/{gFileNameRounded}.log", 'w')
    checkpointing_config.gRoundedWeightsFile.write(f"RECORDED ROUNDED WEIGHT BY NUMBERED EPOCH\n")
    checkpointing_config.gRoundedWeightsFile.write(f"-----------------------------------------\n")
    checkpointing_config.gRoundedWeightsFile.write(f"\n")
  
  else: # non-training run
    archivePrevLogs(weightPath, file=gSelectedNodeFilename)

    # 12/9/24 DH: Chg 'gSelectedNodeFilename' to 'gSelectedNodeFile' (as distinct from the filename)
    checkpointing_config.gSelectedNodeFile = open(f"{weightPath}/{gSelectedNodeFilename}.log", 'w')
    checkpointing_config.gSelectedNodeFile.write(f"ALL LOGITS FROM NODE {trackedNode} IN A 'Bert' LAYER\n")
    checkpointing_config.gSelectedNodeFile.write(f"------------------------------------------\n")
    checkpointing_config.gSelectedNodeFile.write(f"\n")
  # END --- "if overwrite" ---

def getHighestCheckpoint():
  
  lastCheckpointPath = get_last_checkpoint(checkpointing_config.training_args.output_dir)

  # 13/2/24 DH: Accomodate when there is no checkpoint already (knock-3-times-and-ask-for-Alan...)
  checkpointNum = 0
  if lastCheckpointPath:
    lastCheckpoint = os.path.basename(lastCheckpointPath)
    checkpointSplit = lastCheckpoint.split('-')
    checkpointNum = int(checkpointSplit[1])
  
  return checkpointNum

# 9/8/24 DH:
#   CONFIG_NAME = "config.json" (https://github.com/huggingface/transformers/blob/main/src/transformers/utils/__init__.py#L239)
#   'PretrainedConfig.save_pretrained(...)' (https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L423)
#   'Trainer._save(...)' (https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3506)
#
#   ACTUALLY PROBABLE BEST to save in: TRAINER_STATE_NAME = "trainer_state.json"
#     "pretrained_model": false,
#     trainer.state => TrainerState(...) (https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L36)

def saveTrialParams():
  # 9/8/24 DH:
  checkpointing_config.trainer.state.trial_params = {}
  
  if checkpointing_config.data_args.dataset_name is not None:
    checkpointing_config.trainer.state.trial_params['data_type'] = checkpointing_config.data_args.dataset_name
  
  if checkpointing_config.data_args.train_file is not None:
    checkpointing_config.trainer.state.trial_params['data_type'] = checkpointing_config.data_args.train_file

  # 9/8/24 DH: Record whether using: 'model = BertForQuestionAnswering() / BertForQuestionAnswering.from_pretrained()'
  if checkpointing_config.pretrained_modelFlag == False: # set if using 'BertForQuestionAnswering()' 
    checkpointing_config.trainer.state.trial_params['pretrained_model'] = False

  checkpointing_config.trainer.save_state()

#######################################################################################################
# Globals needed from Trainer script:
#   1) training_args
#   2) trainer
#######################################################################################################
def signal_handler(sig, frame):
  print('\nYou pressed Ctrl+C so saving checkpoint')

  global gStoppingFlag
  global gCheckpointNum

  # 18/2/24 DH: Running with '"overwrite_output_dir": "True"' + 'checkpoints' will overwrite high checkpoints with restarted number
  #      (may need 'kill -9 <PID>' from 'seq2seq_qa_INtrainer.log' 'PID: 52979' BECAUSE 'checkpointNum == gCheckpointNum')
  #                                  ...misfire drill error'esk if not required...
  if not gStoppingFlag:
    gStoppingFlag = True
  
    gCheckpointNum = getHighestCheckpoint()
    print("Highest checkpoint: ", gCheckpointNum)

    print("SETTING: 'Trainer.save_steps = 2' + 'Trainer.should_save = True'")
    Trainer.save_steps = 2
    Trainer.should_save = True
    
  # 2nd time Ctrl-C clicked
  else:
    checkpointNum = getHighestCheckpoint()
    
    # 13/2/24 DH: Accomodate when there is no checkpoint already (knock-3-times-and-ask-for-Alan...)
    # 25/5/24 DH: PREV: "if checkpointNum == gCheckpointNum and not checkpointNum != 0" 
    #   (ie...double negative meant always false so exited without advancing checkpoint when "not != 0"...ffs...!!!)
    # 9/8/24 DH: Accom HF partial save issue by ensuring that "Trainer.save_steps" has elapsed
    #            (Preventing Ctrl-C'ing the Ctrl-C when saving is not complete to due "Accelerator" distrib scheduling)
    sufficDiff = Trainer.save_steps + 1
    if checkpointNum < gCheckpointNum + sufficDiff and checkpointNum != 0:
      print()
      print(f"  {checkpointNum} is still less than {gCheckpointNum + sufficDiff}")
      print()
    else:
      checkpointing_config.trainer.save_model()
      checkpointing_config.trainer.save_state()

      # 9/8/24 DH: Now saving 'data_type' + 'pretrained_model' in 'trainer_state.json'
      saveTrialParams()

      sigLogger = logging.getLogger("trainer_log")
      sigLogger.info(f"Saving checkpoint: {checkpointNum}")  

      sys.exit(0)
