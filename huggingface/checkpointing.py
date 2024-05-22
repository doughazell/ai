# 24/2/24 DH:

import sys, os, logging
from pathlib import Path
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

import checkpointing_config

gStoppingFlag = False
gCheckpointNum = 0

# 9/2/24 DH:
def createLoggers(training_args):
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

  fileHandler = logging.FileHandler(f"{logPath}/{fileName}.log", mode="w")
  logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
  fileHandler.setFormatter(logFormatter)
  sigLogger.addHandler(fileHandler)
  sigLogger.info(f"PID: {os.getpid()}")

  # -----------------------------------------------------------------------------
  # 14/5/24 DH: Added to record the rounded weight percentage diffs (for later graphing)
  fileName = "weights"
  # 21/5/24 DH: Probably would have been easier to assign: 
  #             'checkpointing_config.logPath = training_args.output_dir'
  #   (and do everything else locally to where needed, since this is having feature-creep...!)
  logPath = training_args.output_dir
  graphDir = "graphs"
  fileNameFull = "weights-full"

  checkpointing_config.gWeightsFile = open(f"{logPath}/{fileName}.log", 'w')
  # 21/5/24 DH:
  checkpointing_config.gWeightsFile.write(f"RECORDED WEIGHT PERCENTAGE DIFFS BY NUMBERED EPOCH\n")
  checkpointing_config.gWeightsFile.write(f"--------------------------------------------------\n")
  checkpointing_config.gWeightsFile.write(f"\n")

  # 21/5/24 DH:
  checkpointing_config.gGraphDir = f"{logPath}/{graphDir}"
  Path(checkpointing_config.gGraphDir).mkdir(parents=True, exist_ok=True)

  # 21/5/24 DH:
  checkpointing_config.gFullWeightsFile = open(f"{logPath}/{fileNameFull}.log", 'w')
  checkpointing_config.gFullWeightsFile.write(f"RECORDED FULL WEIGHT BY NUMBERED EPOCH\n")
  checkpointing_config.gFullWeightsFile.write(f"--------------------------------------\n")
  checkpointing_config.gFullWeightsFile.write(f"\n")

def getHighestCheckpoint():
  
  lastCheckpointPath = get_last_checkpoint(checkpointing_config.training_args.output_dir)

  # 13/2/24 DH: Accomodate when there is no checkpoint already (knock-3-times-and-ask-for-Alan...)
  checkpointNum = 0
  if lastCheckpointPath:
    lastCheckpoint = os.path.basename(lastCheckpointPath)
    checkpointSplit = lastCheckpoint.split('-')
    checkpointNum = int(checkpointSplit[1])
  
  return checkpointNum

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
    if checkpointNum == gCheckpointNum and not checkpointNum != 0:
      print()
      print(f"  {checkpointNum} is same as {gCheckpointNum}")
      print()
    else:
      checkpointing_config.trainer.save_model()
      checkpointing_config.trainer.save_state()

      sigLogger = logging.getLogger("trainer_log")
      sigLogger.info(f"Saving checkpoint: {checkpointNum}")  

      sys.exit(0)
