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
gSelectedNodeFilename = "node287-logits"
# 29/7/24 DH:
gFileNameRounded = "weights-rounded"

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
    else:
      print(f"  NOT COPIED: '{logFile}'")
    
  print()

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
    checkpointing_config.gSelectedNodeFilename = open(f"{weightPath}/{gSelectedNodeFilename}.log", 'w')
    checkpointing_config.gSelectedNodeFilename.write(f"ALL LOGITS FROM NODE 287 IN A 'Bert' LAYER\n")
    checkpointing_config.gSelectedNodeFilename.write(f"------------------------------------------\n")
    checkpointing_config.gSelectedNodeFilename.write(f"\n")

    # 29/7/24 DH: CURRENTLY NOT adding to 'archivePrevLogs(weightPath)'
    checkpointing_config.gRoundedWeightsFile = open(f"{weightPath}/{gFileNameRounded}.log", 'w')
    checkpointing_config.gRoundedWeightsFile.write(f"RECORDED ROUNDED WEIGHT BY NUMBERED EPOCH\n")
    checkpointing_config.gRoundedWeightsFile.write(f"-----------------------------------------\n")
    checkpointing_config.gRoundedWeightsFile.write(f"\n")
  
  else: # non-training run
    archivePrevLogs(weightPath, file=gSelectedNodeFilename)

    checkpointing_config.gSelectedNodeFilename = open(f"{weightPath}/{gSelectedNodeFilename}.log", 'w')
    checkpointing_config.gSelectedNodeFilename.write(f"ALL LOGITS FROM NODE 287 IN A 'Bert' LAYER\n")
    checkpointing_config.gSelectedNodeFilename.write(f"------------------------------------------\n")
    checkpointing_config.gSelectedNodeFilename.write(f"\n")


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
    # 25/5/24 DH: PREV: "if checkpointNum == gCheckpointNum and not checkpointNum != 0" 
    #   (ie...double negative meant always false so exited without advancing checkpoint when "not != 0"...ffs...!!!)
    if checkpointNum == gCheckpointNum and checkpointNum != 0:
      print()
      print(f"  {checkpointNum} is same as {gCheckpointNum}")
      print()
    else:
      checkpointing_config.trainer.save_model()
      checkpointing_config.trainer.save_state()

      sigLogger = logging.getLogger("trainer_log")
      sigLogger.info(f"Saving checkpoint: {checkpointNum}")  

      sys.exit(0)
