# 11/2/24 DH:

import os, sys, signal, time
import transformers
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

# 2/3/24 DH:
import psutil

# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Arguments:
  output_dir: str = field(
    metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
  )
  model_name_or_path: str = field(
    metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  trainer_log: str = field(
    default="seq2seq_qa_INtrainer.log"
  )
  dataset_name: Optional[str] = field(
    default=None
  )
# ---------------------------------------------------------------------------------------------------------------

scriptDir = os.path.dirname(os.path.realpath(__file__))

# 2/3/24 DH:
def parseTrainerStack(stackFile):

  with open(stackFile) as source :
    print()
    print("Filename: ", stackFile)
    print("---------")

    textLines = [line.strip() for line in source.readlines() if line.strip()]
  
  for line in textLines:
    if "File" in line:

      lineSplit = line.split("site-packages/")
      if len(lineSplit) > 1:
        linePart = lineSplit[1]
      else:
        linePart = lineSplit[0]
        lineSplit = linePart.split("File \"")
        if len(lineSplit) > 1:
          linePart = lineSplit[1]
      
      lineSplit = linePart.split("\", ")
      fileName = lineSplit[0]
      otherParts = lineSplit[1]

      lineSplit = otherParts.split(", in ")
      lineNum = lineSplit[0].lstrip("line ")
      funcName = lineSplit[1]

      print(f"FILE: {fileName:50} LINE: {lineNum:7} FUNCTION: {funcName}")


# 2/3/24 DH: Refactor to use a library rather than cmd line script
def sigintPIDFromTrainerLog(scriptDir, waitFlag=True):
  parser = HfArgumentParser((Arguments))
  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    jsonFile = os.path.abspath(sys.argv[1])
  else:
    jsonFile = os.path.join(scriptDir, "qa_train.json")

  # 11/2/24 DH: Need ',' after 'args' in order to parse tuple response (the "()" are not required but make the tuple clearer)
  (args,) = parser.parse_json_file(json_file=jsonFile, allow_extra_keys=True)

  print()
  print()
  print("sigintPIDFromTrainerLog():")
  print(f"  {args}")

  filename = os.path.join(scriptDir, args.output_dir, args.trainer_log)
  with open(filename) as source :  
    print()
    print("Filename: ", filename)

    textLines = [line.strip() for line in source.readlines() if line.strip()]

  pid = -1
  for line in textLines:
    if "PID" in line:
      print(line)
      pid = int(line.split("PID: ")[1])
      break

  try:
    print(f"Sending {signal.SIGINT} to {pid}")
    iCnt = 1

    os.kill(pid, signal.SIGINT)

    # 2/3/24 DH: Last line of "KeyboardInterrupt" is checked in calling function to ensure sufficient
    #            time has elapsed to get stack trace
    if waitFlag:
      numSecs = 10
      print(f"Sleeping for {numSecs} secs")
      time.sleep(numSecs)

    while waitFlag:
      print(f"  Sending {signal.SIGINT} to {pid}")
      iCnt += 1

      # 2/3/24 DH: When trainer started with 'subprocess.Popen()' then KeyboardInterrupt handled as usual 
      #   but PID is UN-KILLABLE so 'waitFlag=False' returns to subprocess.Popen() script for 'proc.kill()'
      os.kill(pid, signal.SIGINT)

      #retVal = psutil.pid_exists(pid)

      numSecs = 2
      print(f"  {iCnt}) Sleeping for {numSecs} secs")
      print()
      time.sleep(numSecs)

  except ProcessLookupError as e:
    print(f"Process {pid} is cancelled")

if __name__ == "__main__":
  
  sigintPIDFromTrainerLog(scriptDir)
