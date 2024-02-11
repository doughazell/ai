# 11/2/24 DH:

import os, sys, signal, time
import transformers
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

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

parser = HfArgumentParser((Arguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
  jsonFile = os.path.abspath(sys.argv[1])
else:
  jsonFile = os.path.join(scriptDir, "qa_train.json")

# 11/2/24 DH: Need ',' after 'args' in order to parse tuple response (the "()" are not required but make the tuple clearer)
(args,) = parser.parse_json_file(json_file=jsonFile, allow_extra_keys=True)

print(args)

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
  os.kill(pid, signal.SIGINT)

  numSecs = 10
  print(f"Sleeping for {numSecs} secs")
  time.sleep(numSecs)

  while True:
    print(f"  Sending {signal.SIGINT} to {pid}")
    os.kill(pid, signal.SIGINT)
    numSecs = 2
    print(f"  Sleeping for {numSecs} secs")
    print()
    time.sleep(numSecs)

except ProcessLookupError as e:
  print(f"Process {pid} is cancelled")