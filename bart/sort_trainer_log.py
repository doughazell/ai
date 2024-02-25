# 11/2/24 DH:

import os, sys
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
    default="seq2seq_qa_trainer.log"
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
  jsonFile = os.path.join(scriptDir, "qa_train-Google-T5-small.json")

# 11/2/24 DH: Need ',' after 'args' in order to parse tuple response (the "()" are not required but make the tuple clearer)
(args,) = parser.parse_json_file(json_file=jsonFile, allow_extra_keys=True)

print(args)

filename = os.path.join(scriptDir, args.output_dir, args.trainer_log)
with open(filename) as source :  
  print()
  print(f"Filename: {filename} :")

  textLines = [line.strip() for line in source.readlines() if line.strip()]

checkpointNum = 0
checkpointDict = {"largeDiff": 1000, "datetimes": [], "checkpoints": []}

for line in textLines:
  lineSplit = line.split("Saving checkpoint: ")

  datetime = lineSplit[0].split(',')[0]
  checkpoint = int(lineSplit[1])
  
  if checkpoint > checkpointNum + checkpointDict["largeDiff"]:
    print(f"  {checkpoint:>5} from {checkpointNum:>5} ({checkpoint - checkpointNum}) at {datetime}")

    checkpointDict["checkpoints"].append(checkpoint)
    checkpointDict["datetimes"].append(datetime)

  checkpointNum = checkpoint

print()
for key in checkpointDict:
  print(f"{key}: {checkpointDict[key]}")
print()

