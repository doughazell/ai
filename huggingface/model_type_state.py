# 6/8/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

# 6/8/24 DH: How do we name the model type/state from ${TLD} ? 
#            'config.json'        => architectures, model_type
#            'trainer_state.json' => global_step

import sys, os, json

gConfigJSON       = "config.json"
gTrainerStateJSON = "trainer_state.json"

def getModelArch():
  jsonPath = os.path.join(gTLD, gConfigJSON)

  with open(jsonPath) as read_file:
    configData = json.load(read_file)
    modelArch = configData['architectures'][0]

  return modelArch

def getTrainerState():
  jsonPath = os.path.join(gTLD, gTrainerStateJSON)

  with open(jsonPath) as read_file:
    configData = json.load(read_file)
    trainerState = configData['global_step']

    # 9/8/24 DH:
    # 'configData['trial_params']['data_type']' also exists (BUT CURRENTLY DONE VIA 'gRunJSON')
    try:
      # If 'pretrained_model' key present then it is always 'false' (but explicit from code below)
      # https://json-schema.org/understanding-json-schema/reference/boolean
      #   "Note that in JSON, true and false are lower case, whereas in Python they are capitalized (True and False)."
      pretrained_model = configData['trial_params']['pretrained_model']
      if pretrained_model == False:
        nonPretrained = "NoPretrain"
    except KeyError:
      nonPretrained = None

  return (trainerState, nonPretrained)

def getDataType():
  jsonPath = os.path.join(gTLD, gRunJSON)

  with open(jsonPath) as read_file:
    configData = json.load(read_file)
    
    # 6/8/24 DH: Handle SQUAD + Custom JSON data
    try:
      dataType = configData['train_file']
      dataType = os.path.basename(dataType) # eg "data.json"
    except:
      dataType = configData['dataset_name']

  return dataType

if __name__ == "__main__":
  if len(sys.argv) > 2:
    gTLD = sys.argv[1]
    gRunJSON = sys.argv[2]
  else:
    print("INCORRECT cmd args, need <TLD of model> <Run JSON>")
    exit(1)
  
  modelArch = getModelArch()
  (trainerState, nonPretrained) = getTrainerState()
  dataType = getDataType()

  # The script returns the model type/state via the 'stdout'
  #   eg print("BertForQuestionAnswering-1026-squad")
  if nonPretrained:
    print(f"{modelArch}-{nonPretrained}-{trainerState}-{dataType}")
  else:
    print(f"{modelArch}-{trainerState}-{dataType}")