# 10/4/23 DH: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

# 24/2/24 DH: Taken from 'mnist-training-errors/gym_config.py'
# This needs to be imported for read/write-access with: "import checkpointing_config"
#   ("from gym_config import *" provided read-access in 'mnist-training-errors/dqn_c51.py')

# ----------------- Runtime Cfg --------------------

trainer = None
training_args = None
# 9/8/24 DH: Used to record 'data_type' in 'trainer_state.json'
data_args = None
pretrained_modelFlag = None

# 14/5/24 DH: Used to save a layer weights (for later graphing)
gWeightsFile = None
# 21/5/24 DH: Directory used to save a weights graph
gGraphDir = None
gFullWeightsFile = None
# 25/5/24 DH: Used to save the loss for each epoch (for later graphing)
gLossFile = None
# 8/6/24 DH: Used to save 'BertSelfAttention' 384 logits for selected Node (currently 287)
# 12/9/24 DH: Chg 'gSelectedNodeFilename' to 'gSelectedNodeFile' (as distinct from the filename)
gSelectedNodeFile = None
# 29/7/24 DH: (Maybe we don't need this extra layer for accessing globals outside the file with Python 3)
#   Used to save a rounded weight for each node for each epoch during training (in order to graph changes by epoch)
gRoundedWeightsFile = None
# 13/9/24 DH: Specify the Node to track which was originally 287 (when Fine-tuning from Pre-trained rather than Norm Distrib)
gTrackedNode = None

# ----------------- END: Runtime Cfg ---------------