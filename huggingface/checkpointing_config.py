# 10/4/23 DH: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

# 24/2/24 DH: Taken from 'mnist-training-errors/gym_config.py'
# This needs to be imported for read/write-access with: "import checkpointing_config"
#   ("from gym_config import *" provided read-access in 'mnist-training-errors/dqn_c51.py')

# ----------------- Runtime Cfg --------------------

trainer = None
training_args = None

# 14/5/24 DH: Used to save a layer weights (for later graphing)
gWeightsFile = None
# 21/5/24 DH: Directory used to save a weights graph
gGraphDir = None
gFullWeightsFile = None
# 25/5/24 DH: Used to save the loss for each epoch (for later graphing)
gLossFile = None
# 8/6/24 DH: Used to save 'BertSelfAttention' 384 logits for selected Node (currently 287)
gSelectedNodeFilename = None

# ----------------- END: Runtime Cfg ---------------