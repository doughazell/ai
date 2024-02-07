# ========================================================================================
# 19/1/24 DH: Added to github via 'ai' repo but originally run from '/Users/doug/src/gpt'
#             (This is where GPT-2 'models/124M' is situated)
# ========================================================================================

import gpt_2_simple as gpt2

import os
import requests
# 3/1/24 DH:
import subprocess

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
  print(f"Downloading {model_name} model...")
  gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
	
"""
file_name = "shakespeare.txt"
if not os.path.isfile(file_name):
	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	data = requests.get(url)

	with open(file_name, 'w') as f:
		f.write(data.text)
"""
file_name = "checkpoint/run1/doug.txt"

# 3/1/24 DH: Check where the softlink, '~/gpt/checkpoint/run1' is pointing
#   checkpoint$ ln -fns run1-2/ run1

output = subprocess.check_output("ls -al checkpoint/run1", shell=True)
outputSplit = str(output).split("->")
outputSplit2 = outputSplit[1].split("/")
modelVers = outputSplit2[0]

print("------------------")
print("Running:",modelVers)
print("------------------")

sess = gpt2.start_tf_sess()

# 1/1/24 DH: finetune() an empty model with dynamic data
saver = gpt2.finetune(sess,
	file_name,
	model_name=model_name,
	steps=1)   # steps is max number of training steps added to checkpoint (initially '1000')

#exit("Yea fuck off...")

# 1/1/24 DH:
print("Calling 'gpt2.generate()' with: ",sess)

gpt2.generate(sess,
	length=24) # 2/1/24 DH: Number of tokens to output
