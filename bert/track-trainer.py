# 29/2/24 DH: A script to track and learn about Seq2SeqLM + QA training

# 2/3/24 DH:
import subprocess, sys, time, os

# https://docs.python.org/2/whatsnew/2.5.html#pep-328-absolute-and-relative-imports
sys.path.append(os.path.abspath('../huggingface'))
from stop_trainer import sigintPIDFromTrainerLog, parseTrainerStack

if __name__ == "__main__":
  # 2/3/24 DH: Now moved to 'track_trainer_backend.py'
  #run_qa.main()

  # 1/3/24 DH: Having 'run_qa.main()' in 'track_trainer_backend.py' PROVIDES SEPERATE PID AND HENCE KILL TARGET 
  #                     *** WITHOUT CTRL-C SAVING ***
  #                               (we've got leading-lines to "aiLego_Lass")

  # 2/3/24 DH: https://pypi.org/project/gprof/

  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    jsonFile = sys.argv[1]

    # https://docs.python.org/3/library/subprocess.html#older-high-level-api
    stackTextFDname = "stack.txt"
    stackTextFD = open(stackTextFDname, "w")
    proc = subprocess.Popen(f"python track-trainer-backend.py {jsonFile}", shell=True, stderr=stackTextFD)

  else:
    print("You need to provide a JSON config")
  
  sleepSecs = 30
  print(f"  Sleeping for {sleepSecs} secs (to let Trainer get started)")
  time.sleep(sleepSecs)

  scriptDir = os.path.dirname(os.path.realpath(__file__))
  sigintPIDFromTrainerLog(scriptDir, waitFlag=False)

  # 2/3/24 DH: If being used to get a stack trace then it does not wait in 'sigintPIDFromTrainerLog()'
  #            ("KeyboardInterrupt" is the last line of the stack trace, SOMETIMES IT IS PENULTIMATE...!!!)
  with open(stackTextFDname) as source :  
    print()
    lines = source.readlines()
    lastLine = lines[-1]
    ws = ""
    print(f"  Last line: {lastLine.strip():50}(Read lines: {len(lines)})")

    while "KeyboardInterrupt" not in lastLine:
      sleepSecs = 1
      print(f"  Sleeping for {sleepSecs} secs to provided time for stack trace to be returned")
      time.sleep(sleepSecs)

      lines = source.readlines()
      if len(lines) > 1:
        lastLine = lines[-1]
      else:
        lastLine = lines[0]
      print(f"  Last line: {lastLine.strip():50}(Read lines: {len(lines)})")
    print()

  print(f"Terminating {proc}")
  proc.terminate()
  print(f"Closing: {stackTextFDname}")

  scriptDir = os.path.dirname(os.path.realpath(__file__))
  parseTrainerStack( os.path.join(scriptDir, stackTextFDname) )


# =================================================================================================

# Q&A: QuestionAnswering / QuestionAnsweringTrainer ('run_qa.py')
# ---------------------------------------------------------------
""" A SEWING MACHINE THREAD LINE
    ----------------------------
    [THE STACK TRACE WILL DEPEND ON THE EXACT TIMING OF CTRL-C]

File "run_qa.py", line 675, in main
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
File "transformers/trainer.py", line 1557, in train

  [1826] # ------------------------------------------------------------------
  [1827] # 9/2/24 DH: ------ MAIN TRAINING LOOP _inner_training_loop() ------
  [1828] # ------------------------------------------------------------------
File "transformers/trainer.py", line 1890, in _inner_training_loop
File "transformers/trainer.py", line 2824, in training_step

  def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    [2816] model.train()
    [2817] inputs = self._prepare_inputs(inputs)
    [2824] loss = self.compute_loss(model, inputs)
    
File "transformers/trainer.py", line 2847, in compute_loss
  outputs = model(**inputs)
File "torch/nn/modules/module.py", line 1518, in _wrapped_call_impl                 [1518] TORCH
File "torch/nn/modules/module.py", line 1527, in _call_impl                         [1527]   "
  return forward_call(*args, **kwargs)
File "transformers/models/bert/modeling_bert.py", line 1846, in forward                          [1846] BertForQuestionAnswering.forward()
  outputs = self.bert(
File "torch/nn/modules/module.py", line 1518, in _wrapped_call_impl                 [1518] TORCH
File "torch/nn/modules/module.py", line 1527, in _call_impl                         [1527]   "
  return forward_call(*args, **kwargs)
File "transformers/models/bert/modeling_bert.py", line 1013, in forward                          [1013] BertModel.forward()
  encoder_outputs = self.encoder(
File "torch/nn/modules/module.py", line 1518, in _wrapped_call_impl                 [1518] TORCH
File "torch/nn/modules/module.py", line 1527, in _call_impl                         [1527]   "
  return forward_call(*args, **kwargs)
File "transformers/models/bert/modeling_bert.py", line 607, in forward                           [607] BertEncoder.forward()
  layer_outputs = layer_module(
File "torch/nn/modules/module.py", line 1518, in _wrapped_call_impl                 [1518] TORCH
File "torch/nn/modules/module.py", line 1527, in _call_impl                         [1527]   "
  return forward_call(*args, **kwargs)
File "transformers/models/bert/modeling_bert.py", line 539, in forward                           [539] BertLayer.forward()
File "transformers/pytorch_utils.py", line 236, in apply_chunking_to_forward
  return forward_fn(*input_tensors)
File "transformers/models/bert/modeling_bert.py", line 552, in feed_forward_chunk
  layer_output = self.output(intermediate_output, attention_output)
File "torch/nn/modules/module.py", line 1518, in _wrapped_call_impl                 [1518] TORCH
File "torch/nn/modules/module.py", line 1527, in _call_impl                         [1527]   "
File "transformers/models/bert/modeling_bert.py", line 464, in forward                           [464] BertOutput.forward()
  hidden_states = self.dense(hidden_states)
File "torch/nn/modules/module.py", line 1518, in _wrapped_call_impl                 [1518] TORCH
File "torch/nn/modules/module.py", line 1527, in _call_impl                         [1527]   "
File "torch/nn/modules/linear.py", line 114, in forward
  return F.linear(input, self.weight, self.bias)
"""


# Generative: Seq2SeqLM / QuestionAnsweringSeq2SeqTrainer ('run_seq2seq_qa.py')
# -----------------------------------------------------------------------------

