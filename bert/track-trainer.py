# 29/2/24 DH: A script to track and learn about Seq2SeqLM + QA training

# 2/3/24 DH:
import subprocess, sys, time, os

# https://docs.python.org/2/whatsnew/2.5.html#pep-328-absolute-and-relative-imports
sys.path.append(os.path.abspath('../huggingface'))
from stop_trainer import sigintPIDFromTrainerLog, parseTrainerStack, saveStackTraceFile

def runStackTraceCapture():
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
    backendScript = "track-trainer-backend.py"

    print()
    print(f"STARTING: {backendScript} (which is just 'run_qa.main()')")
    proc = subprocess.Popen(f"python {backendScript} {jsonFile}", shell=True, stderr=stackTextFD)
    #proc = subprocess.Popen(f"echo '  nice work, good job'", shell=True, stderr=stackTextFD)

  else:
    print("You need to provide a JSON config")
  
  # 4/3/24 DH: Necessary when change script-function namespace which triggers dataset download
  #sleepSecs = 120
    
  sleepSecs = 30
  print(f"  [Sleeping for {sleepSecs} secs (to let Trainer get started)]")
  time.sleep(sleepSecs)

  scriptDir = os.path.dirname(os.path.realpath(__file__))
  sigintPIDFromTrainerLog(scriptDir, waitFlag=False)

  # 2/3/24 DH: If being used to get a stack trace then it does not wait in 'sigintPIDFromTrainerLog()'
  #            ("KeyboardInterrupt" is the last line of the stack trace, SOMETIMES PENULTIMATE when sched immediately)

  # DB DeBug
  stackWorkingFDname = "stack-WORKING.txt"

  parseFile = stackTextFDname
  print(f"  *** Parse file set to: {parseFile} ***")
  with open(parseFile) as source :
    print()
    initLines = source.readlines()
    initLinesLen = len(initLines)
    lastLine = initLines[-1]
    
    print(f"  Last line: {lastLine.strip():50}(Read lines: {initLinesLen})")

    totalSleeps = 0
    maxSleeps = 5
    while "KeyboardInterrupt" not in lastLine and totalSleeps < maxSleeps:
      sleepSecs = 1
      print(f"  Sleeping for {sleepSecs} secs to provided time for stack trace to be returned")
      time.sleep(sleepSecs)

      # 4/3/24 DH: Catch unusual condition where last line is weird despite having normal stack trace
      totalSleeps += 1

      lines = source.readlines()
      linesLen = len(lines)
      if linesLen > 1:
        lastLine = lines[-1]
      if linesLen == 0: # ie when KeyboardInterrupt scheduled immediately in preference to Trainer
        lastLine = initLines[-3]
      print(f"  Last line: {lastLine.strip():50}(Read lines: {linesLen})")
    print()

  print(f"Terminating {proc}")
  proc.terminate()
  print(f"Closing: {stackTextFDname} ( opened for 'proc = subprocess.Popen(..., stderr=stackTextFD)' )")

  if totalSleeps < maxSleeps:
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    parseTrainerStack( os.path.join(scriptDir, parseFile) )
  else:
    newStackFilename = saveStackTraceFile(stackTextFDname)
    print(f"Total sleeps of {totalSleeps} was in excess of max {maxSleeps} so saving stack trace to {newStackFilename}")


# 4/3/24 DH:
if __name__ == "__main__":
  runStackTraceCapture()

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

