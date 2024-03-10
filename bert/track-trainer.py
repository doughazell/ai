# 29/2/24 DH: A script to track and learn about Seq2SeqLM + QA training

# 2/3/24 DH:
import subprocess, sys, time, os, random, numpy, csv
# 7/3/24 DH:
from subprocess import Popen, PIPE, STDOUT
# 8/3/24 DH:
from datetime import datetime

# https://docs.python.org/2/whatsnew/2.5.html#pep-328-absolute-and-relative-imports
sys.path.append(os.path.abspath('../huggingface'))
from stop_trainer import sigintPIDFromTrainerLog, parseTrainerStack, saveStackTraceFile, getCmdLineArgs, waitForKeyboardInterrupt

# 6/3/24 DH: Run 'runStackTraceCapure()' at random time intervals to get full code coverage
#   ('random  // 10' secs should provide sufficient HuggingFace-Transformer training code coverage)
#   (The used time delays should be a horizontal line histogram over the interval set)
#
#   Use 'delay-hist.csv' for csv of each time interval (which gets updated regularly on delta-reset)
  
def runRandomIntervalCapture(iterations, numOfIntervals=40):
  csvFile = "delay-hist.csv"
  try:
    with open(csvFile, 'r') as csvfile:
      intervalLogReader = csv.reader(csvfile)
      # 6/3/24 DH: 'csv.reader' iteration returns a 2-D list (one list for each row)
      csvFileContents = [row for row in intervalLogReader][0]
      #print(f"{csvFile}: {csvFileContents}, {csvFileContents.__class__} of {csvFileContents[0].__class__}")

      csvIntegers = [int(elem) for elem in csvFileContents]
      print(f"'{csvFile}': {csvIntegers}, {csvIntegers.__class__} of {csvIntegers[0].__class__} (after 'int(elem)' for each {csvFileContents[0].__class__})")
      intervalLog = numpy.array(csvIntegers)
  except FileNotFoundError as e:
    print(f"('{csvFile}' not found so starting from scratch)")
    intervalLog = numpy.zeros(numOfIntervals, dtype=numpy.uint32)
  
  print(f"PRE-LOOP:  {intervalLog}, {intervalLog.__class__}")

  # 7/3/24 DH: The 'numOfIntervals' needs to be the length of time of 1 training cycle
  for i in range(iterations):
    # "Return the next random floating point number in the range 0.0 <= X < 1.0"
    randomTime = random.random()
    deltaSecs = int( (randomTime * numOfIntervals * numOfIntervals) // numOfIntervals )
    #print(f"Random: {randomTime}, Delta: {deltaSecs}, Intervals: {numOfIntervals}")
    
    intervalLog[deltaSecs] += 1

    firstTimeFlag = False
    if i == 0:
      firstTimeFlag = True

    (dt, baseTime) = runStackTraceCapture(deltaSecs, firstTimeFlag)
    print()
    print(f"COMPLETED runStackTraceCapture() {i+1}/{iterations}) numOfIntervals: {numOfIntervals}, dt: {dt}, base time: {baseTime}")

  print()
  print(f"POST-LOOP: {intervalLog}")
  with open(csvFile, 'w') as csvfile:
    # https://docs.python.org/3/library/csv.html#csv.writer
    intervalLogWriter = csv.writer(csvfile)
    intervalLogWriter.writerow(intervalLog)

# 8/3/24 DH:
def getLogTime(line):
  lineParts = line.split(" ")
  if len(lineParts) > 2:
    logTime = lineParts[1]
    return logTime

# 7/3/24 DH:  
def parseTQDMline(line, backendPID):
  dt = 0
  n = 0
  baseTime = 0

  # 2024-03-08 17:15:03,723 [INFO] PID: 13154
  # 2024-03-08 17:15:45,816 [INFO] tqdm.refresh(): dt: 33.263055086135864, n: 12315
  if str(backendPID) in line:
    parseTQDMline.startTime = getLogTime(line)

  lineParts = line.split("tqdm.refresh(): ")
  if len(lineParts) > 1:
    parseTQDMline.firstEndTime = getLogTime(line)

    tqdmInfo = lineParts[1]
    
    tqdmInfoParts = tqdmInfo.split(",")
    if len(tqdmInfoParts) > 1:
      dtStr = tqdmInfoParts[0]
      dtStrParts = dtStr.split(":")
      if len(dtStrParts) > 1:
        dt = round( float(dtStrParts[1]) )

      nStr = tqdmInfoParts[1]
      nStrParts = nStr.split(":")
      if len(nStrParts) > 1:
        n = nStrParts[1]

  if hasattr(parseTQDMline, 'startTime') and hasattr(parseTQDMline, 'firstEndTime'):
    print()
    print(f"  Get time diff between {parseTQDMline.startTime} and {parseTQDMline.firstEndTime}")
    FMT = '%H:%M:%S,%f'
    timeDiff = datetime.strptime(parseTQDMline.firstEndTime, FMT) - datetime.strptime(parseTQDMline.startTime, FMT)
    timeDiffSecs = round(timeDiff.total_seconds())
    print(f"  Time diff (rounded secs): {timeDiffSecs}")

    # The base time is taken from [1st TQDM time - PID time - TQDM dt] in 'seq2seq_qa_INtrainer.log'
    baseTime = timeDiffSecs - dt
    print(f"  Base time: {baseTime}")

  return (dt, n, baseTime)

# 7/3/24 DH:
def getTrainTime(args, backendScript, backendPID):
  # 8/3/24 DH: 'args.trainer_log' default of 'seq2seq_qa_INtrainer.log' in 'stop_trainer.py::Arguments'
  procProgress = subprocess.Popen(f"tail -f {args.output_dir}/{args.trainer_log}", shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
  pid = os.getpid()
  print(f"{__file__} pid: {pid}, {backendScript} pid: {backendPID}")

  displayFlag = False
  dt = 0
  for line in procProgress.stdout:
    # Convert from rec'd 'bytes' to 'string'
    line = line.decode()

    # Make sure tailing correct version of file
    if str(backendPID) in line:
      displayFlag = True

    if displayFlag:
      (dt, n, baseTime) = parseTQDMline(line, backendPID)

      print(f"  TQDM: {dt}, {n}")
    
    if dt > 0:
      break
  
  print(f"Terminating {procProgress}")
  procProgress.terminate()

  return (dt, baseTime)

def runStackTraceCapture(deltaSecs, firstTimeFlag):
  # 2/3/24 DH: Now moved to 'track_trainer_backend.py'
  #run_qa.main()

  # 1/3/24 DH: Having 'run_qa.main()' in 'track_trainer_backend.py' PROVIDES SEPERATE PID AND HENCE KILL TARGET 
  #                     *** WITHOUT CTRL-C SAVING ***
  #                               (we've got leading-lines to "aiLego_Lass")

  # 2/3/24 DH: https://pypi.org/project/gprof/

  # 8/3/24 DH:
  args = getCmdLineArgs()
  scriptDir = os.path.dirname(os.path.realpath(__file__))

  dt = 0 # to cover when not first time (and 'getTrainTime()' not called)
  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    jsonFile = sys.argv[1]

    # https://docs.python.org/3/library/subprocess.html#older-high-level-api
    stackTextFDname = "stack.txt"
    stackTextFDwriter = open(stackTextFDname, "w")
    backendScript = "track-trainer-backend.py"

    print()
    print(f"STARTING: {backendScript} (which is just 'run_qa.main()')")
    proc = subprocess.Popen(f"python {backendScript} {jsonFile}", shell=True, stderr=stackTextFDwriter)
    backendPID = proc.pid
    #proc = subprocess.Popen(f"echo '  nice work, good job'", shell=True, stderr=stackTextFD)
    
    if firstTimeFlag:
      # 8/3/24 DH: 'dt' is needed for the random sleep set in 'runRandomIntervalCapture(iterations, numOfIntervals=40)'
      #
      # (It is currently hard-coded and this was a mechanism to ensure complete coverage of code lines over Q&A training
      #  by storing the max train cycle time in 'delay-hist.csv' (SEE 'baseTime' BELOW))
      (dt, baseTime) = getTrainTime(args, backendScript, backendPID)
    else:
      print("  Not first time so continuing...")

  else:
    print("You need to provide a JSON config")
    return (0,0)
  
  # 4/3/24 DH: Necessary when change script-function namespace which triggers dataset download
  #sleepSecs = 120
  
  # 8/3/24 DH: TODO: Automate calc of base time via 'getTrainTime()'
  #              & pass back to 'runRandomIntervalCapture()' to be stored in LINE 1 & 2 of 'delay-hist.csv': 
  #                <Base time>,<Num of intervals>
  #                <csv Interval Count>
  if "baseTime" in locals():
    baseTime = min(10,baseTime)
  else:
    baseTime = 10
  
  sleepSecs = baseTime + deltaSecs
  print(f"  [Sleeping for ({baseTime} + {deltaSecs}) {sleepSecs} secs (now that Trainer has started)]")
  time.sleep(sleepSecs)

  sigintPIDFromTrainerLog(scriptDir, args, waitFlag=False)

  print(f"stackTextFDwriter: {stackTextFDwriter}")
  stackRecdOK = waitForKeyboardInterrupt(stackTextFDname, stackTextFDwriter)

  print()
  print(f"Terminating {proc}")
  proc.terminate()

  if stackRecdOK:
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    parseTrainerStack( os.path.join(scriptDir, stackTextFDname) )

  return (dt, baseTime)

if __name__ == "__main__":
  
  runRandomIntervalCapture(iterations=2)

  # TEST HARNESS FOR PARSING STDERR
  # -------------------------------
  #waitForKeyboardInterrupt(parseFile="stack-9Mar-TEST.txt", parseFDwriter=[10,51])

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

