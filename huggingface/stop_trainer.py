# 11/2/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import os, sys, signal, time
import transformers
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

# 2/3/24 DH:
import psutil

# 3/3/24 DH:
import sqlite3
# 9/3/24 DH:
import io

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

"""
# -----------------------------------------------------------------------------------
  Functions
  ---------
  saveStackTraceFile(stackTextFDname)
  getOrCreateDB(stackFile, trainingFunction)
  checkForRecord(cursor, stmnt, tallyFieldNum)
  populateDB(stackFile, trainingFunction, records)
  parseTrainerStack(stackFile)
  getCmdLineArgs()
  sigintPIDFromTrainerLog(scriptDir, waitFlag=True)
  waitForKeyboardInterrupt(parseFile, parseFD)
  
# -----------------------------------------------------------------------------------
"""

# 4/3/24 DH:
def saveStackTraceFile(stackTextFDname):
  # Record the stack trace for user inspection for bug-fix
  # copy 'stack.txt' to 'stack-DTG.txt'
  stackDTGname = time.strftime("stack-%Y%m%d-%H%M%S.txt")
  os.rename(stackTextFDname, stackDTGname)
  
  return stackDTGname

# 3/3/24 DH: (This probably will be moved into a separate file at some point)
"""
  # 2/3/24 DH: 'trainingFunction' Table
  ID, File, Line, Function, Tally

  # 3/3/24 DH: Call seq Table
  ID, TrainingFunction, SeqIDs, Tally
"""
def getOrCreateDB(stackFile, trainingFunction):
  # ------------------------ DB CONNECTION -------------------------
  try:
    stack_trace_db=os.path.join(os.path.dirname(stackFile), "stack_trace.db")
    traceDB = sqlite3.connect(stack_trace_db, check_same_thread=False)
    print("Opened connection to cache file: ", stack_trace_db)

  except sqlite3.OperationalError as e:
    print(e)
    return None
  
  # ------------------- 'trainingFunction' TABLE --------------
  try:
    # From '.deeppavlov/downloads/odqa/enwiki_schema.sql' :
    #   CREATE TABLE documents (id, title, text);
    #   CREATE INDEX idx_id ON documents(id);
    cursor = traceDB.cursor()
    cursor.execute(
      # "SELECT text FROM {} WHERE id = ?".format(self.db_name),(doc_id,)

      # "INSERT INTO documents (id, title, text) VALUES (?,?,?)",
      # (record['id'], record['title'], record['text'])

      f"CREATE TABLE {trainingFunction} (id INTEGER PRIMARY KEY, file, line, function, tally)"
    )

    index_name = f"{trainingFunction}_id"
    cursor.execute(
      f"CREATE INDEX {index_name} ON {trainingFunction}(id)"
    )
    traceDB.commit()
    cursor.close()
    print(f"  Created table: '{trainingFunction}'")
    print(f"     with index: '{index_name}'")

  except sqlite3.OperationalError as e:
    if f"table {trainingFunction} already exists" in e.args:
      print(f"  Table: '{trainingFunction}' already exists")
    else:
      print(e)
  
  # --------------------- 'call_sequence' TABLE -----------------
  try:
    cursor = traceDB.cursor()

    table_name = "call_sequence"
    cursor.execute(
      f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, training_function, seq_ids, tally)"
    )

    index_name = "call_sequence_id"
    cursor.execute(
      f"CREATE INDEX {index_name} ON {table_name}(id)"
    )
    traceDB.commit()
    cursor.close()
    print(f"  Created table: '{table_name}'")
    print(f"     with index: '{index_name}'")

  except sqlite3.OperationalError as e:
    if f"table {table_name} already exists" in e.args:
      print(f"  Table: '{table_name}' already exists")
    else:
      print(e)

  return traceDB

def checkForRecord(cursor, stmnt, tallyFieldNum):
  try:
    cursor.execute(stmnt)
    # 'fetchall() returns an array of tuples for wanted fields
    result = cursor.fetchall()
    resultNum = len(result)
    if resultNum > 0:
      id = result[0][0]
      tally = result[0][tallyFieldNum]
      return (id, tally)
    else:
      return (0, 0)

  except sqlite3.OperationalError as e:
    print(e)

def populateDB(stackFile, trainingFunction, records):
  print()
  print(f"Adding {len(records)} record call-stack to '{trainingFunction}' SQLite table")

  traceDB = getOrCreateDB(stackFile, trainingFunction)
  
  try:
    cursor = traceDB.cursor()

    seqIDs = []
    for record in records:
      # Need to firstly check to see whether "file-line" has already been added to DB and its associated 'tally' (prior to incrementing)
      #
      # 'checkForRecord()' returns (0,0) for unseen "file-line"
      stmnt = f"SELECT * FROM {trainingFunction} WHERE file = '{record['file']}' AND line = '{record['line']}'"
      (id, tally) = checkForRecord(cursor, stmnt, tallyFieldNum=4)

      newTally = int(tally) + 1
      if tally == 0:
        stmnt = f"INSERT INTO {trainingFunction} (file, line, function, tally) VALUES ('{record['file']}', '{record['line']}', '{record['function']}', {newTally})"
      else:
        stmnt = f"UPDATE {trainingFunction} SET tally = {newTally} WHERE id={id}"

      print(stmnt)
      # 4/3/24 DH: Need ID of new "file-line" record for 'call-sequence' Table
      cursor.execute(stmnt)
      if "INSERT INTO" in stmnt:
        # https://peps.python.org/pep-0249/#lastrowid, "most databases return a rowid only when a single INSERT operation is performed"
        id = cursor.lastrowid
      traceDB.commit()

      # 4/3/24 DH: The 'call_sequence' Table requires complete seq of valid ID's for complete stack-trace
      seqIDs.append(id)
    
    # 4/3/24 DH: Now insert a record into 'call_sequence' Table (if it does not already exist)
    print(f"Call sequence ({len(seqIDs)}): {seqIDs}")

    stmnt = f"SELECT * FROM call_sequence WHERE training_function = '{trainingFunction}' AND seq_ids = '{seqIDs}'"
    (id, tally) = checkForRecord(cursor, stmnt, tallyFieldNum=3)

    newTally = int(tally) + 1
    if tally == 0:
      stmnt = f"INSERT INTO call_sequence (training_function, seq_ids, tally) VALUES ('{trainingFunction}', '{seqIDs}', '{newTally}')"
    else:
      stmnt = f"UPDATE call_sequence SET tally = {newTally} WHERE id={id}"

    print(f"  {stmnt}")
    cursor.execute(stmnt)
    traceDB.commit()

    cursor.close()
  except sqlite3.OperationalError as e:
    print(e)

# 2/3/24 DH:
def parseTrainerStack(stackFile):
  records = []

  with open(stackFile) as source :
    print()
    print("Filename: ", stackFile)
    print("---------")

    textLines = [line.strip() for line in source.readlines() if line.strip()]
  
  trainingFunction = None

  # Eg:
  # 1) File "/Users/doug/src/ai/bert/run_qa.py", line 338, in main
  # 2) File "/Users/doug/.pyenv/versions/3.9.15/lib/python3.9/site-packages/datasets/data_files.py", line 689, in from_patterns
  # 3) File "<string>", line 3, in raise_from

  for line in textLines:
    # BE AWARE OF LINES LIKE: "    data_files = DataFilesDict.from_patterns("
    #                     *** critical space after "File" ***
    #
    # (Currently no except handler for "IndexError: list index out of range" in order to inspect Traceback)
    if "File " in line:

      lineSplit = line.split("site-packages/")
      if len(lineSplit) > 1:
        linePart = lineSplit[1]
      else: # non 'site-packages' path
        linePart = lineSplit[0] # ie whole line
        lineSplit = linePart.split("File \"")
        if len(lineSplit) > 1:
          linePart = lineSplit[1]
      
      lineSplit = linePart.split("\", ")
      fileName = lineSplit[0]
      otherParts = lineSplit[1]

      lineSplit = otherParts.split(", in ")
      lineNum = lineSplit[0].lstrip("line ")
      funcName = lineSplit[1]

      # 2/3/24 DH: The last 'trainer.py' entry in the stack is the 'training_step()' function
      if "transformers/trainer.py" in fileName:
        trainingFunction = funcName

      recordDict = {'file': fileName, 'line': lineNum, 'function': funcName}
      print(f"FILE: {recordDict['file']:50} LINE: {recordDict['line']:7} FUNCTION: {recordDict['function']}")
      
      records.append(recordDict)
  # ---------------------------- END: for line in textLines --------------------------

  # 4/3/24 DH: If HuggingFace is downloading a dataset then the normal time to start is insufficient for training to start
  if trainingFunction:
    populateDB(stackFile, trainingFunction, records)
  else:
    newStackFilename = saveStackTraceFile(stackFile)
    print(f"There is no 'trainingFunction' in {stackFile} so saving to {newStackFilename}")

# 8/3/24 DH:
def getCmdLineArgs():
  parser = HfArgumentParser((Arguments))
  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    jsonFile = os.path.abspath(sys.argv[1])
  else:
    jsonFile = os.path.join(scriptDir, "qa_train.json")

  # 11/2/24 DH: Need ',' after 'args' in order to parse tuple response (the "()" are not required but make the tuple clearer)
  (args,) = parser.parse_json_file(json_file=jsonFile, allow_extra_keys=True)

  return args

# 2/3/24 DH: Refactor to use a library rather than cmd line script
def sigintPIDFromTrainerLog(scriptDir, args, waitFlag=True):

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

# 9/3/24 DH: If 'parseFDwriter != 'io.IOBase' then this is being used as A TEST HARNESS TO PARSE KEPT STDERR'S
#            ('parseFDwriter' is INTEGER ARRAY used to chunk the file to simulate phased IO)
#
# NOTE: TQDM uses '^M' to get new-line which is displayed by Vi on same line as following text
def waitForKeyboardInterrupt(parseFile, parseFDwriter):
  # 2/3/24 DH: If being used to get a stack trace then it does not wait in 'sigintPIDFromTrainerLog()'
  #            ("KeyboardInterrupt" is the last line of the stack trace, SOMETIMES PENULTIMATE when sched immediately)

  # DB DeBug
  #parseFile = "stack-WORKING.txt"

  print(f"  *** Parse file set to: {parseFile} ***")
  with open(parseFile) as source :
    # 8/3/24 DH: Needed to append to renamed 'parseFile'
    printLineList = []

    # https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
    # "If you want to read all the lines of a file in a list you can also use list(f) or f.readlines()."
    initLines = []
    if isinstance(parseFDwriter,io.IOBase):
      initLines = source.readlines()    
    else: # TEST HARNESS FOR PARSING STDERR: read the number of lines specified by 'parseFDwriter' integer array
      linesToRead = parseFDwriter[0]
      for _ in range(linesToRead):
        line = source.readline() # "if f.readline() returns an empty string, the end of the file has been reached"
        if len(line) > 0:
          initLines.append(line)

    initLinesLen = len(initLines)
    if len(initLines) > 1:
      lastLine = initLines[-1]
    else:
      lastLine = "Error"
    
    sourceLine = "INIT LINES"
    printLine = f"  Last line: {lastLine.strip():50}(Read lines: {initLinesLen} from {sourceLine})"
    printLineList.append(printLine)
    print()
    print(printLine)

    totalSleeps = 0
    totalEmptyLines = 0
    maxSleeps = 5
    lines = []
    while "KeyboardInterrupt" not in lastLine and totalSleeps < maxSleeps:
      sleepSecs = 1
      print(f"  Sleeping for {sleepSecs} secs to provide time for stack trace to be returned")
      time.sleep(sleepSecs)

      # 4/3/24 DH: Catch unusual condition where last line is weird despite having normal stack trace
      totalSleeps += 1

      if isinstance(parseFDwriter,io.IOBase):
        lines = source.readlines() # overrides 'lines = []' added above for TEST HARNESS
        linesLen = len(lines)
      else: # TEST HARNESS FOR PARSING STDERR: read the number of lines specified by 'parseFDwriter' integer array
        try:
          linesToRead = parseFDwriter[totalSleeps]
        except IndexError:
          linesToRead = 0

        for _ in range(linesToRead):
          line = source.readline() # "if f.readline() returns an empty string, the end of the file has been reached"
          if len(line) > 0:
            lines.append(line)
          
        linesLen = linesToRead
      # ------------------ END: TEST HARNESS ------------------
      
      if linesLen > 1:
        lastLine = lines[-1]
        sourceLine = "LAST LINE"

      # FALLBACK OPTIONS
      # ----------------
      if linesLen == 0: # ie unusual scheduling set theory overlap
        totalEmptyLines += 1
        try:
          if totalEmptyLines == 1:
            # Accommodating the "^M" line (NOT OBVIOUS FROM 'VI' LINENUMS)
            lastLine = lines[-3]
            sourceLine = "LAST LINES[-3]"
          if totalSleeps > 2:
            lastLine = initLines[-3]
            sourceLine = "INIT LINES[-3]"
        except IndexError:
          pass # for next sleep/read cycle
      # ----------------
      # 'lastLine' & 'sourceLine' maintained from last condition met
      printLine = f"  Last line: {lastLine.strip():50}(Read lines: {linesLen} so using {sourceLine})"
      printLineList.append(printLine)
      print(printLine)
  # ------------------------------------ END: with open(parseFile) as source --------------------------------
  
  # https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
  # "If you’re not using the with keyword, then you should call f.close() to close the file and immediately free up any system resources used by it."
  print()
  print(f"Closing: {parseFile} ( opened for 'proc = subprocess.Popen(..., stderr=stackTextFD)' )")

  if totalSleeps > maxSleeps and isinstance(parseFDwriter,io.IOBase):
    # 8/3/24 DH: Need to add "Last line" printout above to end of complete file 
    #            (in order to add new code pathways to cover unusual scheduling events)
    parseFDwriter.write("\n")
    parseFDwriter.write("Script used lines\n")
    parseFDwriter.write("-----------------\n")
    for line in printLineList:
      parseFDwriter.write(line + "\n")

    newStackFilename = saveStackTraceFile(parseFile)
    print(f"Total sleeps of {totalSleeps} was in excess of max {maxSleeps} so saving stack trace to {newStackFilename}")

    return False
  
  return True

if __name__ == "__main__":
  
  sigintPIDFromTrainerLog(scriptDir)
