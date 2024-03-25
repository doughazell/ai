# 16/3/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import subprocess, time, os, copy
from subprocess import Popen, PIPE, STDOUT

from stop_trainer import getSIGTERMcnt
from stop_trainer_utils import graphSleeptimeDistrib
# 20/3/24 DH:
import matplotlib.pyplot as plt

# ------------------------- GLOBALS -----------------------------
gStringList = [
  'objects to clean up at shutdown',
  'Last line: To enable the following instructions:',
  '_load_pretrained_model',
  'download_loading_script',
  'import_main_class',
  'evaluate/loading.py',
  'AutoConfig.from_pretrained',
  'from_pretrained',
  'frozen importlib._bootstrap',
  'load_dataset',
  'load_processed_shard_from_cache',
  '_load_from_checkpoint',
]
gListLen = len(gStringList)

gStackFileGlob = "stack-2024*"

# 19/3/24 DH:
gStatementList = []
gDeleteList = []
gOutputFile = "stack_error_logs.txt"
gFilenamesFile = "stack_error_log_files.txt"
# ---------------------------------------------------------------

# --------------------------- FUNCTIONS -------------------------
"""
deleteLogFiles(strDict)
recordLogFiles(strDict)
displayOrderedList(strDict)
checkLogFileDetails(allFiles, totalCnt)
graphSleeptimeDistrib(xVals, yVals)
getNoTrgFuncTotals(xVals, yVals)
outputStatementList()
getTypesWithoutTrgFuncError(tmpStrDict, noTrgFuncFileList)
sortErrorLogs()
"""
# ---------------------------------------------------------------
def deleteLogFiles(strDict):
  gDeleteList.append("")
  gDeleteList.append("Removing:")
  for key in strDict:
    gDeleteList.append(f"  '{key}':")
    for elem in strDict[key]:
      gDeleteList.append(f"    {elem}")

      os.remove(elem)

    gDeleteList.append("")

# 20/3/24 DH:
def recordLogFiles(strDict):
  gDeleteList.append("")
  for key in strDict:
    gDeleteList.append(f"'{key}':")
    for elem in strDict[key]:
      gDeleteList.append(f"  {elem}")

    gDeleteList.append("")

# 17/3/24 DH:
def displayOrderedList(strDict):
  global gListLen
  cnt = 0

  # Returns a list: https://docs.python.org/3/library/functions.html#sorted
  orderedList = sorted(strDict, key=lambda key: len(strDict[key]), reverse=False)

  for key in orderedList:
    cnt += 1

    gStatementList.append(f"{cnt}/{gListLen}) '{key}'")
    gStatementList.append(f"-----")
    gStatementList.append(f"  {len(strDict[key])}")
    gStatementList.append("")

def checkLogFileDetails(allFiles, totalCnt):
  subTotal = 0
  strDict = {}

  # ------------------- Cycle through each type of file ----------
  cnt = 0
  global gListLen

  for searchStr in gStringList:
    cnt += 1

    proc = subprocess.Popen(f"grep -l '{searchStr}' {gStackFileGlob}", shell=True, stdout=PIPE)
    for line in proc.stdout:
      # Convert from rec'd 'bytes' to 'string'
      line = line.decode().strip()

      # Only add filename if not already added elsewhere (to simulate later deletion)
      fileToAdd = True
      for key in strDict:
        if line in strDict[key]:
          #print(f"(Already added: {line} in '{key}')")
          fileToAdd = False

      if fileToAdd:
        # Firstly delete it from the whole set to find any missed
        allFiles.remove(line)

        try:
          strDict[searchStr].append(line)
        except KeyError:
          strDict[searchStr] = []
          strDict[searchStr].append(line)
    
    try:
      searchStrNum = len(strDict[searchStr])
    except KeyError:
      searchStrNum = 0

    subTotal += searchStrNum

    proc.wait()
    proc.terminate()
  # END: --- for searchStr in gStringList ---
  
  displayOrderedList(strDict)

  sigtermCnt = getSIGTERMcnt()
  totalRemaining = totalCnt - subTotal
  gStatementList.append(f"Total: {totalCnt}, Sub total: {subTotal}, Total remaining: {totalRemaining}, SIGTERM cnt: {sigtermCnt}")
  if totalRemaining > 0:
    gStatementList.append(f"Total files remaining:")
    for logFile in allFiles:
      gStatementList.append(f"  {logFile}")

  return strDict

# 18/3/24 DH:
def getNoTrgFuncTotals(xVals, yVals):
  fileList = []
  sleepTimeDict = {}

  searchStr = "There is no 'trainingFunction'"
  grepStr = f"grep \"{searchStr}\" {gStackFileGlob}"

  proc = subprocess.Popen(grepStr, shell=True, stdout=PIPE)
  number = 0
  for line in proc.stdout:
    number += 1

    # Convert from rec'd 'bytes' to 'string'
    line = line.decode().strip()

    lineSplit = line.split("sleeping")
    if len(lineSplit) > 1:
      sleepSecs = int(lineSplit[1].rstrip(" secs"))

      try:
        sleepTimeDict[sleepSecs] += 1
      except KeyError:
        sleepTimeDict[sleepSecs] = 1

      # 20/3/24 DH: Getting the type of files that don't have (stack + no trainingFunction)
      filename = lineSplit[0].split(":")[0]
      fileList.append(filename)
      
  
  proc.wait()
  proc.terminate()

  try:
    # 19/3/24 DH: Gets sorted entries for number in each time interval
    #orderedList = sorted(sleepTimeDict, key=lambda key: sleepTimeDict[key], reverse=False)

    #keyNum = 23
    #del sleepTimeDict[keyNum]
    #gStatementList.append("")
    #gStatementList.append(f"  REMOVED KEY: {keyNum}")

    orderedList = sorted(sleepTimeDict.keys())
    minVal = orderedList[0]
    maxVal = orderedList[-1]

  except (ValueError, IndexError) as e:
    minVal = 0
    maxVal = 0
  
  gStatementList.append("")
  gStatementList.append(f"Sleep time resulting in no 'trainingFunction' necessary for DB in {number} files:")
  gStatementList.append(f"  Min: {minVal} secs - {sleepTimeDict[minVal]}")
  
  # ------------------------------------------------------------------------------
  
  cnt = orderedList[1] - 1
  for key in orderedList[1:-1]:

    # Add missing times
    if key != cnt + 1:
      for cnt in range(cnt, key-1):
        gStatementList.append(f"     {cnt + 1}")
    cnt = key

    gStatementList.append(f"    ({key} secs) {sleepTimeDict[key]}")

    # 20/3/24 DH: Gets propagated back to caller
    xVals.append(key)
    yVals.append(sleepTimeDict[key])
  # ------------------------------------------------------------------------------

  gStatementList.append(f"  Max: {maxVal} secs - {sleepTimeDict[maxVal]}")

  return fileList
  

# 19/3/24 DH:
def outputStatementList():
  with open(gOutputFile, "w") as outFile:
    for line in gStatementList:
      print(line)
      outFile.write(line + "\n")

  with open(gFilenamesFile, "w") as outFile:
    for line in gDeleteList:
      outFile.write(line + "\n")

# 20/3/24 DH:
def getTypesWithoutTrgFuncError(tmpStrDict, noTrgFuncFileList):
  # 'strDict' is array of files for each grep search str
  for file in noTrgFuncFileList:
    for key, keyFileList in tmpStrDict.items():
      # "substr in" needs "for item in list" (whereas "whole string" can be found with just "in list")
      #   (See 'stop_trainer.py::parseTrainerStack()')
      if file in keyFileList:
        # https://docs.python.org/3/tutorial/controlflow.html#id2, "if a mutable object is passed, the caller will see 
        # any changes the callee makes to it"
        # https://docs.python.org/3/library/stdtypes.html#mapping-types-dict, "[Dictionarys] are mutable" (so like pointer to object)
        tmpStrDict[key].remove(file)
    
    # IDEAS:
    #   1) if file in strDict.values():
    #   2) key = list(filter(lambda x: my_dict[x] == 100, my_dict))[0]
  
  print()
  print(f"  STACK LOG TYPES WITHOUT TRG FUNC ERROR")
  print( "  --------------------------------------")

  totalCnt = 0
  keyNum = 0
  totalKeys = len(tmpStrDict.keys())
  for key in tmpStrDict:
    keyNum += 1
    if len(tmpStrDict[key]) > 0:
      print(f"  '{key}':", end='')

    cnt = 0
    for filename in tmpStrDict[key]:
      if cnt == 0:
        fileKeep = filename
      cnt += 1
    
    if cnt > 0:
      print(f"  {cnt}")
      print(f"    {fileKeep}")

      if keyNum != totalKeys:
        print()

      totalCnt += cnt
  
  print( "  ===========")
  print(f"  TOTAL FILES: {totalCnt}")
  print()

def sortErrorLogs():
  
  allFiles = []
  
  #proc = subprocess.Popen(f"echo '  nice work, good job'", shell=True)
  #proc.wait()
  #proc.terminate()
  
  # ------------------- Count total log files --------------------
  
  proc = subprocess.Popen(f"ls {gStackFileGlob}", shell=True, stdout=PIPE, stderr=PIPE)
  for line in proc.stdout:
    # Convert from rec'd 'bytes' to 'string'
    line = line.decode().strip()
    allFiles.append(line)
  
  for line in proc.stderr:
    # Convert from rec'd 'bytes' to 'string'
    line = line.decode().strip()
    if "No such file or directory" in line:
      print()
      print(f"No files found with: '{gStackFileGlob}'")

  proc.wait()
  proc.terminate()

  totalCnt = len(allFiles)
  if totalCnt > 0:
    # 'strDict' needed for 'deleteLogFiles(strDict)'
    strDict = checkLogFileDetails(allFiles, totalCnt)

    # 18/3/24 DH: Lists are mutable (like dictionaries)
    xVals = []
    yVals = []
    noTrgFuncFileList = getNoTrgFuncTotals(xVals, yVals)
    # 20/3/24 DH: Realised at 2am that 'dicts' were "call by reference" so changes propagated outside functions...
    #             ...NOTICE: 'deepcopy()' REQUIRED. (https://docs.python.org/3/library/copy.html)
    tmpStrDict = copy.deepcopy(strDict)
    getTypesWithoutTrgFuncError(tmpStrDict, noTrgFuncFileList)

    recordLogFiles(strDict)
    #deleteLogFiles(strDict)

    outputStatementList()

    print()
    print(f"(NOT deleting files)")

    # 20/3/24 DH:
    graphSleeptimeDistrib(xVals, yVals)

