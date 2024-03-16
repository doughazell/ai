# 16/3/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import subprocess, time, os
from subprocess import Popen, PIPE, STDOUT

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
]

gStackFileGlob = "stack-2024*"

def deleteLogFiles(strDict):
  print()
  print("Removing:")
  for key in strDict:
    print(f"'  {key}':")
    for elem in strDict[key]:
      print(f"    {elem}")
      os.remove(elem)

def checkLogFileDetails(allFiles, totalCnt):
  subTotal = 0
  strDict = {}

  # ------------------- Cycle through each type of file ----------
  cnt = 0
  gListLen = len(gStringList)
  for searchStr in gStringList:
    cnt += 1
    print(f"{cnt}/{gListLen}) '{searchStr}'")
    print(f"-----")

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

    print(f"  {searchStrNum}")
    print()
    subTotal += searchStrNum

    proc.wait()
    proc.terminate()
  # END: --- for searchStr in gStringList ---
  
  totalRemaining = totalCnt - subTotal
  print(f"Total: {totalCnt}, Sub total: {subTotal}, Total remaining: {totalRemaining}")
  if totalRemaining > 0:
    print(f"Total files remaining:")
    for logFile in allFiles:
      print(f"  {logFile}")


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
    checkLogFileDetails(allFiles, totalCnt)

  print(f"(Not deleting files)")
  #deleteLogFiles(strDict)
