# 6/8/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys
import sqlite3

# TAKEN FROM: 'stop_trainer.py'
def getOrCreateTable():
  # ------------------------ DB CONNECTION -------------------------
  try:
    statsDB = sqlite3.connect(dbFQN, check_same_thread=False)
    print()
    print("Opened connection to DB: ", dbFQN)

  except sqlite3.OperationalError as e:
    print(e)
    return None
  
  # ------------------- 'gTableName' TABLE --------------
  try:
    cursor = statsDB.cursor()
    cursor.execute(
      # 8/8/24 DH: Added 'sample_seq' to store distrib of runs before correct answer: 
      #            "sqlite> alter table model_efficacy add sample_seq;"
      
      # FIELD NUMBERS:              0                             1               2           3           4
      f"CREATE TABLE {gTableName} (id INTEGER PRIMARY KEY, model_type_state, correct_num, sample_num, sample_seq)"
    )

    index_name = f"{gTableName}_id"
    cursor.execute(
      f"CREATE INDEX {index_name} ON {gTableName}(id)"
    )
    statsDB.commit()
    cursor.close()
    print(f"  Created table: '{gTableName}'")
    print(f"     with index: '{index_name}'")
    print()

  except sqlite3.OperationalError as e:
    if f"table {gTableName} already exists" in e.args:
      print(f"  Table: '{gTableName}' already exists")
    else:
      print(e)
  
  return statsDB

# 6/8/24 DH: Reorg of 'stop_trainer.py' done in Spring 2024...
def checkForModelRecord(cursor, stmnt, correctFieldNum, samplesFieldNum, sampleSeqFieldNum):
  try:
    cursor.execute(stmnt)
    # 'fetchall() returns an array of tuples for wanted fields
    result = cursor.fetchall()
    resultNum = len(result)
    if resultNum > 0:
      id = result[0][0]
      correctNum = result[0][correctFieldNum]
      samplesNum = result[0][samplesFieldNum]
      sampleSeq  = result[0][sampleSeqFieldNum]

      return (id, correctNum, samplesNum, sampleSeq)
    else:
      return (0, 0, 0, 0)

  except sqlite3.OperationalError as e:
    print(e)

def updateTableStats(statsDB):
  try:
    cursor = statsDB.cursor()
    
    # See 'stop_trainer.py::populateDB(...)'
    #stmnt = f"SELECT * FROM call_sequence WHERE training_function = '{trainingFunction}' AND seq_ids = '{seqIDs}'"
    stmnt = f"SELECT * FROM {gTableName} WHERE model_type_state = '{gModelTypeState}'"
    print(f"  {stmnt}")

    (id, correctNum, samplesNum, sampleSeq) = checkForModelRecord(cursor, stmnt, correctFieldNum=2, samplesFieldNum=3, sampleSeqFieldNum=4)

    updatedCorrectNum = int(correctNum) + int(gCorrectNum)
    updatedSamplesNum = int(samplesNum) + int(gSampleNum)

    # 8/8/24 DH:
    if sampleSeq:
      updatedSampleSeq  = f"{sampleSeq},{gSampleNum}"
    else:
      updatedSampleSeq  = f"{gSampleNum}"

    print()
    print(f"UPDATING 'correct_num': {updatedCorrectNum} (extra {gCorrectNum} from {correctNum})")
    print(f"UPDATING 'sample_num': {updatedSamplesNum} (extra {gSampleNum} from {samplesNum})")
    print(f"UPDATING 'sample_seq': {updatedSampleSeq}")

    if int(samplesNum) == 0:
      stmnt = f"INSERT INTO {gTableName} (model_type_state, correct_num, sample_num, sample_seq) VALUES (\
        '{gModelTypeState}', '{updatedCorrectNum}', '{updatedSamplesNum}', '{updatedSampleSeq}')"
    else:
      stmnt = f"UPDATE {gTableName} SET correct_num = {updatedCorrectNum}, sample_num = {updatedSamplesNum}, sample_seq = '{updatedSampleSeq}' WHERE id={id}"

    print(f"  {stmnt}")
    cursor.execute(stmnt)
    statsDB.commit()

    cursor.close()

  except sqlite3.OperationalError as e:
    print(e)
  
  except ValueError as e:
    print(f"Incorrect args: {sys.argv[3:]}")

# This is written to be called from BASH (eg 'get-model-output')
if __name__ == "__main__":
  if len(sys.argv) > 5:
    dbFQN = sys.argv[1]
    gTableName = sys.argv[2]
    gModelTypeState = sys.argv[3]
    gCorrectNum = sys.argv[4]
    gSampleNum = sys.argv[5]
  else:
    print(f"{sys.argv[0]}:")
    print("  INCORRECT cmd args, need <FQN of DB> + <Table name> + <Model-Type-State> + <Correct number> + <Total samples>")
    exit(1)
  
  statsDB = getOrCreateTable()
  updateTableStats(statsDB)

  # 7/8/24 DH: Neater print out with newline at end
  print()
