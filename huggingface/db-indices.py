# 12/8/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

import sys
import db_utils

# 12/8/24 DH: Don't need to check for existing record like 'db_utils.updateTableStats(...)' since each sequence is unique
def insertSequenceRecord(statsDB):
  try:
    cursor = statsDB.cursor()

    stmnt = f"INSERT INTO {gTableName} (model_efficacy_id, seq_num, seq_ids) VALUES (\
'{gEfficacyID}', '{gSampleNum}', '{gIdxs}')"

    print(f"  {stmnt}")
    cursor.execute(stmnt)
    statsDB.commit()

    cursor.close()
  except db_utils.sqlite3.OperationalError as e:
    print(e)
  
  except ValueError as e:
    print(f"Incorrect args: {sys.argv}")

# This is written to be called from BASH (eg 'get-model-output')
if __name__ == "__main__":
  if len(sys.argv) > 5:
    gDbFQN = sys.argv[1]
    gTableName = sys.argv[2]
    gEfficacyID = sys.argv[3]
    gSampleNum = sys.argv[4]
    gIdxs = sys.argv[5]
  else:
    print(f"{sys.argv[0]}:")
    print("  INCORRECT cmd args, need <FQN of DB> + <Table name> + <Efficacy record ID> + <Total samples> + <Indices string>")
    exit(1)
  
  # 'model_efficacy' columns
  #columnsStr = "model_type_state, correct_num, sample_num, sample_seq"

  columnsStr = "model_efficacy_id, seq_num, seq_ids"
  statsDB = db_utils.getOrCreateTable(gDbFQN, gTableName, columnsStr)

  insertSequenceRecord(statsDB)

  # Newline for output neatness
  print()