# 19/5/24 DH: I was hoping to do this in 'sqlite3' from 'sqlite> .tables'
#   via something like:
#     for table in $(.tables): SELECT * from $table where function like '%backward%';

import sqlite3, os, sys

# =========================================================================
# NON-FILLER version:
"""

cursor = sqlite3.connect(sys.argv[1], check_same_thread=False).cursor()
cursor.execute("SELECT name FROM sqlite_master where type='table';")
for tup in cursor.fetchall():
  tab = tup[0]  
  col = "function"
  searchStr = "backward"

  try:
    cursor.execute(f"SELECT * from {tab} where {col} like '%{searchStr}%';")
    for searchTup in cursor.fetchall():
      print(searchTup)
  except:
    pass

exit(0)
"""

# =========================================================================
def getDBconnection():
  # ------------------------ DB CONNECTION -------------------------
  # Taken from: 'graph-weights.py', 
  #             'stop_trainer.py'
  # ----------------------------------------------------------------
  if len(sys.argv) == 2:
    db_dir = os.path.abspath(sys.argv[1])
    db_file = sys.argv[1]
  else:
    print(f"You need to provide a DB")
    exit(0)

  try:
    traceDB = sqlite3.connect(db_file, check_same_thread=False)
    print(f"Opened connection to DB: '{db_file}'")
    print()
    cursor = traceDB.cursor()
    return cursor

  except sqlite3.OperationalError as e:
    print(f"OperationlError: '{e}'")

def getTables(cursor):
  tables = []

  try:
    # Get list of tables in DB
    # Same as: sqlite> .tables
    stmnt = "SELECT name FROM sqlite_master where type='table';"

    cursor.execute(stmnt)
    # 'fetchall() returns an array of tuples for wanted fields
    tabTups = cursor.fetchall()

    for tup in tabTups:
      tables.append(tup[0])

    return tables

  except sqlite3.OperationalError as e:
    print(f"OperationlError: '{e}'")

def searchTable(cursor, stmnt):
  try:
    cursor.execute(stmnt)
    # 'fetchall() returns an array of tuples for wanted fields
    results = cursor.fetchall()
    resultNum = len(results)
    if resultNum > 0:
      print("    Results:")
      print("    --------")
      for item in results:
        print(f"    {item}")
      print()

  except sqlite3.OperationalError as e:
    print(f"    OperationlError: '{e}'")

# =============================== MAIN =================================
if __name__ == "__main__":
  cursor = getDBconnection()
  tables = getTables(cursor)

  col = "function"
  #searchStr = "backward"
  searchStr = "dropout"

  print(f"SEARCHING for '{searchStr}', in COL: '{col}', from:")

  for tab in tables:
    print(f"  TABLE: '{tab}'")
      
    stmnt = f"SELECT * from {tab} where {col} like '%{searchStr}%';"
    #stmnt = f"SELECT * from {table} where {col} = '{searchStr}';"
    
    searchTable(cursor, stmnt)
   

