import csv

# 14/1/24 DH:
print("--------------------------------")
print("Using Transformers 'pipeline'")
print("--------------------------------")

# 14/1/24 DH: 'inputs' printed from 'pipeline.py' are 'BART-double-coded' values
# --------------------------------------------------------------------------------------------------------------

# 16/1/24 DH:
def createNumBeamsDict(filename):
  nbDict = {}
  with open(filename) as source :
    
    # LINE: "10", "As of 31 December, ..."

    # The use of 'skipinitialspace=True' makes 'csv.reader()' a Dialect..."knock 3 times and ask for Doug"
    # https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters
    parsedFile = csv.reader(source, skipinitialspace=True)
    for row in parsedFile:
      if row:
        #print("LINE: '{}', '{}'".format(row[0], row[1]))
        nbDict[row[0]] = row[1]
  
  return nbDict
# --------------------------------------------------------------------------------------------------------------

print()
print("Yup...well that's pretty much sorted then...")
print()
nBeamsDict = createNumBeamsDict("num_beams.txt")

for key in nBeamsDict:
  print("{}: '{}'".format(key, nBeamsDict[key]))

# 16/1/24 DH: ...ok, so now what ?


