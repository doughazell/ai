import csv

# 22/1/24 DH:
print("--------------------------------")
print("Using Transformers 'TODO'")
print("--------------------------------")

# --------------------------------------------------------------------------------------------------------------

# 16/1/24 DH:
def createNumBeamsDict(filename):
  nbDict = {}
  with open(filename) as source :
    
    """
    "1", "As of 31 December, the UK government has banned the breeding, selling or abandonment of the dogs. Additional measures will make it a criminal offence to own an XL bully without an exemption certificate. The certificate involves paying a £92.40 application fee and dogs must also be kept on a lead and muzzled in public."
    ...
    "10", "As of 31 December, the UK government has banned the breeding, selling or abandonment of the dogs in England. Additional measures will make it a criminal offence to own an XL bully without an exemption certificate."
    """

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
"""
OUTPUT TOKENS:
--------------
<1> [As of 31 December, the UK government has banned the breeding, selling or abandonment of the dogs] 
<2> [Additional measures will make it a criminal offence to own an XL bully]
<3> [without an exemption certificate]
<4> [The certificate involves paying a £92.40 application fee and dogs must also be kept on a lead and muzzled in public.]
---
<5> [<1>"."<2><3>"."]
<6> [<1>"in England."<2><3>"."]
<7> [<1>"in England."<2>]

NUM_BEAMS (atomised):
---------------------
1)  <1>"."<2><3>"."<4>
    <5><4>
2)  <1>"in England."<2><3>"in England and Wales."
    <7><3>"in England and Wales."
3)  <1>"in England."<2>"."
    <7>"."
4)  <1>"."<2><3>"."           
    <5>
5)  <1>"in England."<2><3>"."
    <6>
6)  <1>"."<2><3>"."
    <5>
7)  <1>"in England."<2><3>"."
    <6>
8)  <1>"."<2><3>"."
    <5>
9)  <1>"."<2><3>"."
    <5>
10) <1>"in England."<2><3>"."
    <6>

Text2TextGenerationPipeline - <class 'transformers.models.bart.tokenization_bart_fast.BartTokenizerFast'>
"""

