####################################################################################
#
#                                      Huggin API
#
####################################################################################

# 17/6/24 DH: Now collate the various PNG's from matplotlib (https://www.graphviz.org/pdf/dotguide.pdf)
# https://www.graphviz.org/doc/info/shapes.html#images-example
# https://www.graphviz.org/doc/info/attrs.html
# https://graphviz.org/doc/info/colors.html
import os, sys
from datetime import datetime
import graphviz

gScriptDir = os.path.dirname(os.path.realpath(__file__))
gFilename = "qa-output.gv"

def getDotTxt(img1a, img1b, img1c, img2, img3, qcaTableTxt, contextLabel):
  # Escape the '{' with '{{'
  # Escape the '}' with '}}'
  # NOTE: Currently using hard-coded, scaled graphs

  # DUMP:
  #legend -> table1 [label="NOTE: This is Bert after 11 epochs (cf 1-10 epochs below)" fontsize=20 fontcolor="red"]

  dotTxt = f'''
strict digraph qaOutput {{
  graph [ordering=in rankdir=TB size="96.89999999999999,96.89999999999999"]
  node [align=left fontname="Linux libertine" fontsize=10 height=0.2 margin=0 ranksep=0.1 shape=plaintext style=filled]
  edge [penwidth=5.0]

  legend [label=<
    <TABLE BORDER="0" CELLBORDER="1"
    CELLSPACING="0" CELLPADDING="4">
      ['<TR><TD>11</TD><TD>Embedding(30522, 768, padding_idx=0)</TD></TR>', '<TR><TD>13</TD><TD>Embedding(2, 768)</TD></TR>', '<TR><TD>19</TD><TD>LayerNorm((768,), eps=1e-12, elementwise_affine=True)</TD></TR>', '<TR><TD>21</TD><TD>Dropout(p=0.1, inplace=False)</TD></TR>', '<TR><TD>[23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89]</TD><TD>BertAttention(<BR/>  (self): BertSelfAttention(<BR/>    (query): Linear(in_features=768, out_features=768, bias=True)<BR/>    (key): Linear(in_features=768, out_features=768, bias=True)<BR/>    (value): Linear(in_features=768, out_features=768, bias=True)<BR/>    (dropout): Dropout(p=0.1, inplace=False)<BR/>  )<BR/>  (output): BertSelfOutput(<BR/>    (dense): Linear(in_features=768, out_features=768, bias=True)<BR/>    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<BR/>    (dropout): Dropout(p=0.1, inplace=False)<BR/>  )<BR/>)</TD></TR>', '<TR><TD>[25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91]</TD><TD>BertIntermediate(<BR/>  (dense): Linear(in_features=768, out_features=3072, bias=True)<BR/>  (intermediate_act_fn): GELUActivation()<BR/>)</TD></TR>', '<TR><TD>[27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93]</TD><TD>BertOutput(<BR/>  (dense): Linear(in_features=3072, out_features=768, bias=True)<BR/>  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<BR/>  (dropout): Dropout(p=0.1, inplace=False)<BR/>)</TD></TR>', '<TR><TD>95</TD><TD>Linear(in_features=768, out_features=2, bias=True)</TD></TR>']
    </TABLE>
  > fillcolor=darkseagreen1]

  image2 [image="{img2}" label=""]
  image3 [image="{img3}" label=""]
  
  table1 [label=<
  <TABLE BORDER="1"  BGCOLOR="white">
    <TR>
      <TD BORDER="0"><IMG SRC="{img1a}"/></TD>
      <TD BORDER="0"><IMG SRC="{img1b}"/></TD>
      <TD BORDER="0"><IMG SRC="{img1c}"/></TD>
    </TR>
  </TABLE>
  >]

  qcaTable [label=<{qcaTableTxt}> fontsize=30 fontcolor="red" fillcolor="skyblue"]

  context [label="{contextLabel}" fontsize=30 fontcolor="red" fillcolor="skyblue"]

  image2 -> image3
  table1 -> image2
  table1 -> context [color="lightcoral" style="dashed"]
  context -> qcaTable [color="lightcoral" style="dashed" fontcolor="red" fontsize=50 label=<
                         <table border="0"><tr><td border="1">CONTEXT + QUESTION</td></tr></table>
                       >]
}}
'''
  return dotTxt

# 18/6/24 DH:
def get_qcaDict(gvDir):
  qcaDict = {}

  # Get log "TLD/gv-graphs/qca.log"
  try:
    qcaLog = f"{gvDir}/qca.log"
    with open(qcaLog) as source :
      textLines = [line.strip() for line in source.readlines() if line.strip()]
  except FileNotFoundError:
    print(f"Filename: {qcaLog} NOT FOUND")
    exit(0)
  
  for line in textLines:
    # EXAMPLE: 
    #    1-79: START INDEX=35
    #    1-79: END INDEX=37
    if ":" in line:
      # 18/6/24 DH: We need to handle context grammar like: "Argentina: The torch relay leg in Buenos Aires, ..."
      #             (which means that the "value" will be spread over several elements of 'lineSplit')
      #      https://docs.python.org/3/library/stdtypes.html#str.split (ie specify "maxsplit=1")
      lineSplit = line.split(": ", maxsplit=1)
      if len(lineSplit) > 1:
        # Key is common across many lines (but value is unique)
        key = lineSplit[0]
        value = lineSplit[1]

        # EXAMPLE: "START INDEX=35"
        valueSplit = value.split("=")
        if len(valueSplit) > 1:
          qaPart = valueSplit[0]
          qaValue = valueSplit[1]

          # 18/6/24 DH: Need to accom SQuAD entries like 'a giant banner reading "Free Tibet", and an alternative "human rights torch"'
          #             (so need to replace " with ')
          qaValue = qaValue.replace("\"", "'")

          # 'sort_error_logs.py::checkLogFileDetails(...)' handles dict of lists with "try...except KeyError"
          try:
            qcaDict[key][qaPart] = qaValue
          except KeyError:
            qcaDict[key] = {}
            qcaDict[key][qaPart] = qaValue
        else:
          print()
          print(f"ERROR WITH: '{value}'")
          print()
          print("(Handling context split over multiple lines)")
          # 'key' will be part of the context that contains a "<label>:" eg: "3-74: C=\n Turkey: The torch relay..."
          try:
            # https://docs.python.org/3/library/collections.html#ordereddict-objects
            #   "built-in dict class gained the ability to remember insertion order"
            key = list(qcaDict.keys())[-1]
            qcaDict[key]['C'] = line
          except KeyError:
            print("Things do not look good...exiting...")
            exit(0)

    else:
      print()
      print(f"ERROR, no ':' in: {line}")
      print()
  
  return qcaDict

def display_qcaDict(qcaDict):
  print()
  print("CONTENTS OF 'qcaDict':")
  print("-----------")
  for tldKey in qcaDict:
    print(f"'{tldKey}':")
    for valKey in qcaDict[tldKey]:
      print(f"  {valKey}: {qcaDict[tldKey][valKey]}")
    print()

def getQCATableTxt(gvDir):
  qcaDict = get_qcaDict(gvDir)
  display_qcaDict(qcaDict)

  # TODO: Use QCA entry matching the token length graphs (currently createDotFile(...): 'img3', 'img4')
  keyList = list(qcaDict.keys())
  firstKey = keyList[0]
  # Key contains "{sample number}-{sample token length}" eg "3-284"
  keySplit = firstKey.split("-")
  if len(keySplit) > 1:
    tokenLen = keySplit[1]

  """ Now sending the chosen token length entry back to match the correct graphs
  qcaDictKey = None
  for key in keyList:
    if str(graphTokenLen) in key:
      print(f"Yea baby, found '{graphTokenLen}' in '{key}'")
      qcaDictKey = key
      break
  if qcaDictKey:
    print(f"...yup still got '{qcaDictKey}' for '{graphTokenLen}'")
    print()
  """

  # DUMP:
  # ----
  #<TR><TD> Context: </TD><TD>{qcaDict[firstKey]['C']}</TD></TR>
  #<TR><TD> <font color="red">Context: </font></TD><TD width="50"><font color="red">{qcaDict[firstKey]['C']}</font></TD></TR>
  # -------------------------------------------------------------------------------------------------------------------------

  qcaTableTxt = f'''
  <TABLE BGCOLOR="skyblue">
    <TR><TD> Question: </TD><TD>{qcaDict[firstKey]['Q']}</TD></TR>
    
    <TR><TD> Answer: </TD><TD>{qcaDict[firstKey]['A']}</TD></TR>
    <TR><TD> Start index: </TD><TD>{qcaDict[firstKey]['START INDEX']}</TD></TR>
    <TR><TD> End index: </TD><TD>{qcaDict[firstKey]['END INDEX']}</TD></TR>
    <TR><TD> Token len: </TD><TD>{qcaDict[firstKey]['TOKEN LEN']}</TD></TR>
  </TABLE>
'''
  # Need to add newlines into context string
  import textwrap
  # 'wrap()' returns list are strings without newlines
  contextLabel = "\n".join(textwrap.wrap(qcaDict[firstKey]['C'], 64))

  return (qcaTableTxt, contextLabel, tokenLen)

def createDotFile(tld):
  gvDir = os.path.join(tld, "gv-graphs")

  # See 'checkpointing.py::archivePrevLogs(...)' for "today = datetime.today().strftime('%-d%b%-H%-M')"
  today = datetime.today().strftime('%-d%b')
  dotFile = f"qa-output-{today}.gv"
  dotFile = os.path.join(gvDir, dotFile)

  #img1 = "/Users/doug/src/ai/t5/graphs/loss-by-epoch-scaled.png"
  #img2 = "/Users/doug/src/ai/t5/graphs/bert_num_train_epochs=11-scaled.png"
  #
  #img3 = "/Users/doug/src/ai/t5/graphs/logits-by-token-by-epoch.png"

  oldDir = "old-logs/weights-graphs"
  h_utilsDir = "graphs"
  nodeGraphsDir = "gv-graphs"
  h_utilsWGDir = "weights/weights-graphs"

  img1a = f"{tld}/{oldDir}/0-fullValues.png"
  img1b = f"{tld}/{oldDir}/total-weight-change.png"

  # Prev work path (no longer active)
  #img1c = f"{tld}/../{h_utilsDir}/loss-by-epoch-scaled.png"
  img1c = f"{tld}/{h_utilsWGDir}/losses-by-epochs-1026.png"

  # 19/6/24 DH: Correlate graphs (from Q+Context token length) with key chosen from 'qcaDict' in 'getQCATableTxt(...)'
  (qcaTableTxt, contextLabel, graphTokenLen) = getQCATableTxt(gvDir)

  img2 = f"{tld}/{nodeGraphsDir}/all_layers-287-{graphTokenLen}.png"
  img3 = f"{tld}/{h_utilsDir}/logits-by-token-{graphTokenLen}.png"

  with open(dotFile, "w") as outFile:
    dotTxt = getDotTxt(img1a, img1b, img1c, img2, img3, qcaTableTxt, contextLabel)
    outFile.write(dotTxt)

  return dotFile

def createOutput(filename):
  dotFilePDF = None

  try:
    #fp = os.path.join(scriptDir, filename)
    graphviz.render('dot', 'pdf', filename).replace('\\', '/')
    dotFilePDF = f"{filename}.pdf"
    print(f"Created: '{dotFilePDF}'")
  except Exception as e:
    print(f"ERROR MSG:")
    print(f"  {e}")
  
  return dotFilePDF

if __name__ == "__main__":
  if len(sys.argv) > 1:
    tld = sys.argv[1]
  else:
    tld = gScriptDir
  
  print(f"Using:   '{tld}'")
  
  dotFile = createDotFile(tld)
  dotFilePDF = createOutput(dotFile)

  with open("gv_filename.txt", "w") as outFile:
    outFile.write(dotFilePDF)

  
