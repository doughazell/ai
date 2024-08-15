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
from pathlib import Path

gScriptDir = os.path.dirname(os.path.realpath(__file__))
# 9/7/24 DH: Path refactor so can be run from any dir
gCWD = Path.cwd()

# 15/8/24 DH: SQUAD training has 'img1d' + 'img1e' as None since same sample from 88524 is not tracked over 2 epochs
def getDotTxtJSON(img1a, img1b, img1c, img1d, img1e):
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

  table1 [label=<
  <TABLE BORDER="1"  BGCOLOR="white">
    <TR>
      <TD BORDER="0"><IMG SRC="{img1a}"/></TD>
      <TD BORDER="0"><IMG SRC="{img1b}"/></TD>
      <TD BORDER="0"><IMG SRC="{img1c}"/></TD>
    </TR>
  </TABLE>
  >]

  image1d [image="{img1d}" label=""]
  image1e [image="{img1e}" label=""]
  
  table1 -> image1d
  table1 -> image1e
  
}}
'''
  return dotTxt

# 15/8/24 DH: SQUAD training has 'img1d' + 'img1e' as None since same sample from 88524 is not tracked over 2 epochs
def getDotTxtSQUAD(img1a, img1b, img1c):
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

  table1 [label=<
  <TABLE BORDER="1"  BGCOLOR="white">
    <TR>
      <TD BORDER="0"><IMG SRC="{img1a}"/></TD>
      <TD BORDER="0"><IMG SRC="{img1b}"/></TD>
      <TD BORDER="0"><IMG SRC="{img1c}"/></TD>
    </TR>
  </TABLE>
  >]
  
}}
'''
  return dotTxt


def createDotFile(hfDir):
  gvDir = os.path.join(gCWD, "gv-graphs")
  print(f"Using 'gvDir':   '{gvDir}' (in 'createDotFile()')")
  print()

  # See 'checkpointing.py::archivePrevLogs(...)' for "today = datetime.today().strftime('%-d%b%-H%-M')"
  today = datetime.today().strftime('%-d%b')
  dotFile = f"qa-training-{today}.gv"
  dotFile = os.path.join(gvDir, dotFile)
  
  h_utilsWGDir = "weights/weights-graphs"

  img1a = f"{gCWD}/{hfDir}/{h_utilsWGDir}/0-fullValues.png"
  img1b = f"{gCWD}/{hfDir}/{h_utilsWGDir}/total-weight-change.png"
  img1c = f"{gCWD}/{hfDir}/{h_utilsWGDir}/losses-by-epochs.png"

  try:
    lossesFilenameFile = f"{gCWD}/gv-graphs/losses_filename.txt"
    with open(lossesFilenameFile) as inFile:
      lossesFilename = inFile.readline()

    img1d = f"{gCWD}/gv-graphs/logits-by-epoch.png"
    img1e = f"{gCWD}/gv-graphs/{lossesFilename}"
  
  # 11/8/24 DH: Handling when SQUAD trg does NOT PROVIDE 'input_ids' etc in 'seq2seq_qa_INtrainer.log' due to same sample from (88524 *2) NOT TRACKED
  #   ( handled by 'graph-logits.py::pruneLogits(...)' prior to 'graphLosses(...)' producing 'losses_filename.txt' )
  except FileNotFoundError:
    print(f"Filename: {lossesFilenameFile} NOT FOUND")
    print("This is probably an artifact of SQUAD trg in 'huggin_utils::logLogits(...)'")
    
    img1d = None
    img1e = None

  print("USING")
  print("-----")
  print(f"Full weights graph:         {img1a}")
  print(f"Weight change graph:        {img1b}")
  print(f"Loss from all epochs graph: {img1c}")
  print()
  print(f"  Logits by epoch graph:       {img1d}")
  print(f"  Loss by sample epochs graph: {img1e}")
  print()

  with open(dotFile, "w") as outFile:
    if img1d == None:
      dotTxt = getDotTxtSQUAD(img1a, img1b, img1c)
    else:
      dotTxt = getDotTxtJSON(img1a, img1b, img1c, img1d, img1e)
    
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
    hfDir = sys.argv[1]
  else:
    print("INCORRECT cmd args, need 'JSON::output_dir'")
    exit(1)
  
  dotFile = createDotFile(hfDir)
  dotFilePDF = createOutput(dotFile)

  # Save path of 'pdf' for it to be opened by 'get-training-output'
  with open("gvTrain_filename.txt", "w") as outFile:
    outFile.write(dotFilePDF)

  
