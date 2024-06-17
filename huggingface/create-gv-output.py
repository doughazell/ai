####################################################################################
#
#                                      Huggin API
#
####################################################################################

# 17/6/24 DH: Now collate the various PNG's from matplotlib (https://www.graphviz.org/pdf/dotguide.pdf)
# https://www.graphviz.org/doc/info/shapes.html#images-example
# https://www.graphviz.org/doc/info/attrs.html
import os, sys
from datetime import datetime
import graphviz

gScriptDir = os.path.dirname(os.path.realpath(__file__))
gFilename = "qa-output.gv"

def getDotTxt(img1, img2, img3):
  # Escape the '{' with '{{'
  # Escape the '}' with '}}'
  # NOTE: Currently using hard-coded, scaled graphs
  dotTxt = f'''
strict digraph qaOutput {{
  graph [ordering=in rankdir=TB size="96.89999999999999,96.89999999999999"]
  node [align=left fontname="Linux libertine" fontsize=10 height=0.2 margin=0 ranksep=0.1 shape=plaintext style=filled]

  legend [label=<
    <TABLE BORDER="0" CELLBORDER="1"
    CELLSPACING="0" CELLPADDING="4">
      ['<TR><TD>11</TD><TD>Embedding(30522, 768, padding_idx=0)</TD></TR>', '<TR><TD>13</TD><TD>Embedding(2, 768)</TD></TR>', '<TR><TD>19</TD><TD>LayerNorm((768,), eps=1e-12, elementwise_affine=True)</TD></TR>', '<TR><TD>21</TD><TD>Dropout(p=0.1, inplace=False)</TD></TR>', '<TR><TD>[23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89]</TD><TD>BertAttention(<BR/>  (self): BertSelfAttention(<BR/>    (query): Linear(in_features=768, out_features=768, bias=True)<BR/>    (key): Linear(in_features=768, out_features=768, bias=True)<BR/>    (value): Linear(in_features=768, out_features=768, bias=True)<BR/>    (dropout): Dropout(p=0.1, inplace=False)<BR/>  )<BR/>  (output): BertSelfOutput(<BR/>    (dense): Linear(in_features=768, out_features=768, bias=True)<BR/>    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<BR/>    (dropout): Dropout(p=0.1, inplace=False)<BR/>  )<BR/>)</TD></TR>', '<TR><TD>[25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91]</TD><TD>BertIntermediate(<BR/>  (dense): Linear(in_features=768, out_features=3072, bias=True)<BR/>  (intermediate_act_fn): GELUActivation()<BR/>)</TD></TR>', '<TR><TD>[27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93]</TD><TD>BertOutput(<BR/>  (dense): Linear(in_features=3072, out_features=768, bias=True)<BR/>  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)<BR/>  (dropout): Dropout(p=0.1, inplace=False)<BR/>)</TD></TR>', '<TR><TD>95</TD><TD>Linear(in_features=768, out_features=2, bias=True)</TD></TR>']
    </TABLE>
  > fillcolor=darkseagreen1]

  layoutTable [label=<
  <TABLE BGCOLOR="skyblue">
    <TR><TD> ??? </TD></TR>
    <TR><TD> Now what??? </TD></TR>
  </TABLE>
  >]

  image2 [image="{img1}" label=""]
  
  table1 [label=<
  <TABLE BORDER="1"  BGCOLOR="white">
    <TR>
      <TD BORDER="0"><IMG SRC="{img2}"/></TD>
      <TD BORDER="0"><IMG SRC="{img3}"/></TD>
    </TR>
  </TABLE>
  >]

  legend -> table1 [label="NOTE: This is Bert after 11 epochs (cf 1-10 epochs below)" fontsize=20 fontcolor="red"]
  legend -> layoutTable [color="lightcoral" style="dashed"]
  table1 -> image2
}}
'''
  return dotTxt

def createDotFile(gvDir):
  # See 'checkpointing.py::archivePrevLogs(...)' for "today = datetime.today().strftime('%-d%b%-H%-M')"
  today = datetime.today().strftime('%-d%b')
  dotFile = f"qa-output-{today}.gv"
  dotFile = os.path.join(gvDir, dotFile)

  img1 = "/Users/doug/src/ai/t5/graphs/logits-by-token-by-epoch.png"
  img2 = "/Users/doug/src/ai/t5/graphs/loss-by-epoch-scaled.png"
  img3 = "/Users/doug/src/ai/t5/graphs/bert_num_train_epochs=11-scaled.png"

  with open(dotFile, "w") as outFile:
    dotTxt = getDotTxt(img1, img2, img3)
    outFile.write(dotTxt)

  return dotFile

def createOutput(filename):
  try:
    #fp = os.path.join(scriptDir, filename)
    graphviz.render('dot', 'pdf', filename).replace('\\', '/')
    print(f"Created: '{filename}.pdf'")
  except Exception as e:
    print(f"ERROR MSG:")
    print(f"  {e}")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    gvDir = sys.argv[1]
  else:
    gvDir = gScriptDir
  
  print(f"Using:   '{gvDir}'")
  
  dotFile = createDotFile(gvDir)
  createOutput(dotFile)
  
