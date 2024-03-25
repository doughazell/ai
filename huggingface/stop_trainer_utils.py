# 20/3/24 DH:
##############################################################################
#
#                                HugginAPI
#
##############################################################################

# https://matplotlib.org/stable/api/pyplot_summary.html
import matplotlib.pyplot as plt
from scipy import stats

# 20/3/24 DH:
def displayIntervals(intervalLog):
  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure() # 'plt.figure()' is ONLY NEEDED for subsequent graphs

  plt.title("Distribution of sleep intervals from a random number")
  plt.xlabel("Sleep Times")
  plt.ylabel("Number of occurances")
  plt.ylim(ymin=50, ymax=200)

  xVals = range(len(intervalLog))
  yVals = intervalLog
  plt.plot(xVals, yVals, label="Distribution")

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
  # "*_" unpacks remaining vals (rvalue, pvalue, stderr, intercept_stderr)
  #   VARIANT OF "*args"  https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists
  # m, b, *_ = stats.linregress(xVals, yVals)
  m, b, rvalue, stderr, intercept_stderr = stats.linregress(xVals, yVals)

  mRound = round(m, 3)
  bRound = round(b, 1)
  rvalue = round(rvalue, 2) # Usual tomfoolery with decimal place determinism...
  stderr = round(stderr, 2)
  intercept_stderr = round(intercept_stderr, 2)

  mTxt = "Grad : "
  bTxt = "Intercept : "
  rvTxt = "RValue : "
  sdTxt = "Slope StdErr : "
  inTxt = "Intercept StdErr : "
  legendStr = f"{mTxt:>24} {mRound}\n{bTxt:>22} {bRound}\n{rvTxt:>23} {rvalue}\n{sdTxt:>20} {stderr}\n{inTxt} {intercept_stderr}"
  plt.figtext(0.2, 0.2, legendStr)

  plt.axline(xy1=(0, b), slope=m, color='green', linestyle='dashed', linewidth=0.5)

  #plt.draw()
  #plt.show()
  plt.show(block=False)

# 20/3/24 DH: Taken from 'qa_lime.py::graphTokenVals(startVals, endVals)'
def graphSleeptimeDistrib(xVals, yVals):
  #plt.plot(range(len(startVals)), startVals, label="Start logits")
  #plt.legend(loc="upper left")
  #plt.ylim(ymin=0, ymax=50)
  #plt.axhline(y=0, color='green', linestyle='dashed', linewidth=0.5)

  # 4/9/23 DH: Display all graphs simultaneously with 'plt.show(block=False)' (which needs to be cascaded)
  plt.figure()

  plt.title("Distribution of intervals giving no training function necessary for DB")
  plt.xlabel("Times")
  plt.ylabel("Number of files")

  plt.plot(xVals, yVals, label="Time distribution")

  #plt.draw()
  plt.show()

