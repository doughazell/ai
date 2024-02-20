import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd

# 25/5/23 DH: Ctrl-C while in matplotlib loop
import sys
import signal
#signal.signal(signal.SIGINT, signal.SIG_DFL)

# dist is the distribution of birthdays. If it is not known it is assumed uniform
def collect_until_repeat(repeatNumExceed, dist = np.ones(365)/365):
    
  # We'll use this variable as the stopping condition
  repeat = False

  # We'll store the birthdays in this array
  outcomes = np.zeros(dist.shape[0])
  
  # n will count the number of people we meet. Each loop is a new person.
  n = 0
  while not repeat:
    # add one to the counter
    n += 1
           
    # simulate adding a person with a "random birthday" ie a 365 array with {1*1 + 364*0}
    # this {1+364} array is added to 'outcomes' 365 array
    outcomes += np.random.multinomial(1,dist)
    #if n < 3:
    #  print(outcomes)
    
    # check if we got a repeat
    if np.any(outcomes > repeatNumExceed):
      repeat = True
 
  return n

def run_many_simulations(repeatNumExceed, sim, dist = np.ones(365)/365):
  # count stores the result of each simulation in a big array
  count = np.zeros(sim)
  
  printNum = 0
  for idx_sim in range(sim):
    count[idx_sim] = collect_until_repeat(repeatNumExceed, dist)

    if printNum != idx_sim and idx_sim % 1000 == 0:
      print("Sim total:",idx_sim)
      printNum = idx_sim
  
  return count

def printHist(hist, trials, bins, timeout=1):
  plt.clf()
  
  if trials > 90000:
    xAxis = 1200
  else:
    xAxis = 1000

  randHist = plt.hist(hist, bins = np.arange(0,xAxis))
  plt.title("Random histogram after "+str(trials)+" trials on "+str(bins)+" bins")
  plt.xlabel("$n$")
  plt.ylabel("Number of occurences")
  
  #plt.show()
  plt.draw()
  plt.waitforbuttonpress(timeout=timeout)

# 25/5/23 DH:
def getRandomHist(bins,sim,printout=False):
  outcomes = np.zeros(bins)

  printNum = 0
  for num in range(sim):
    outcomes += np.random.multinomial(1, np.ones(bins)/bins )

    if printNum != num and num % 1000 == 0 and printout:
      printNum = num

      print(outcomes)
      printHist(outcomes, num, bins)

  return outcomes

def printRandomHist():
  bins = 100
  sim = 5000
  hist = getRandomHist(bins, sim, printout=True)

  print("Printing final",sim,"trial")
  print("\nClick on graph to continue with analysis")
  print("(otherwise it will hang and you will need: 'ps -ef|grep python ; kill -9 <pid>')\n")
  printHist(hist,sim, bins, timeout=-1)

def signal_handler(sig, frame):
    print('\nYou pressed Ctrl+C...')
    
    sys.exit(0)

# =========================== MAIN ==============================
signal.signal(signal.SIGINT, signal_handler)

printRandomHist()

#sim = 1000000
sim = 10000
repeatNumExceed = 1
print("Running",sim,"simulations...")
counts = run_many_simulations(repeatNumExceed,sim)

# 24/8/23 DH: Added to debug 'lime_image.py' binomial distrib 'plt.hist' display of blankness...
#             ...when use bin number rather than bin values...!!!
#print("counts (",type(counts),",",counts.shape,"):",counts)

plt.clf()

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
repeat_histogram = plt.hist(counts, bins = np.arange(0,100))

# 4/8/23 DH: User runtime instructions
print("\nPrinting distrib of "+str(repeatNumExceed+1)+" collisions with 365 bins", end='')
print("...then prob of collision with total pairs")
print("(total pairs = total people * (total people - 1) / 2)")
print("\nClose graph window to continue with analysis")

# 25/5/23 DH: Reset the x-axis values set above
plt.xlim(xmin=0, xmax=100)
plt.title("The number of additions before "+str(repeatNumExceed+1)+" repeats with 365 bins")
plt.xlabel("$n$")
plt.ylabel("Number of occurences")

plt.draw()
plt.show()
#plt.waitforbuttonpress(timeout=-1)

probFree = [
  1,
  0.9973,
  0.9918,
  0.9837,
  0.9729,
  0.9597,
  0.9440,
  0.9261,
  0.9060,
  0.8839,
  0.8599,
  0.8344,
  0.8074,
  0.7791,
  0.7497,
  0.7195,
  0.6886,
  0.6572,
  0.6255,
  0.5938,
  0.5621,
  0.5306,
  0.4995,
  0.4690, # 24
]

pairs = [
  1,
  3,
  6,
  10,
  15,
  21,
  28,
  36,
  45,
  55,
  66,
  78,
  91,
  105,
  120,
  136,
  153,
  171,
  190,
  210,
  231,
  253,
  276,
  300, # 24
]

"""
  325,
  351,
  378,
  406,
  435,
  465,
  496,
  528,
  561,
  595,
  630,
  666,
  703,
  741,
  780,
  820,
"""

# 19/2/24 DH: Display of last graph of prob of collision vs number of pairs
# -------------------------------------------------------------------------
plt.plot(pairs, probFree)
plt.ylim(ymin=0, ymax=1.2)
plt.title("Birthday Paradox")
plt.xlabel("Number of pairs")
plt.ylabel("Prob of free cell")

plt.draw()
plt.show()

"""
print("2 people for a repeat occurred {} times, which is relatively {:.4%}".format(
  repeat_histogram[0][2], repeat_histogram[0][2]/sim
))
print("1/365 = 0.0027")
"""

rel_dist = plt.hist(counts, bins = np.arange(0,100), density=True)

print("50% of time, no more than {} people were needed for a repeat.".format(
  np.where(np.cumsum(rel_dist[0])>0.5)[0][0]
  )
)

