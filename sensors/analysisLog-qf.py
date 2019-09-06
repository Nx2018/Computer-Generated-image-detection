import matplotlib.pyplot as plt
import numpy as np
from sys import argv


if (len(argv)<2):
    print "usage: python logAnalysis.py logFileName1 logFileName2 ...\n"
    exit()

str_dict={
    'Train net output #0: loss =':'',
    ' (* 1 =':' ',
    ' loss)':''
    }


def logFileAnalysis(logFileName):
    acc = []
    loss = []

    #logFile
    logfile = open(logFileName, 'r')
    if not logfile:
        exit()

    #process log file line by line

    #get accuracy
    lines = logfile.readlines()
    logfile.close()
    cnt = 0
    for line in lines:
        line = line.strip('\n').strip('\r')
        if line.find(" accuracy = ")<0:
            continue
        acc.append(float(line.split()[-1]))
        #print line
        cnt += 1
    del acc[-1]

    #get loss
    cnt = 0
    for line in lines:
        line = line.strip('\n').strip('\r')
        if line.find("Train net output")<0:
            continue
        #print line
        for key in str_dict:
            line = line.replace(key, str_dict[key])
        loss.append(float(line.split()[-1]))    
        cnt += 1
    return acc, loss


accLst = []
lossLst = []

for i in range(1, len(argv)):
    #print i
    accret, lossret = logFileAnalysis(argv[i])
    accLst.append(accret)
    lossLst.append(lossret)

#draw plot
#xaxis = range(0, 150000, 1500)
xaxis = range(len(accLst[0]))
#print xaxis, len(xaxis)
print len(xaxis), len(accLst[0]), len(lossLst[0])

#accLst[1] = accLst[1][:50]
#lossLst[1] = lossLst[1][:50]

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(xaxis, accLst[0], label='QualityFactor=95')
ax.plot(xaxis, accLst[1], label='QualityFactor=85')
ax.plot(xaxis, accLst[2], label='QualityFactor=75')

#ax.plot(xaxis, lossLst[0], label='QualityFactor=95')
#ax.plot(xaxis, lossLst[1], label='QualityFactor=85')
#ax.plot(xaxis, lossLst[2], label='QualityFactor=75')

ax.set(xlabel='Iteration (epoch)', ylabel='Validation Accuracy')
#ax.set(xlabel='Iteration (epoch)', ylabel='Training Loss')
#ax.set_title("Loss & Accuracy")
#ax.set_ylim([0.4, 1])
ax.legend(loc='center right')
#ax.legend(loc=(.72,.18))
#ax.grid()

#fig.savefig("qf-train-loss.eps", bbox_inches='tight')
fig.savefig("qf-validation-acc.eps", bbox_inches='tight')
plt.show()