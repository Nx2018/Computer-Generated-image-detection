import numpy as np
from sys import argv


if (len(argv)<2):
    print "usage: python analysisTest.py lstFileName resultFileName\n"
    exit()

str_dict={
    'Real/':'',
    'CGG/':'',
    '.bmp':'',
    '#':' '
    }
testImages = []
pgFiles = []
cgFiles = []

#listFile
lstfile = open(argv[1], 'r')
if not lstfile:
    exit()

#process list file line by line
lines = lstfile.readlines()
lstfile.close()
for line in lines:
    line = line.strip('\n').strip('\r')
    for key in str_dict:
        line = line.replace(key, str_dict[key])
    linelst = line.split()
    fname, blkid, flag = linelst[0], linelst[1], linelst[2]
    #print fname, blkid, flag #, raw_input()

    testImages.append([fname, int(blkid), int(flag)])

#logFile
logfile = open(argv[2], 'r')
if not logfile:
    exit()

#process log file line by line
lines = logfile.readlines()
logfile.close()
cnt = 0
for line in lines:
    line = line.strip('\n').strip('\r')
    if line.find(", accuracy = ")<0:
        continue
    testImages[cnt].append(int(line.split()[-1]))
    cnt += 1
    if cnt>=44283:
        break
testImages.sort()


#process each line/block of testImages
preIdx = testImages[0][0]
framelabel1 = 0  #set by list file
framelabel2 = 0  #get by computation
ftype = " "
blkcnt = 0
for line in testImages:
    blkid, blklabel, blkaccuracy = line[1], line[2], line[3]
    curIdx = line[0]
    if (preIdx == curIdx):
        framelabel2 += (blklabel if blkaccuracy else (blklabel+1)%2)
        framelabel1 += blklabel
        ftype = "pg" if blklabel else "cg"
        blkcnt += 1
    else:        
        framelabel2 = 1 if framelabel2>blkcnt//2 else 0
        framelabel1 = 1 if framelabel1>0 else 0
        frameaccuracy = 1 if framelabel1==framelabel2 else 0
        if ftype=="pg":
            pgFiles.append([preIdx, framelabel1, framelabel2, frameaccuracy])
        elif ftype=="cg":
            cgFiles.append([preIdx, framelabel1, framelabel2, frameaccuracy])
        framelabel2 = (blklabel if blkaccuracy else (blklabel+1)%2)
        framelabel1 = blklabel
        blkcnt = 1
    preIdx = curIdx


#computation for accuracy
TP = FP = FN = TN = 0
cnt1 = 0
accuracy1 = 0
for i in cgFiles:
    #print i
    label = i[1]
    result = i[2]
    TP = TP + (1 if label==1 and result==1 else 0)
    FP = FP + (1 if label==0 and result==1 else 0)
    FN = FN + (1 if label==1 and result==0 else 0)
    TN = TN + (1 if label==0 and result==0 else 0)

    cnt1 += 1
    accuracy1 += i[3]
print "CG file accuracy1: %.2f" % (accuracy1*1.0/cnt1*100)

cnt2 = 0
accuracy2 = 0
for i in pgFiles:
    #print i
    label = i[1]
    result = i[2]
    TP = TP + (1 if label==1 and result==1 else 0)
    FP = FP + (1 if label==0 and result==1 else 0)
    FN = FN + (1 if label==1 and result==0 else 0)
    TN = TN + (1 if label==0 and result==0 else 0)

    cnt2 += 1
    accuracy2 += i[3]
print "pg file accuracy2: %.2f" % (accuracy2*1.0/cnt2*100)
print "total file accuracy: %.2f" % ((accuracy1+accuracy2)*1.0/(cnt1+cnt2)*100)

print TP+TN+FP+FN, cnt1, cnt2, len(cgFiles), len(pgFiles)
precision = (TP)*1.0/(TP+FP)
recall = (TP)*1.0/(TP+FN)
print "precision: %.2f" % (precision*100)
print "recall: %.2f" % (recall*100)
print "F1: %.2f" % (2*(precision*recall)/(precision+recall)*100)

