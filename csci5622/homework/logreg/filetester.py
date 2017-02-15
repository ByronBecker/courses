import numpy as np
import matplotlib.pyplot as plt

import csv, re


updates = []
tp = []
hp = []
ta = []
ha = []

f = open('0.01conv.txt', 'r+')

csv_f = csv.reader(f, delimiter=' ')

for row in csv_f:
    if row[0] == "Update":
        #print row
        updates.append(int(row[1].split('\t')[0]))
        #tp.append(float(row[2].split('\t')[0]))
        #hp.append(float(row[3].split('\t')[0]))
        ta.append(float(row[4].split('\t')[0]))
        ha.append(float(row[5]))



#print updates
#print tp
#print hp
#print ha

Hold, = plt.plot(updates, ha)
Test, = plt.plot(updates, ta, 'r')
plt.xlabel('Updates')
plt.ylabel('Accuracy')
plt.legend([Hold, Test], ['Holdout Set', 'Test Set'], loc=4)
plt.title('Updates vs. Accuracy with eta = 0.01')
plt.show()

