


import csv

f = open('../data/spoilers/train.csv', 'r+')
w = open('truepredict.csv', 'w')

writer = csv.writer(w)

csv_f = csv.reader(f, delimiter=',')

csv_f.next()
i = 0
writer.writerow(['Id,Spoiler'])
for row in csv_f:
    if row[1] == 'True':
        writer.writerow([i,'True'])
    else:
        writer.writerow([i,'False'])
    i = i+1

