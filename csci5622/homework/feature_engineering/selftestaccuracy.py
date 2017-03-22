

import csv

    
my_predict = open('predictionstest.csv', 'r')
truth = open('truepredict.csv', 'r')

print(my_predict.readline())
print(truth.readline())


score = 0
total = 0
for line in my_predict:
    elem = line.split(',')
    el = truth.readline().split(',')
    if elem[1] == el[1]:
        print("match of " + str(elem[1]) + " " + str(el[1]))
        score += 1
    total +=1



print ("score is " + str(score))
print ("total is " + str(total))
percent = float(score)/total

print("Training Accuracy is: " + str(percent))
