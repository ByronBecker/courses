import numpy as np
from numpy import array, median
from operator import itemgetter


a = [1, 2]
np.asarray(a)
print(a)
b = np.array([1, 2, 3.0])
print(b)

c = np.array(np.mat('1 2; 3 4'))
print(c)

d = np.array([[1, 2], [3, 4]])
print(d)

x = array([[1., 1.],[5, 3]])
print(x)

np.sort(x)
print(x)
print("here")

y = np.array([+1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1])

for label in np.nditer(x):
    print x

a = np.arange(6).reshape(2,3)
print(a)

'''
count = {}
for label in y:
    #print(label)
    if label in count:
        count[label] += 1
    else:
        count[label] = 1

common = max(count, key = lambda i: y[i])

print(count)
print(common)

countnew = sorted(count.items(), key = lambda x: x[1],  reverse = True) 

print(countnew)

countaswell = sorted(count.items(), key = itemgetter(1),  reverse = True) 

print(countaswell)
print(len(countaswell))
'''

