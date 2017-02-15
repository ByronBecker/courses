import numpy as np




x = np.arange(4)
#print(x)
x = np.transpose([x])
#print(x.shape)
y = np.array([5,0,0,2,4,0,0,3,0])
print(y)
print(y.shape)
print y[0]
#print np.dot(y, x)

#print np.multiply(y, 5)
test = [5*y for y in y[1:]]
test = np.insert(test, 0, 5)
print test




w = np.array([1, 2, 3, 1, 0])

#print np.concatenate([5], w)

z = np.empty

print(z)
z = np.append(z, 1)
print(z)
