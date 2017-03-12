import numpy as np
a = np.array([0,0])
for d in a:
    print(d)
print np.sum([1, 1])
print np.all(a==0)
a=1
b=2
print a,b
a,b= (b,a)
print a,b

for j in range(2,5):
    print j
def test(a,b,c,d):
    print b
    b = [1]
b = (1,2,3,4)
test(*b)
print b


a=[
    [
        [1,2],
        [2,3]
    ],
    [
        [10.11],
        [12,13]
    ]
]

print a[:1:]

print ("d",4,"3")[1]

print "abc"[1:]



