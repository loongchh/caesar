import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gr = np.array([1,2,3,4,5])

plt.title("Loss")
plt.plot(np.arange(gr.size), gr.flatten(), label="Loss")
plt.ylabel("Loss")
output_path = "../plots/train.png"
plt.savefig(output_path)

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



