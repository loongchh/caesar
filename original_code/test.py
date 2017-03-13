from collections import OrderedDict
grads_to_look=OrderedDict()
grads_to_look["Q_LSTM/RNN/LSTMCell"] = []
grads_to_look["P_LSTM/RNN/LSTMCell"] = []
grads_to_look["Match_LSTM_fwd/LSTMCell"] = []
grads_to_look["Match_LSTM_rev/LSTMCell"] = []
grads_to_look["ANSWER_POINTER/LSTMCell"] = []
grads_to_look["REST"] = []

for i,key in enumerate(grads_to_look):
    print i,key

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    gr = np.array([1,2,3,4,5])
    plt.title("Loss")
    plt.plot(np.arange(gr.size), gr.flatten(), label="Loss")
    plt.ylabel("Loss")
    pdf.savefig()
    plt.close()

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



