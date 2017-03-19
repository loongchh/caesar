from collections import OrderedDict
import numpy as np
import tensorflow as tf
L = np.random.rand(10,100,300)
W = np.random.rand(300,300)
with tf.Session() as session:
    a = session.run(tf.map_fn(lambda x: tf.matmul(x, W), L))
    b = session.run(tf.reshape(tf.matmul(tf.reshape(L, [-1, 300]), W), [-1, 100, 300]))
    assert np.all(a==b)

L = [
        [
            [1.,2.,3.],
            [3.,4.,5.]
        ],
        [
            [10.,12.,13.],
            [3.,4.,5.]
        ],
]
L = tf.constant(np.array(L))

with tf.Session() as session:
    L_shape = session.run(tf.shape(L))
    print(L_shape)

    A_q1 = session.run(tf.nn.softmax(L, dim=0))
    print (A_q1)

    A_q2 = session.run(tf.nn.softmax(L))
    print (A_q2)

exit()

grads_to_look=OrderedDict()
grads_to_look["Q_LSTM/RNN/LSTMCell"] = []
grads_to_look["P_LSTM/RNN/LSTMCell"] = []
grads_to_look["Match_LSTM_fwd/LSTMCell"] = []
grads_to_look["Match_LSTM_rev/LSTMCell"] = []
grads_to_look["ANSWER_POINTER/LSTMCell"] = []
grads_to_look["REST"] = []

for i,key in enumerate(grads_to_look):
    print i,key


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



