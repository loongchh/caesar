import numpy as np
# ========================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
    return aa

with PdfPages('multipage_pdf.pdf') as pdf:
    fig = plt.figure()
    fig.text(.1,.99," Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laborisnisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in")
    pdf.savefig()
    # Display 2 matrices of different sizes
    dimlist = [(12, 12), (15, 35)]
    for d in dimlist:
        plt.matshow(samplemat(d))

    pdf.savefig()
    # Display a random matrix with a specified figure number and a grayscale
    # colormap
    a= np.random.rand(64, 64)
    print a
    plt.matshow(a, fignum=100, cmap=plt.cm.gray)

    pdf.savefig()
    plt.close()

exit()

# ========================================================================
import tensorflow as tf

L = np.random.rand(10,100,300)
W = np.random.rand(300,300)
with tf.Session() as session:
    a = session.run(tf.map_fn(lambda x: tf.matmul(x, W), L))
    b = session.run(tf.reshape(tf.matmul(tf.reshape(L, [-1, 300]), W), [-1, 100, 300]))
    assert np.all(a==b)
# ========================================================================
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
# ========================================================================
exit()
# ========================================================================
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

# ========================================================================
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
# ========================================================================
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



