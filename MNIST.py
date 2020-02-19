

import keras 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import math

mnist_data = input_data.read_data_sets("MNIST_data/",one_hot=True)

x,y = mnist_data.train.next_batch(1)
def imformat(x):
  horl = int(math.sqrt(len(x)))
  verl = horl
  return np.reshape(x,(horl,verl))

plt.imshow(imformat(x[0]),cmap="gray")
print(np.argmax(y))
print(mnist_data.train.num_examples)

print(np.amax(x),np.amin(x))

"""###Tensorboard"""

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir=LOG_DIR, histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)

"""###Network"""

from keras.layers import Conv2D, BatchNormalization, Dropout, Input, Dense , Flatten

inp = Input(shape=(28,28,1))

y1 = Conv2D(filters=64,activation="relu",kernel_size=(5,5))(inp)

y2 = Conv2D(filters=128,activation="relu",kernel_size=(5,5), padding="same", kernel_initializer="glorot_uniform")(y1)

y2_flatten = Flatten()(y2)

pred = Dense(10,activation="softmax")(y2_flatten)

from keras.models import Model
from keras.optimizers import Adam

opt = Adam(lr = 0.001)

convNet = Model(inputs = inp, outputs = pred)
convNet.compile(optimizer = opt,loss = "categorical_crossentropy", metrics = ["accuracy"])

dataset_size = 10000
batch_size = 1000
input_data, gt_data = mnist_data.train.next_batch(dataset_size)
input_data = input_data.reshape((dataset_size,28,28,1))

convNet.fit(input_data,gt_data,batch_size=batch_size, epochs = 10, validation_split=0.1, callbacks=[tbCallBack])

sample_size = 5000
input_test,gt_test = mnist_data.test.next_batch(sample_size)
input_test = input_test.reshape((sample_size,28,28,1))
convNet.evaluate(input_test, gt_test)

