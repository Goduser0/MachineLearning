import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X,Y=load_coffee_data()
plt_roast(X,Y)

# Normalize Data
norm_l=tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn=norm_l(X)

# increase training set size
Xt=np.tile(Xn,(1000,1))
Yt=np.tile(Y,(1000,1))

# tensorflow model
tf.random.set_seed(1234)
model=Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3,activation='sigmoid',name='Layer1'),
        Dense(1,activation='sigmoid',name='Layer2'),
    ]
)
model.summary() # provides a description of the network

W1,b1=model.get_layer("Layer1").get_weights()
W2,b2=model.get_layer("Layer2").get_weights()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)
model.fit(
    Xt,Yt,            
    epochs=10,
)

# After fitting,the weights have been changed
W1,b1=model.get_layer("Layer1").get_weights()
W2,b2=model.get_layer("Layer2").get_weights()

# set weighets
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("Layer1").set_weights([W1,b1])
model.get_layer("Layer2").set_weights([W2,b2])

# Predictions
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

plt_output_unit(W2,b2)

netf= lambda x : model.predict(norm_l(x))
plt_network(X,Y,netf)