import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten
import numpy as np
class TransferNetwork(Model):
    def __init__(self, num_of_filters, mean=0, std=0.01):
        super(TransferNetwork, self).__init__()
        self.tc = Conv2D(filters=num_of_filters, kernel_size=(3, 3), padding='same', name='conv_to_transfer', 
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=mean, stddev=std, seed=None),
                           bias_initializer = tf.keras.initializers.Constant(value=0.01))  # 卷积层
        self.a = Activation('relu', name='TN_relu')  # 激活层
        self.p = MaxPool2D(pool_size=(2, 2), strides=2, padding='same', name='TN_padding')  # 池化层
        self.flatten = Flatten()

    def call(self, x):
        x = self.tc(x)
        x = self.a(x)
        x = self.p(x)
        y = self.flatten(x)
        return y
    
def transfer(num_of_filters, training_input_set, ground_truth_set):
    net = TransferNetwork(num_of_filters)
    net.compile(optimizer='adam', loss='mean_squared_error')
    print('now fitting')
    training_input_set = np.array(training_input_set)
    ground_truth_set = np.array(ground_truth_set)
    ground_truth_set = ground_truth_set.reshape(ground_truth_set.shape[0], ground_truth_set.shape[1] * ground_truth_set.shape[2] * ground_truth_set.shape[3])
    print(training_input_set.shape)
    print(ground_truth_set.shape)
    #input()
    net.fit(training_input_set, ground_truth_set, batch_size=8, epochs=40, shuffle=True)
    return net.tc.get_weights()[0], net.tc.get_weights()[1]

def information_combination(kernel, multiplier):
    net = TransferNetwork(kernel.shape[3], mean=np.mean(kernel), std=np.std(kernel))
    net.build(input_shape=(1, 9, 9, kernel.shape[2]))
    net.call(Input(shape=(9, 9, kernel.shape[2])))
    new_kernel = multiplier * kernel + (1 - multiplier) * net.tc.get_weights()[0]
    return new_kernel