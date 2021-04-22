# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:57:42 2020

@author: Sahil sutty
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras. applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
K.set_image_data_format('channels_last')
from matplotlib.pyplot import imshow
from cnn_utils import *
np.random.seed(1)


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

index = 102
plt.imshow(X_train_orig[index])
#print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
print ("number of training examples = " + str(X_train.shape[0]))
#print ("number of test examples = " + str(X_test.shape[0]))
#print ("X_train shape: " + str(X_train.shape))
#print ("Y_train shape: " + str(Y_train.shape)
#print ("X_test shape: " + str(X_test.shape))
#print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

def Cat_Model(input_shape):
    X_input = Input(input_shape)
    
    X=ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(32, (3,3),strides = (1,1),kernel_initializer="glorot_uniform",name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(32, (3,3),strides = (1,1),kernel_initializer="glorot_uniform",name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    
    
    
    X = MaxPooling2D((2,2), name = 'max_pool1')(X) 
    
    
    
    X = Flatten()(X)
    X = Dense(90,activation='relu', name='fc0',kernel_initializer="glorot_uniform",bias_initializer='zeros')(X)
    X = Dense(60,activation='relu', name='fc0.1',kernel_initializer="glorot_uniform",bias_initializer='zeros')(X)
    X = Dense(30,activation='relu', name='fc1',kernel_initializer="glorot_uniform",bias_initializer='zeros')(X)
    X = Dense(1,activation='sigmoid', name='fc')(X)
    
    model = Model(inputs = X_input, output = X, name='Cat_Model')
    
    return model

catmodel = Cat_Model(X_train.shape[1:])
catmodel.compile('adamax', 'binary_crossentropy', metrics=['accuracy'])
catmodel.fit(X_train, Y_train, epochs=40, batch_size=64)

preds = catmodel.evaluate(X_test, Y_test, batch_size=16, sample_weight=None)
print()
print("Loss = "+ str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

catmodel.summary()
plot_model(catmodel, to_file='catModel.png')

def pre_dict(x1):
    img_path = x1
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    x=int(catmodel.predict(x))
    
    if x==0:
        return 'not_cat'
    else:
        return 'cat'
print()
print(pre_dict('cat2.jpg'))
























'''def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32,[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32,[None, n_y]) 
    
    return X,Y
X, Y = create_placeholders(64, 64, 3, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))

def initialize_parameters():
    tf.set_random_seed(1)
    
    W1=tf.get_variable("W1", [4,4,3,8], initializer=tf1.keras.initializers.GlorotUniform(seed=0))
    W2=tf.get_variable("W2", [2,2,8,16], initializer=tf1.keras.initializers.GlorotUniform(seed=0))
    
    parameters = {"W1":W1,
                  "W2":W2}
    return parameters
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = "+str(parameters["W1"].eval()[1,1,1]))
    print("W2 = "+str(parameters["W2"].eval()[1,1,1]))

def forward_propagation(X,parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding ='SAME')
    
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides=[1,4,4,1], padding='SAME')
    
    F = tf.layers.flatten(P2)
    
    Z3 = tf.layers.dense(F, 6)
    
    return Z3

tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X,Y = create_placeholders(64,64,3,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6,)})
    print("Z3 = " + str(a))
def compute_cost(Z3 ,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    return cost
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64,64,3,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
    print("cost = "+str(a))
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []
    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y )
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([cost, optimizer], feed_dict={X:minibatch_X, Y:minibatch_Y})
                minibatch_cost += temp_cost
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost) 
        
        plt.plot(np.squeeze(costs))
        plt.ylable('cost')
        plt.xlabel('ittration (per tens)')
        plt.title("Learning rate ="+ str(learning_rate))
        plt.show()
        
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        return train_accuracy, test_accuracy, parameters

_,_, parameters = model(X_train, Y_train, X_test, Y_test)'''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    