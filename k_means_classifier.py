# -*- coding: utf-8 -*-
"""
K-Means Classifier
------------------------------------------------------------------------------
v1.0 - 2018-09-17
v1.1 - 2018-10-13 -> Update to clean up inputs and outputs
v2.0 - 2019-07-07 -> Update to improve efficiency and "clean" code
v2.1 - 2019-07-08 -> Improved readability
------------------------------------------------------------------------------

The k_means_classifier functions use k-means clustering and logistic
regression to classify points in an m-dimensional space. 

k_means_classifier.train() finds the k-means cluster centres (g) and 
their associated weight values (w) by using logistic regression and stochastic
gradient descent. 

k_means_classifier.predict() predicts the labels of provided data points
by using the found w and g values.

"""

import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
import random

def train(x, y, k, vb = False, learn_rate = 1e-2, max_iters = 2500, sgd_size = 500):
    
    """
    Train the k-means classifier by finding the w and g values.
    
    Inputs: 
        x -> shape(n, m). n data points with m-dimensional coordinates
        y -> shape(n, ). n data points with class between 0 and v-1, where v is the 
                number of possible classes
        k -> shape(v, ). The number of k-cluster centres desired for each class.
        vb (opt.) -> Verbose. Whether or not the function should print its results
        learn_rate (opt.) -> Learning rate for stochastic gradient descent
        max_iters (opt.) -> The number of iterations for training
        sgd_size (opt.) -> The batch size for stochastic gradient descent
        
    Outputs:
        w_out -> shape(sum(k), v). The predicted weight values
        g -> shape(sum(k), m). The found k-cluster centres
        acc -> The overall training accuracy of the model
        acc_by_label -> shape(v). The training accuracy for each class
    """
    
    if vb: print("k_means_logistic_regression training...\n\nInitializing variables")
    
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    
    n = y.size ## Number of data points in the provided data
    v = np.max(y) + 1 ## Number of classes in the provided data
    m = x.shape[1] ## Number of dimensions in the provided data
    K = np.sum(k) # The total number of clusters
    
    ## x_by_label is a list of length (v)
    ## x_by_label[i] contains all data points in x with class == class i
    x_by_label = [x[np.where(y == c_v),:][0] for c_v in range(v)]
    
    ## y_by_label is an array of shape (n, v)
    ## Contains a boolean encoding of each y value e.g. y = 4 when v = 6 would
    ## be represented as (0, 0, 0, 0, 1, 0)
    y_by_label = np.expand_dims(np.arange(v), axis = 0) == np.expand_dims(y, axis = 1)
    
    ## Raise exceptions if inputs are invalid
    if x.shape[0] != y.shape[0]: raise Exception("The first dimensions of x and y should be equal, but instead have sizes "+str(x.shape)+", "+str(y.shape))
    if len(k) > v: raise Exception("The length of k should equal the number of classes, but instead the length of k is "+str(len(k))+" and the number of classes is "+str(v))
    if np.any(np.asarray(k) > np.sum(y_by_label, 0)): raise Exception("k values must not exceed the number of labels in each class, but instead the k values are "+str(k)+" and the number of labels in each class are "+str(np.sum(y_by_label, 0)))
    
    if vb: print("\nCalculating k-means clusters")
    
    ## Find the k-cluster centres for each class. g is a list of all centres among all classes
    g = np.zeros((0,m))
    for c_v in range(v):
        temp_kmeans = KMeans(k[c_v]).fit(x_by_label[c_v])
        g = np.vstack((g,(temp_kmeans.cluster_centers_).tolist()))
        
    ## Randomly select indices of x and y to split the data into testing and training data sets
    shuf_ind = np.arange(n)
    random.shuffle(shuf_ind)
    train_ind = shuf_ind[:int(0.8 * n)]
    test_ind = shuf_ind[int(0.8 * n):]
    
    ## tf_x and tf_y are tf placeholders for the data point locations and labels respectively
    tf_x = tf.placeholder(tf.float64, (None, m)) 
    tf_y = tf.placeholder(tf.float64, (None, v))
    
    ## d calculates the euclidean distance between the given data points tf_x and
    ## the k-cluster centres g
    d = tf.sqrt(tf.sqrt(tf.reduce_sum((tf.expand_dims(tf_x, axis = 1) - tf.expand_dims(g, axis = 0)) ** 2, axis = -1)))

    ## Intitialize random weight values    
    w = tf.Variable(np.random.normal(0.0,1.0,(K,v)), dtype=tf.float64)
    
    ## Intialize model outputs
    d_by_w = tf.reduce_sum(tf.expand_dims(d, axis = -1) * tf.expand_dims(w, axis = 0),axis=1)
    y_hat = tf.sigmoid(d_by_w)
    
    ## The loss function used is a multi-class log loss function
    loss = tf.reduce_mean(-(tf_y * tf.log(y_hat + 1e-7) + (1 - tf_y) * tf.log(1 - y_hat + 1e-7)))
    
    ## Initialize the learning function. The learning function uses Adam Optimization,
    ## a variation on gradient descent developed by D.P. Kingma and J. Ba (2014)
    learn = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    
    if vb: print("\nTraining using k-means clustering...\n")
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())        
    
        for it in range(max_iters): ## Iterate for specified number of epochs
            
            ## Divide the training data points randomly into lists of 
            ## length (sgd_size) in order to perform stochastic gradient descent. 
            shuf_ind = train_ind[:]
            random.shuffle(shuf_ind)
            sgd_ind = [shuf_ind[c_shuf * sgd_size:(1+c_shuf) * sgd_size] for c_shuf in range(int(np.ceil(len(train_ind) / sgd_size)))]
            
            ## For each list in sgd_ind, perform the stochastic gradient descent
            ## on those training data points. 
            for c_shuf in sgd_ind: sess.run(learn,feed_dict={tf_x: x[c_shuf,:], tf_y: y_by_label[c_shuf]})
    
            ## Print the MSE loss every 100 epochs if vb = True
            if it % 100 == 0 and vb:
                print("Epoch "+str(it)+", MSE Loss: "+str(sess.run(loss,feed_dict={tf_x: x[train_ind,:], tf_y: y_by_label[train_ind]})))
    
        ## Print the final MSE loss if vb = True
        if vb: print("\nFinal MSE Loss: "+str(sess.run(loss,feed_dict={tf_x: x[train_ind,:], tf_y: y_by_label[train_ind]})))
        
        w_out = sess.run(w)                                         ## Predicted weight
        y_out = sess.run(y_hat,feed_dict={tf_x: x[test_ind,:]})     ## Predicted y values for each class
        l_out = np.argmax(y_out, axis = 1)                          ## Predicted label
        
        ## Calculate the total accuracies
        acc = np.mean(l_out == y[test_ind])
        if vb: print("\nFinal total testing accuracy: "+str(np.round(acc * 100, 2))+"%")
        acc_by_label = [np.sum((l_out == c_v) * (y[test_ind] == c_v)) / np.sum(y[test_ind] == c_v) for c_v in range(v)]
        if vb: print("Final testing accuracy for each label: "+str(np.round(np.asarray(acc_by_label) * 100, 2))+"%")
        
    return w_out, g, acc, acc_by_label

def predict(x, w, g, y = []):
    
    """
    Use the k-means classifier to predict the classes of each given data point. 
    
    Inputs: 
        x -> shape(n, m). n data points with m-dimensional coordinates
        w -> The weight values calculated by the train() function
        g -> The k-cluster centres calculated by the train() function
        y (opt.) -> shape(n, ). n data points with class between 0 and v-1, where v is the 
                number of possible classes. If this is provided the function will print
                the accuracy of the model
        
    Outputs:
        l_out -> shape(n, ). The predicted labels for each data point
        y_out -> shape(n, v). The certainty that each data point belongs to each class.
    """
    
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    
    v = w.shape[1] ## Number of classes in the provided data
    m = x.shape[1] ## Number of dimensions in the provided data
    
    if x.shape[0] != y.shape[0] and y.size > 0: raise Exception("The first dimensions of x and y should be equal, but instead have sizes "+str(x.shape)+", "+str(y.shape))
    if w.shape[0] != g.shape[0]: raise Exception("The first dimensions of w and g should be equal, but instead have sizes "+str(w.shape)+", "+str(g.shape))
    if g.shape[1] != m: raise Exception("The second dimensions of g and x should be equal, but instead have sizes "+str(g.shape)+", "+str(x.shape))
    
    ## Initialize model inputs
    tf_x = tf.placeholder(tf.float64, (None, m)) 
    d = tf.sqrt(tf.sqrt(tf.reduce_sum((tf.expand_dims(tf_x, axis = 1) - tf.expand_dims(g, axis = 0)) ** 2, axis = -1)))
    w = tf.Variable(w, dtype=tf.float64)
    
    ## Initialize model outputs
    d_by_w = tf.reduce_sum(tf.expand_dims(d, axis = -1) * tf.expand_dims(w, axis = 0),axis=1)
    y_hat = tf.sigmoid(d_by_w)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        
        ## Calculate the predictions of the model.
        y_out = sess.run(y_hat,feed_dict={tf_x: x})     ## Predicted y values for each class
        l_out = np.argmax(y_out, axis = 1)              ## Predicted label
        
        if y.size > 0:
            ## Calculate the total accuracies if y is provided
            acc = np.mean(l_out == y)
            print("\nFinal total accuracy: "+str(np.round(acc * 100, 2))+"%")
            acc_by_label = [np.sum((l_out == c_v) * (y == c_v)) / np.sum(y == c_v) for c_v in range(v)]
            print("Final accuracy for each label: "+str(np.round(np.asarray(acc_by_label) * 100, 2))+"%")
            
    return l_out, y_out