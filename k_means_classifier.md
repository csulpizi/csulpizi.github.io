## K-Means Classifier
Code can be found [here](https://github.com/cory-sulpizi/k_means_classifier/blob/master/k_means_classifier.py).<br>
Readme can be found [here](https://github.com/cory-sulpizi/k_means_classifier/blob/master/README.md).<br>

One project I worked on was an automated bike counter that tracked objects and classified whether or not each object was a bike. One of the inputs of the classifier was a k-Means Classifier algorithm as described below. The k-Means Classifier was used to classify objects based on their 2-dimensional on-screen position on screen and also their 2-dimensional on-screen velocity. This classifier can be used to classify any data set that has m-dimensional continuous coordinates. 

### The Model
The model relies on using [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) to find the approximate cluster centres of the provided data points. These clusters are found for each class. i.e. a set of cluster centres is found for class 0, a set is found for class 1, etc. The set "g" contains all of the cluster centres among all classes. 

In order to predict the class of a data point, the following algorithm is performed:<br>
1. For each cluster centre ```g(i)```, find the distance ```d(i)``` between this data point and that cluster centre.<br>
2. For each cluster centre ```g(i)``` and each class j, find ```logit(i,j) = d(i) * w(i,j)```. ```w(i,j)``` is the weight for cluster centre i and class j. This value needs to be estimated.<br>
3. For each class j, find ```ŷ(j)``` by using the softmax function, defined in the equation below. Softmax ensures that ```ŷ(j)``` is positive and that ```sum(ŷ)``` is equal to 1, effectively transforming the model into a probability density function. ```b``` is the softmax bias, which needs to be estimated. <img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/softmax.jpg?raw=true"/>
4. Use argmax to find the predicted label for the data point, by finding the j value that maximizes ```ŷ(j)```.<br>

The image below demonstrates the algorithm visually in a tree diagram.<br>
<img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/model_diagram.png?raw=true"/>

In order to use the model, we need to find the weights ```w```, the softmax bias ```b```, and the cluster centres ```g```. The train() function uses stochastic gradient descent and logistic regression to find these values.

The train() function works as follows:
1. Find the k-cluster centres for each class. The number of cluster centres to find per class is specified by the user through the parameter ```k```. All of the cluster centres are combined into the set ```g```, which is an array of shape ```(sum(k), m)``` where ```m``` is the number of dimensions.
2. Divide the dataset provided into training and testing sets. 80% of the data points are used for training and 20% are used for testing.
3. Set initial values for array ```w``` and scalar ```b```.
4. Divide the data set into batches for stochastic gradient descent. 
5. Find the loss of each data point in the batch using the log-loss function, defined below. ```loss_coef``` is a user provided value that defines how much to scale the loss of each class. <img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/log_loss.jpg?raw=true"/>
6. Use [Adam Optimization](https://arxiv.org/abs/1412.6980) to adjust the ```w``` and ```b``` values by minimizing ```sum(loss)``` among all points in the batch. 
7. Repeat steps 4 - 6 for the specified number of iterations.
8. Calculate the accuracy of the trained classifier by comparing the predicted labels for the testing data set to their actual values.
9. Output the results.

Once the train() function is performed, the predict() function can be used to predict labels for any provided points.

### Example 1: 2 dimensional space, 2 classes
This example uses data points gathered from a motion tracking camera. Each data point has a 2-dimensional position and an associated class that was identified manually by a user. A label of 1 means that the data point corresponds to a bicycle that passed by the camera, whereas a label of 0 means that the data point was not a bike (instead it might have been a car, a pedestrian, noise, etc.). The number of points with label == 0 is roughly 10,000, whereas the number of points with label == 1 is roughly 500.

Below is an example using the data and k = [30, 10]:<br>
```w, b, g, _, _ = train(x, y, k=[30,10])```<br>

Total testing accuracy: 97.56% <br>
Testing accuracy for label 0: 98.67% <br>
Testing accuracy for label 1: 75.26% <br>

<img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/example_0.png?raw=true"/>

Since there are more data points with label == 0, the training function skewed the results towards label 0 (notice that the testing accuracy for label 0 is nearly 100%, whereas the accuracy for label 1 is only 75%). 

If we are interested in finding a classifier that improves the accuracy of label 1, we could add a loss coefficient to weight each class equally by setting the optional parameter loss_coef to [1/n_0, 1/n_1], where n_0 and n_1 are the number of points with labels 0 and 1 respectively. 

Below is an example using loss coefficients weighted by population:<br>
```w, b, g, _, _ = train(x, y, k=[30,10], loss_coef=[1/np.sum(y==0),1/np.sum(y==1)])```<br>

Total testing accuracy: 96.1%<br>
Testing accuracy for label 0: 95.92%<br>
Testing accuracy for label 1: 100.0%<br>

<img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/example_1.png?raw=true"/>

As you can see the overall accuracy was slightly impacted, but the accuracy of label 1 was significantly improved. The decision region for class 0 became much smaller, and the decision region for class 1 became much larger.

### Example 2: 3 dimensional space, 5 classes
This example uses randomly generated data in a 3-dimensional space. A series of ellipsoids was generated, then a series of data points in a grid (each position between 0 and 1) was classified by whether or not they were in each ellipsoid, then the position of each point was randomly translated by ~N(0,0.03) in each dimension. k = [50, 20, 10, 20, 30] was selected based on visual inspection. 

The number of points in each class are as follows: [6876, 562, 94, 568, 1161]<br>

Below is an example using the data and k = [50, 20, 10, 20, 30]:<br>
```w, b, g, _, _ = train(x, y, k=[50, 20, 10, 20, 30])```<br>

Total testing accuracy: 91.69%<br>
Testing accuracy by label: [97.96, 61.03, 56., 67.88, 91.74]%<br>

<img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/example_2.gif?raw=true"/>

It is obvious that the large classes (specifically classes 0 and 4) have significantly larger accuracies. In the animation above you can even see that the smaller classes (specifically class 2) have very little "real estate" in the decision boundary. 

If we are interested in improving the accuracy of each class, we can again weight each class by 1/n_i, where n_i is the number of data points in that class. 

Below is an example using loss coefficients weighted by population:<br>
```w, b, g, _, _ = train(x, y, k=[50,20,10,20,30], loss_coef=[1/np.sum(y==0),1/np.sum(y==1),1/np.sum(y==2),1/np.sum(y==3),1/np.sum(y==4)])```<br>

Total testing accuracy: 84.51%<br>
Testing accuracy by label: [84.29, 90.08, 100., 79.82, 83.89]%<br>

Below is a comparison between the default loss_coef and the adjusted loss coefficients as described above:
<img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/example_3.gif?raw=true"/>

As you can see, the overall accuracy went down, but the accuracies by label became much more equal. As you can see in the above animation the decision regions defined for the smaller classes became much larger. The decision region of class 2 exemplifies this: before adjusting the loss_coef there the decision region for class 2 was small, whereas afterwards it is significantly larger. Similarly, the decision region for class 0 became much smaller. 
