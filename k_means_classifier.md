## K-Means Classifier
Code can be found [here](https://github.com/cory-sulpizi/k_means_classifier/blob/master/k_means_classifier.py).<br>
Readme can be found [here](https://github.com/cory-sulpizi/k_means_classifier/blob/master/README.md).<br>

One project I worked on was an automated bike counter that tracked objects and classified whether or not each object was a bike. One of the inputs of the classifier was a k-Means Classifier algorithm as described below. The classifier was used to classify objects based on their 2-dimensional on-screen position on screen as well as their 2-dimensional on-screen position. The classifier can be used to classify any data set that has m-dimensional continuous coordinates. 

### The Model
The model relies on using k-means clustering to find the approximate cluster centres of the provided data points. These clusters are found for each class. i.e. a set of cluster centres is found for class 0, a set is found for class 1, etc. The set "g" contains all of the cluster centres among all classes. 

In order to predict the class of a data point, the following algorithm is performed:<br>
1. For each cluster centre ```g(i)```, find the distance ```d(i)``` between this data point and that cluster centre.<br>
2. For each cluster centre ```g(i)``` and each class j, find ```logit(i,j) = d(i) * w(i,j)```. ```w(i,j)``` is the weight for cluster centre i and class j. This value needs to be estimated.<br>
3. For each class j, find ```ŷ(j)``` by using the softmax function, defined in the equation below. Softmax ensures that ```ŷ(j)``` is positive and that ```sum(ŷ)``` is equal to 1, effectively transforming the model into a probability density function. ```b``` is the softmax bias, which needs to be estimated. <img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/softmax.jpg?raw=true"/>
4. Use argmax to find the predicted label for the data point, by finding the j value that maximizes ```ŷ(j)```.<br>

The image below demonstrates the algorithm visually in a tree diagram.<br>
<img src="https://github.com/cory-sulpizi/k_means_classifier/blob/master/images/model_diagram.png?raw=true"/>

In order to use the model, we need to find the weights w, the softmax bias b, and the cluster centres g. The train() function uses stochastic gradient descent and logistic regression to find these values.



### Example 1: Bicycle Position

This example uses the position of objects from . Each data point 

Total testing accuracy: 97.56% <br>
Testing accuracy for label 0: 98.67% <br>
Testing accuracy for label 1: 75.26% <br>

<img src="images/k_means_example_1.png?raw=true"/>

Total testing accuracy: 96.1%
Testing accuracy for label 0: 95.92%
Testing accuracy for label 1: 100.0%

### Example 2: 3-D Sample

Noise ~N(0,0.03)

91.69%
[97.96 61.03 56.   67.88 91.74]%

<img src="images/k_means_example_2.gif?raw=true"/>

2: 
84.51%
[ 84.29  90.08 100.    79.82  83.89]%
