## K-Means Classifier
Code can be found [here](https://github.com/cory-sulpizi/k_means_classifier/blob/master/k_means_classifier.py).
Readme can be found [here](https://github.com/cory-sulpizi/k_means_classifier/blob/master/README.md).


One project I worked on was a bicycle counter that tracked objects and classified whether or not the object was a bike. One of the models used to classify the objects was a K-Means Classifier. This classifier took object position and velocity as inputs.






The k_means_classifier functions use k-means clustering and logistic
regression to classify points in an m-dimensional space. 

### Code

[Link](https://github.com/cory-sulpizi/cory-sulpizi.github.io/blob/master/k_means_classifier.py)

### The Model

The model relies on finding a number of k-cluster centres within the given data. 

The model takes a series of data points with m-dimensional coordinates as inputs. A series of k-cluster centres is found within the data

For each 


<img src="images/k_means_model.png?raw=true"/>

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
