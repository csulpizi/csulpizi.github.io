## K-Means Classifier

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

Total testing accuracy: 97.22% <br>
Testing accuracy for label 0: 98.21% <br>
Testing accuracy for label 1: 77.55% <br><br>

<img src="images/k_means_example_1.png?raw=true"/>

One observation about the results is that the accuracy for label 1 is quite a bit lower than the accuracy for label 0. This is because the default loss function used in the train() function weighs the loss of each point equally. In this example, there are roughly 9,500 data points with class == 0, and only 500 data points with class == 1. It is obvious, therefore, that the loss function   <br><br>

One of the optional inputs for the learn() function is loss_coefficients. This input allows you to weigh each class's loss separately. If we were interested in finding training the model such that the loss for label 0 and label 1 was weighted equally, we could set loss_coefficients = [1/n_0, 1/n_1] where n_0 and n_1 are the number of data points with labels 0 and 1 respectively. The results of the model when these coefficients are used are shown below. <br><br>

Total testing accuracy: <br>
Testing accuracy for label 0: <br>
Testing accuracy for label 1: <br><br>

Similarly, if we were interested in getting close to 100% accuracy for label 1 and did not care as much about the accuracy of label 0, you could set the loss_coefficients = [1e-5, 1]. The results of the model when these coefficients are used are shown below.<br><br>

### Example 2: 3-D Sample
