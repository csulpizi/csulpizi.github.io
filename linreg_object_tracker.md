[Back to portfolio](index.md)

## LinReg Object Tracker
Code can be found [here](https://github.com/csulpizi/linreg_object_tracker/blob/master/linreg_track_objects.py).<br>
Readme can be found [here](https://github.com/csulpizi/linreg_object_tracker/blob/master/README.md).<br>

One project I worked on was developing a computer vision program that detected, tracked, and counted the number of bicycles traveling through a video. In order to perform the tracking, I developed the following function. The function takes 2-dimensional coordinates and their associated time stamp and figures out which "items" (individual data points) belong to which "objects" (a series of data points that corresponds to a single object moving through the scene).

The algorithm uses [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) to find new objects, uses linear regression to find the path of motion of each object, and decides whether or not unassigned items should be assigned to each object by finding the distance between each item and the predicted location of each object at that time stamp. 

### The Algorithm
The algorithm works frame by frame to progressively detect new objects, and determine whether or not items should be assigned to each existing object.

There are 2 sets of objects:
1. ```obj_list```. This list contains all of the objects and the indices of each item belonging to it
2. ```s_act_obj```. This list contains the indices of objects ```in obj_list``` that are "active". Since we know that objects which are too old or too far off-screen cannot be assigned new items, those objects are considered inactive and are removed from ```s_act_obj```. Active objects on the other hand are objects that should be considered when looking at new items. 

The algorithm is described in the subsections below, and it is demonstrated visually in the example section below.

#### Item Assignment
Each frame, the following actions are performed:
1. Determine whether or not any of the active objects should be changed to inactive. If the object's predicted location is off-screen, or if too much time has passed since the object was assigned a new point, that object is removed from the active objects list.
2. Calculate the predicted location of each active object.
3. Calculate the distance between each of the items in this frame and the predicted locations of each active object.
4. Find the item and the object with the minimum distance. If that distance is less than ```bound_tight``` then assign that item to that object. ```bound_tight``` is a user-provided input that governs the maximum distance an item can be from the predicted location of an object and still be considered a member of that object. 
5. Repeat step 4 with all of the remaining items and objects until there are no items, no objects, or until the minimum distance exceeds ```bound_tight```.
6. Use linear regression to recalculate the trajectory of each of the objects. The trajectory is calculated using the ```m``` most recent items in each object. ```m``` is a user-inputted value.

#### Object Creation
After performing item assignment, if there are still items left over, the algorithm determines whether or not those points should be considered new objects using the following actions (this algorithm is effectively a [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) algorithm):
1. For each combination of unused items in this frame and unused items in the previous frame, find the parameters of the line passing through those points. Call these combinations the "pairs".
2. For each of the pairs, find the line's predicted location for the next 5 time frames.
3. Calculate the distance between those predicted locations and all of the items in the next 5 time frames.
4. Calculate the "score" of each of the lines. For each distance calculated above, add 2 to the score for every distance that's smaller than the ```bound_tight```, and add 1 to the score for every distance that's between ```bound_tight``` and ```2 * bound_tight```.
5. Find the pair with the highest score that has at least one item within ```bound_tight``` of the predicted locations, and create a new object and assigned that pair of items to it. 
6. Repeat steps 1 through 5 until there are no remaining pairs with at least 1 point within the ```bound_tight```. 

#### Limitations
One of the major drawbacks of this algorithm is that it does not use visuals to determine which items belong to which objects. As such, if objects get to close or cross paths, it is possible that the algorithm would fail to detect which object is which. It is also possible that some items could be accidentally assigned to objects despite the fact that they are visually very different. 

The flipside is that this algorithm is quite simple. You do not need to provide any visual information, and the only information you need to provide is a series of coordinates.

Lastly, this algorithm is intended to be used on objects whose motion is fairly predictable and is close-to-linear. This algorithm would not work well for objects that change direction very rapidly (say, for example, a ball bouncing off of a wall) or objects with very non-linear motion (say, for example, a ball falling to the ground). 

### Example: Tracking Cars through a Scene.
This example uses items that were found by using background detection on a video of cars moving down a street. Each of the points represents a car that was detected. 

The plot below shows 4 frames taken from the aforementioned data set. 
<img src="https://github.com/csulpizi/linreg_object_tracker/blob/master/images/example_1.jpg?raw=true">

The blue object demonstrates how items are assigned to existing objects. The blue "x" points are all of the items belonging to the blue object*, the blue line is the calculated trajectory of that object, the end of the line represents the predicted location of that object in this time frame, and the grey circle shows the ```bound_tight``` distance from the predicted location. The black "x" points represent items in this time frame that have not yet been classified. You can see that in each frame there is an item within the grey circle, and so each of those items is added to the blue object. <br>
*Note: The blue items are all from previous time frames. They are shown simply to show the motion of the object over time.

The green object demonstrates how new objects are created. At t=551 you can see an item that is not assigned to any object (the black "x" at the bottom of the plot). At t=552, you can see the item from the previous frame (shown in grey) next to a new item (shown in black). These two points are considered a potential new object. The algorithm decides that there are enough items in the future trajectory of this potential object, and so the green object is created. You can see in t=553 and t=554 that new items are within the ```bound_tight``` distance and are therefore added to the green object. 

The animation below shows object tracking results superimposed on the source video. 
<img src="https://github.com/csulpizi/linreg_object_tracker/blob/master/images/example_2.gif?raw=true">

As you can see the algorithm works well even when there are multiple objects on screen at the same time, and even when the motion of the object is non-linear. 
