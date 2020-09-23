#--------------------------------------- SVM SUPPORT VECTOR MACHINE ------------------------------------------------------#
'''
SUPPORT VECTOR MACHINES
Support Vector Machines
A Support Vector Machine (SVM) is a powerful supervised machine learning model used for classification. An SVM makes classifications by defining a decision boundary and then seeing what side of the boundary an unclassified point falls on. In the next few exercises, we’ll learn how these decision boundaries get defined, but for now, know that they’re defined by using a training set of classified points. That’s why SVMs are supervised machine learning models.

Decision boundaries are easiest to wrap your head around when the data has two features. In this case, the decision boundary is a line. Take a look at the example below.

Two clusters of points separated by a line
Note that if the labels on the figures in this lesson are too small to read, you can resize this pane to increase the size of the images.

This SVM is using data about fictional games of Quidditch from the Harry Potter universe! The classifier is trying to predict whether a team will make the playoffs or not. Every point in the training set represents a “historical” Quidditch team. Each point has two features — the average number of goals the team scores and the average number of minutes it takes the team to catch the Golden Snitch.

After finding a decision boundary using the training set, you could give the SVM an unlabeled data point, and it will predict whether or not that team will make the playoffs.

Decision boundaries exist even when your data has more than two features. If there are three features, the decision boundary is now a plane rather than a line.

Two clusters of points in three dimensions separated by a plane.
As the number of dimensions grows past 3, it becomes very difficult to visualize these points in space. Nonetheless, SVMs can still find a decision boundary. However, rather than being a separating line, or a separating plane, the decision boundary is called a separating hyperplane.

Instructions
1.
Run the code to see two graphs appear. Right now they should be identical. We’re going to fix the bottom graph so it has a good decision boundary. Why is this decision boundary bad?


The decision boundary doesn’t separate the two classes from each other!

2.
Let’s shift the line on the bottom graph to make it separate the two clusters. The slope of the line looks pretty good, so let’s keep that at -2.

We want to move the boundary up, so change intercept_two so the line separates the two clusters.


intercept_two = 15 works pretty well!'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from graph import ax, x_1, y_1, x_2, y_2

#Top graph intercept and slope
intercept_one = 8
slope_one = -2

x_vals = np.array(ax.get_xlim())
y_vals = intercept_one + slope_one * x_vals
plt.plot(x_vals, y_vals, '-')

#Bottom Graph
ax = plt.subplot(2, 1, 2)
plt.title('Good Decision Boundary')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.scatter(x_1, y_1, color = "b")
plt.scatter(x_2, y_2, color = "r")

#Change the intercept to separate the clusters
intercept_two = 14
slope_two = -2

x_vals = np.array(ax.get_xlim())
y_vals = intercept_two + slope_two * x_vals
plt.plot(x_vals, y_vals, '-')

plt.tight_layout()
plt.show()

'''
SUPPORT VECTOR MACHINES
Optimal Decision Boundaries
One problem that SVMs need to solve is figuring out what decision boundary to use. After all, there could be an infinite number of decision boundaries that correctly separate the two classes. Take a look at the image below:

6 different valid decision boundaries
There are so many valid decision boundaries, but which one is best? In general, we want our decision boundary to be as far away from training points as possible.

Maximizing the distance between the decision boundary and points in each class will decrease the chance of false classification. Take graph C for example.

An SVM with a decision boundary very close to the blue points.
The decision boundary is close to the blue class, so it is possible that a new point close to the blue cluster would fall on the red side of the line.

Out of all the graphs shown here, graph F has the best decision boundary.

Instructions
1.
Run the code. Both graphs have suboptimal decision boundaries. Why? We’re going to fix the bottom graph.


2.
We’re going to have to make the decision boundary much flatter, which means we first need to lower its y-intercept. Change intercept_two to be 8.

3.
Next, we want the slope to be pretty flat. Change the value of slope_two. The resulting line should split the two clusters.


slope_two = -0.5 works well!
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from graph import ax, x_1, y_1, x_2, y_2

#Top graph intercept and slope
intercept_one = 98
slope_one = -20

x_vals = np.array(ax.get_xlim())
y_vals = intercept_one + slope_one * x_vals
plt.plot(x_vals, y_vals, '-')

#Bottom graph
ax = plt.subplot(2, 1, 2)
plt.title('Good Decision Boundary')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.scatter(x_1, y_1, color = "b")
plt.scatter(x_2, y_2, color = "r")

#Bottom graph intercept and slope
intercept_two = 8
slope_two = -0.5

x_vals = np.array(ax.get_xlim())
y_vals = intercept_two + slope_two * x_vals
plt.plot(x_vals, y_vals, '-')

plt.tight_layout()
plt.show()


'''
SUPPORT VECTOR MACHINES
Support Vectors and Margins
We now know that we want our decision boundary to be as far away from our training points as possible. Let’s introduce some new terms that can help explain this idea.

The support vectors are the points in the training set closest to the decision boundary. In fact, these vectors are what define the decision boundary. But why are they called vectors? Instead of thinking about the training data as points, we can think of them as vectors coming from the origin.

Points represented as vectors.
These vectors are crucial in defining the decision boundary — that’s where the “support” comes from. If you are using n features, there are at least n+1 support vectors.

The distance between a support vector and the decision boundary is called the margin. We want to make the margin as large as possible. The support vectors are highlighted in the image below:

decision boundary with margin highlighted
Because the support vectors are so critical in defining the decision boundary, many of the other training points can be ignored. This is one of the advantages of SVMs. Many supervised machine learning algorithms use every training point in order to make a prediction, even though many of those training points aren’t relevant. SVMs are fast because they only use the support vectors!'''

'''
SUPPORT VECTOR MACHINES
scikit-learn
Now that we know the concepts behind SVMs we need to write the code that will find the decision boundary that maximizes the margin. All of the code that we’ve written so far has been guessing and checking — we don’t actually know if we’ve found the best line. Unfortunately, calculating the parameters of the best decision boundary is a fairly complex optimization problem. Luckily, Python’s scikit-learn library has implemented an SVM that will do this for us.

Note that while it is not important to understand how the optimal parameters are found, you should have a strong conceptual understanding of what the model is optimizing.

To use scikit-learn’s SVM we first need to create an SVC object. It is called an SVC because scikit-learn is calling the model a Support Vector Classifier rather than a Support Vector Machine.
'''
classifier = SVC(kernel = 'linear')
'''
We’ll soon go into what the kernel parameter is doing, but for now, let’s use a 'linear' kernel.

Next, the model needs to be trained on a list of data points and a list of labels associated with those data points. The labels are analogous to the color of the point — you can think of a 1 as a red point and a 0 as a blue point. The training is done using the .fit() method:
'''
training_points = [[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]]
labels = [1, 1, 1, 0, 0, 0]
classifier.fit(training_points, labels) 
'''
The graph of this dataset would look like this:

An SVM with a decision boundary very close to the blue points.
Calling .fit() creates the line between the points.

Finally, the classifier predicts the label of new points using the .predict() method. The .predict() method takes a list of points you want to classify. Even if you only want to classify one point, make sure it is in a list:
'''
print(classifier.predict([[3, 2]]))
'''
In the image below, you can see the unclassified point [3, 2] as a black dot. It falls on the red side of the line, so the SVM would predict it is red.

An SVM with a decision boundary very close to the blue points.
In addition to using the SVM to make predictions, you can inspect some of its attributes. For example, if you can print classifier.support_vectors_ to see which points from the training set are the support vectors.

In this case, the support vectors look like this:
'''
[[7, 5],
 [8, 2],
 [2, 2]]
'''
 Instructions
1.
Let’s start by making a SVC object with kernel = 'linear'. Name the object classifier.


classifier = SVC(kernel = 'linear')
2.
We’ve imported the training set and labels for you. Call classifier‘s .fit() method using points and labels as parameters.


classifier.fit(points, ____)
Fill in the second parameter in the code above.

3.
We can now classify new points. Try classifying both [3, 4] and [6, 7]. Remember, the .predict() function expects a list of points to predict.

Print the results.


Use [[3, 4], [6, 7]] as the parameter of .predict().'''


from sklearn.svm import SVC
from graph import points, labels

classifier = SVC (kernel = 'linear')

classifier.fit(points, labels)

print(classifier.predict([[3,4],[6,7]]))

'''
SUPPORT VECTOR MACHINES
Outliers
SVMs try to maximize the size of the margin while still correctly separating the points of each class. As a result, outliers can be a problem. Consider the image below.

One graph with a hard margin and one graph with a soft margin
The size of the margin decreases when a single outlier is present, and as a result, the decision boundary changes as well. However, if we allowed the decision boundary to have some error, we could still use the original line.

SVMs have a parameter C that determines how much error the SVM will allow for. If C is large, then the SVM has a hard margin — it won’t allow for many misclassifications, and as a result, the margin could be fairly small. If C is too large, the model runs the risk of overfitting. It relies too heavily on the training data, including the outliers.

On the other hand, if C is small, the SVM has a soft margin. Some points might fall on the wrong side of the line, but the margin will be large. This is resistant to outliers, but if C gets too small, you run the risk of underfitting. The SVM will allow for so much error that the training data won’t be represented.

When using scikit-learn’s SVM, you can set the value of C when you create the object:
'''
classifier = SVC(C = 0.01)

'''
The optimal value of C will depend on your data. Don’t always maximize margin size at the expense of error. Don’t always minimize error at the expense of margin size. The best strategy is to validate your model by testing many different values for C.

Instructions
1.
Run the code to see the SVM’s current boundary line. Note that we’ve imported some helper functions we wrote named draw_points and draw_margins to help visualize the SVM.

2.
Let’s add an outlier! Before calling .fit(), append [3, 3] to points and append 0 to labels. This will add a blue point at [3, 3]


3.
Right now, our classifier has hard margins because C = 1. Change the value of C to 0.01 to see what the SVM looks like with soft margins.


When you create classifier, change the value of C to 0.01.

4.
append at least two more points to points. If you want the points to appear on the graph, make sure their x and y values are between 0 and 12.

Make sure to also append a label to labels for every point you add. A 0 will make the point blue and a 1 will make the point red.

Make sure to add the points before training the SVM.


If you wanted to add a red point at [10, 8], do the following:

points.append([10,8])
labels.append(1)
5.
Play around with the C variable to see how the decision boundary changes with your new points added. Change C to be a value between 0.01 and 1.'''


import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from graph import points, labels, draw_points, draw_margin

classifier = SVC(kernel='linear', C = 1)
points.append([8,4])
points.append([2,8])
points.append([8,8])
labels.append(1)
labels.append(0)
labels.append(1)
classifier.fit(points, labels)

draw_points(points, labels)
draw_margin(classifier)

plt.show()

'''
SUPPORT VECTOR MACHINES
Kernels
Up to this point, we have been using data sets that are linearly separable. This means that it’s possible to draw a straight decision boundary between the two classes. However, what would happen if an SVM came along a dataset that wasn’t linearly separable?

data points clustered in concentric circles
It’s impossible to draw a straight line to separate the red points from the blue points!

Luckily, SVMs have a way of handling these data sets. Remember when we set kernel = 'linear' when creating our SVM? Kernels are the key to creating a decision boundary between data points that are not linearly separable.

Note that most machine learning models should allow for some error. For example, the image below shows data that isn’t linearly separable. However, it is not linearly separable due to a few outliers. We can still draw a straight line that, for the most part, separates the two classes. You shouldn’t need to create a non-linear decision boundary just to fit some outliers. Drawing a line that correctly separates every point would be drastically overfitting the model to the data.

A straight line separating red and blue clusters with some outliers.

Instructions
1.
Let’s take a look at the power of kernels. We’ve created a dataset that isn’t linearly separable and split it into a training set and a validation set.

Create an SVC named classifier with a 'linear' kernel.


2.
Call the .fit() method using training_data and training_labels as parameters.


classifier.fit(___, ___)
3.
Let’s see how accurate our classifier is using a linear kernel.

Call classifier‘s .score() function using validation_data and validation_labels as parameters. Print the results.

This will print the average accuracy of the model.


classifier.score(___, ___)
4.
That’s pretty bad! The classifier is getting it right less than 50% of the time! Change 'linear' to 'poly' and add the parameter degree = 2. Run the program again and see what happens to the score.


Wow! It’s now getting every single point in the validation set correct!

Let’s go learn what kernels are really doing!'''

import codecademylib3_seaborn
from sklearn.svm import SVC
from graph import points, labels
from sklearn.model_selection import train_test_split

training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = 'linear')
#cclassifier = SVC(kernel = 'poly', degree = 2)

classifier.fit(training_data, training_labels)

print(classifier.score(validation_data, validation_labels))

'''
SUPPORT VECTOR MACHINES
Polynomial Kernel
That kernel seems pretty magical. It is able to correctly classify every point! Let’s take a deeper look at what it was really doing.

We start with a group of non-linearly separable points that looked like this:

A circle of red dots surrounding a cluster of blue dots.
The kernel transforms the data in a clever way to make it linearly separable. We used a polynomial kernel which transforms every point in the following way:

(x,\ y) \rightarrow (\sqrt{2}\cdot x \cdot y,\ x^2,\ y^2)(x, y)→( 
2
​	 ⋅x⋅y, x 
2
 , y 
2
 )
The kernel has added a new dimension to each point! For example, the kernel transforms the point [1, 2] like this:

(1,\ 2) \rightarrow (2\sqrt{2},\ 1,\ 4)(1, 2)→(2 
2
​	 , 1, 4)
If we plot these new three dimensional points, we get the following graph:

A cluster of red points and blue points in three dimensions separated by a plane.
Look at that! All of the blue points have scooted away from the red ones. By projecting the data into a higher dimension, the two classes are now linearly separable by a plane. We could visualize what this plane would look like in two dimensions to get the following decision boundary.

The decision boundary is a circle around the inner points.
Instructions
1.
In this exercise, we will be using a non-linearly separable dataset similar to the concentric circles above.

Rather than using a polynomial kernel, we’re going to stick with a linear kernel and do the transformation ourselves. The SVM running a linear kernel on the transformed points should perform identically to the SVM running a polynomial kernel on the original points.

To begin, at the bottom of your code, print training_data[0] to see the first data point. You will also see the accuracy of the SVM when the data is not projected into 3 dimensions.


The SVM is pretty bad! Because it is using a linear kernel, it is trying to draw a straight decision boundary.

2.
Let’s transform the data into three dimensions! Begin by creating two empty lists called new_training and new_validation.


3.
Loop through every point in training_data. For every point, append a list to new_training. The list should contain three numbers:

The square root of 2 times point[0] times point[1].
point[0] squared.
point[1] squared.
Remember, to square a number in Python do number ** 2. To take the square root, do number ** 0.5.


4.
Do the same for every point in validation_data. For every point in validation_data, add the new list to new_validation.


5.
Retrain classifier by calling the .fit() method using new_training and training_labels as parameters.


6.
Finally, run classifier‘s .score() method using new_validation and validation_labels as parameters. Print the results. How did the SVM do when the data was projected to three dimensions?'''

from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Makes concentric circles
points, labels = make_circles(n_samples=300, factor=.2, noise=.05, random_state = 1)

#Makes training set and validation set.
training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = "linear", random_state = 1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

print(training_data[0])

new_training = []
new_validation = []

for point in training_data:  
  new_training.append([(2**0.5)*point[0]*point[1], point[0]**2, point[1]**2 ])
for point in validation_data:  
  new_validation.append([(2**0.5)*point[0]*point[1], point[0]**2, point[1]**2 ])
  
print(new_training[0])

classifier.fit(new_training, training_labels)
print(classifier.score(new_validation, validation_labels))

'''
SUPPORT VECTOR MACHINES
Radial Bias Function Kernel
The most commonly used kernel in SVMs is a radial basis function (rbf) kernel. This is the default kernel used in scikit-learn’s SVC object. If you don’t specifically set the kernel to "linear", "poly" the SVC object will use an rbf kernel. If you want to be explicit, you can set kernel = "rbf", although that is redundant.

It is very tricky to visualize how an rbf kernel “transforms” the data. The polynomial kernel we used transformed two-dimensional points into three-dimensional points. An rbf kernel transforms two-dimensional points into points with an infinite number of dimensions!

We won’t get into how the kernel does this — it involves some fairly complicated linear algebra. However, it is important to know about the rbf kernel’s gamma parameter.

classifier = SVC(kernel = "rbf", gamma = 0.5, C = 2)
gamma is similar to the C parameter. You can essentially tune the model to be more or less sensitive to the training data. A higher gamma, say 100, will put more importance on the training data and could result in overfitting. Conversely, A lower gamma like 0.01 makes the points in the training data less relevant and can result in underfitting.

Instructions
1.
We’re going to be using a rbf kernel to draw a decision boundary for the following points:

A cluster of blue points in the middle surrounded by red points.
We’ve imported the data for you and split it into training_data, validation_data, training_labels, and validation_labels.

Begin by creating an SVC named classifier with an "rbf" kernel. Set the kernel’s gamma equal to 1.


The following code would create an SVC with gamma = 100. Your gamma should be 1.

classifier = SVC(kernel = "rbf", gamma = 1)
2.
Next, train the model using the .fit() method using training_data and training_labels as parameters.


classifier.fit(___, ___)
3.
Let’s test the classifier’s accuracy when its gamma is 1. Print the result of the .score() function using validation_data and validation_labels as parameters.


The decision boundary when gamma = 1 looks like this:

The decision boundary fits the blue points nicely.

4.
Let’s see what happens if we increase gamma. Change gamma to 10. What happens to the accuracy of our model?


The decision boundary when gamma = 10 looks like this:

The decision boundary fits to blue points too closely.

5.
The accuracy went down. We overfit our model. Change gamma to 0.1. What happens to the accuracy of our model this time?


Now we’re underfitting. The decision boundary looks like this:'''

#-------script.py---------------
from data import points, labels
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = 'rbf', gamma = 0.4)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

#-------data.py---------------
from random import uniform, seed
import numpy as np

seed(1)

blue_x = []
blue_y = []

red_x = []
red_y = []

#First Blue Column
for i in range(50):
    blue_x.append(uniform(2, 4))
    blue_y.append(uniform(0, 8))

#Horizontal Blue
for i in range(25):
    blue_x.append(uniform(4, 8))
    blue_y.append(uniform(5, 6))

#Left Red
for i in range(30):
    red_x.append(uniform(0,1.9))
    red_y.append(uniform(0, 10))

#Red above blue column
for i in range(15):
    red_x.append(uniform(2, 4))
    red_y.append(uniform(8.1, 10))

#Red below blue horizontal
for i in range(25):
    red_x.append(uniform(4.1,10))
    red_y.append(uniform(0, 4.9))

#Red above blue horizontal
for i in range(25):
    red_x.append(uniform(4.1,10))
    red_y.append(uniform(6.1, 10))

#Smaller blue column
for i in range(10):
    blue_x.append(uniform(6.3, 6.8))
    blue_y.append(uniform(0, 6))


all_x = blue_x + red_x
all_y = blue_y + red_y

points = np.array(list(zip(all_x, all_y)))

labels = np.array([0] * len(blue_x) + [1] * len(red_x))

'''
SUPPORT VECTOR MACHINES
Review
Great work! Here are some of the major takeaways from this lesson on SVMs:

SVMs are supervised machine learning models used for classification.
An SVM uses support vectors to define a decision boundary. Classifications are made by comparing unlabeled points to that decision boundary.
Support vectors are the points of each class closest to the decision boundary. The distance between the support vectors and the decision boundary is called the margin.
SVMs attempt to create the largest margin possible while staying within an acceptable amount of error.
The C parameter controls how much error is allowed. A large C allows for little error and creates a hard margin. A small C allows for more error and creates a soft margin.
SVMs use kernels to classify points that aren’t linearly separable.
Kernels transform points into higher dimensional space. A polynomial kernel transforms points into three dimensions while an rbf kernel transforms points into infinite dimensions.
An rbf kernel has a gamma parameter. If gamma is large, the training data is more relevant, and as a result overfitting can occur.'''

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- SVC PROJECT #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
'''
MACHINE LEARNING: SUPERVISED LEARNING 🤖
Sports Vector Machine
Support Vector Machines are powerful machine learning models that can make complex decision boundaries. An SVM’s decision boundary can twist and curve to accommodate the training data.

In this project, we will use an SVM trained using a baseball dataset to find the decision boundary of the strike zone.

A batter standing in front of the plate with the strike zone outlined.
The strike zone can be thought of as a decision boundary that determines whether or not a pitch is a strike or a ball. There is a strict definition of the strike zone: however, in practice, it will vary depending on the umpire or the player at bat.

Let’s use our knowledge of SVMs to find the real strike zone of several baseball players.

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
17/17Complete
Mark the tasks as complete by checking them off
Create the labels
1.
We’ve imported several DataFrames related to some of baseball’s biggest stars. We have data on Aaron Judge and Jose Altuve. Judge is one of the tallest players in the league and Altuve is one of the shortest. Their strike zones should be pretty different!

Each row in these DataFrames corresponds to a single pitch that the batter saw in the 2017 season. To begin, let’s take a look at all of the features of a pitch. Print aaron_judge.columns.

In this project, we’ll ask you to print out a lot of information. To avoid clutter, feel free to delete the print statements once you understand the data.

2.
Some of these features have obscure names. Let’s learn what the feature description means.

Print aaron_judge.description.unique() to see the different values the description feature could have.

3.
We’re interested in looking at whether a pitch was a ball or a strike. That information is stored in the type feature. Look at the unique values stored in the type feature to get a sense of how balls and strikes are recorded.

'''
print(aaron_judge.type.unique())
'''
You should see that every pitch is either a 'S', a 'B', or an 'X'.

4.
Great! We know every row’s type feature is either an 'S' for a strike, a 'B' for a ball, or an 'X' for neither (for example, an 'X' could be a hit or an out).

We’ll want to use this feature as the label of our data points. However, instead of using strings, it will be easier if we change every 'S' to a 1 and every 'B' to a 0.

You can change the values of a DataFrame column using the map() functions. For example, in the code below, every 'A' in example_column is changed to a 1, and every 'B' is changed to a 2.
'''
df['example_column'] = df['example_column'].map({'A':1, 'B':2})
'''
Finish the following code:
'''
aaron_judge['type'] = aaron_judge['type'].map({'S': ____, 'B': ____})
'''
5.
Let’s make sure that worked. Print the type column from the aaron_judge DataFrame.

'''
print(aaron_judge[____])
'''
Plotting the pitches
6.
There were some NaNs in there. We’ll take care of those in a second. For now, let’s look at the other features we’re interested in.

We want to predict whether a pitch is a ball or a strike based on its location over the plate. You can find the ball’s location in the columns plate_x and plate_z.

Print aaron_judge['plate_x'] to see what that column looks like.

plate_x measures how far left or right the pitch is from the center of home plate. If plate_x = 0, that means the pitch was directly in the middle of the home plate.

7.
We now have the three columns we want to work with: 'plate_x', 'plate_z', and 'type'.

Let’s remove every row that has a NaN in any of those columns.

You can do this by calling the dropna function. This function can take a parameter named subset which should be a list of the columns you’re interested in.

For example, the following code drops all of the NaN values from the columns 'A', 'B', and 'C'.
'''
data_frame = data_frame.dropna(subset = ['A', 'B', 'C'])
'''
Fill in the names of the columns that you don’t want NaN values in:
'''
aaron_judge = aaron_judge.dropna(subset = [____, ____, ____])
'''
8.
We now have points to plot using Matplotlib. Call plt.scatter() using five parameters:

The parameter x should be the plate_x column.
The parameter y should be the plate_z column.
To color the points correctly, the parameter c should be the type column.
To make the strikes red and the balls blue, set the cmap parameter to plt.cm.coolwarm.
To make the points slightly transparent, set the alpha parameter to 0.25.
Call plt.show() to see your graph.

plate_z measures how high off the ground the pitch was. If plate_z = 0, that means the pitch was at ground level when it got to the home plate.

'''
plt.scatter(x = aaron_judge['plate_x'], y = ____, c = ____, cmap = plt.cm.coolwarm, alpha = 0.5)
'''
Building the SVM
9.
Now that we’ve seen the location of every pitch, let’s create an SVM to create a decision boundary. This decision boundary will be the real strike zone for that player. For this section, make sure to write all of your code below the call to the scatter function but above the show function.

To begin, we want to validate our model, so we need to split the data into a training set and a validation set.

Call the train_test_split function using aaron_judge as a parameter.

Set the parameter random_state equal to 1 to ensure your data is split in the same way as our solution code.

This function returns two objects. Store the return values in variables named training_set and validation_set.


Finish the code block below:
'''
training_set, ____ = train_test_split(____, random_state = 1)
'''
10.
Next, create an SVC named classifier with kernel = 'rbf'. For right now, don’t worry about setting the C or gamma parameters.


The SVC should have kernel = 'rbf'.

11.
Call classifier‘s .fit() method. This method should take two parameters:

The training data. This is the plate_x column and the plate_z column in training_set.
The labels. This is the type column in training_set.
The code below shows and example of selecting two columns from a DataFrame:
'''
two_columns = data_frame[['A', 'B']]
'''
The first parameter should be training_set[['plate_x', 'plate_z']].

The second parameter should be training_set['type'].

12.
To visualize the SVM, call the draw_boundary function. This is a function that we wrote ourselves - you won’t find it in scikit-learn.

This function takes two parameters:

The axes of your graph. For us, this is the ax variable that we defined at the top of your code.
The trained SVM. For us, this is classifier. Make sure you’ve called .fit() before trying to visualize the decision boundary.
Run your code to see the predicted strike zone!

Note that the decision boundary will be drawn based on the size of the current axes. So if you call draw_boundary before calling scatter function, you will only see the boundary as a small square.

To get around this, you could manually set the size of the axes by using something likeax.set_ylim(-2, 2) before calling draw_boundary.


Call the following line of code after training the model but before calling plt.show().

draw_boundary(ax, classifier)
Optimizing the SVM
13.
Nice work! We’re now able to see the strike zone. But we don’t know how accurate our classifier is yet. Let’s find its accuracy by calling the .score() method and printing the results.

.score() takes two parameters — the points in the validation set and the labels associated with those points.

These two parameters should be very similar to the parameters used in .fit().


Finish the line of code below:

print(classifier.score(validation_set[['plate_x', 'plate_z']], ______))
14.
Let’s change some of the SVM’s parameters to see if we can get better accuracy.

Set the parameters of the SVM to be gamma = 100 and C = 100.

This will overfit the data, but it will be a good place to start. Run the code to see the overfitted decision boundary. What’s the new accuracy?


When you create SVC, set gamma = 100 and C = 100.

15.
Try to find a configuration of gamma and C that greatly improves the accuracy. You may want to use nested for loops.

Loop through different values of gamma and C and print the accuracy using those parameters. Our best SVM had an accuracy of 83.41%. Can you beat ours?


We used gamma = 3 and C = 1 to get 83.41%.

Explore Other Players
16.
Finally, let’s see how different players’ strike zones change. Aaron Judge is the tallest player in the MLB. Jose Altuve is the shortest player. Instead of using the aaron_judge variable, use jose_altuve.

To make this easier, you might want to consider putting all of your code inside a function and using the dataset as a parameter.

We’ve also imported david_ortiz.

Note that the range of the axes will change for these players. To really compare the strike zones, you may want to force the axes to be the same.

Try putting ax.set_ylim(-2, 6) and ax.set_xlim(-3, 3) right before calling plt.show()

17.
See if you can make an SVM that is more accurate by using more features. Perhaps the location of the ball isn’t the only important feature!

You can see the columns available to you by printing aaron_judge.columns.

For example, try adding the strikes column to your SVM — the number of strikes the batter already has might have an impact on whether the next pitch is a strike or a ball.

Note that our draw_boundary function won’t work if you have more than two features. If you add more features, make sure to comment that out!

Try to make the best SVM possible and share your results with us!
'''

#------------------Script.py---------------
import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

#print(type(aaron_judge))
#print(aaron_judge.columns)
#print(aaron_judge.columns.unique())
#print(aaron_judge.type)
#print(aaron_judge.type.unique())

aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})
#print(aaron_judge.type.unique())

#print(aaron_judge.plate_x)
#print(len(aaron_judge))
aaron_judge= aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])
#print(len(aaron_judge))

y = aaron_judge['type']
plt.scatter(x = aaron_judge['plate_x'], y = aaron_judge['plate_z'], c = y, alpha = 0.25, cmap = plt.cm.coolwarm )

training_set, validation_set = train_test_split(aaron_judge, random_state = 1)
#print(len(aaron_judge))
#print(len(training_set))

classifier = SVC(kernel = 'rbf', gamma = 3, C = 1 )
#training_data = [training_set['plate_x'],training_set['plate_z']]
training_data = training_set[['plate_x', 'plate_z']]
training_labels= training_set['type']

validation_data = validation_set[['plate_x', 'plate_x']]
validation_labels = validation_set['type']

#print(training_data)
#print(training_labels.unique())#
#print(training_data['plate_z'].max())
#print(training_data['plate_x'].unique())
classifier.fit(training_data, training_labels)

ax.set_ylim(-2,6)
ax.set_xlim(-3, 3)
draw_boundary(ax, classifier)
plt.show()

#find optimized gamma and C
'''
for j in range(1,100, 10):
  for i in range(10):
    gamma = j * 0.01 + 0.001
    C = i * 0.09 + 0.01
    classifier = SVC(kernel = 'rbf', gamma = gamma, C = C )
    classifier.fit(training_data, training_labels)
    print('current gamma = {}, C = {}'.format(gamma,C))
    print(classifier.score(validation_data, validation_labels))
'''

#---------------player.py---------------
import pickle

aaron_judge = pickle.load( open( "aaron_judge.p", "rb" ) )
jose_altuve = pickle.load( open( "jose_altuve.p", "rb" ) )
david_ortiz = pickle.load( open( "david_ortiz.p", "rb" ) )

#---------------svm_visualization.py---------------
import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(ax, h=.02):
    # x_min, x_max = x.min() - 1, x.max() + 1
    # y_min, y_max = y.min() - 1, y.max() + 1
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):

    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)