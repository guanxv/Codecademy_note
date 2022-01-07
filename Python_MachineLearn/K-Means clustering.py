








K-MEANS CLUSTERING
Introduction to Clustering
Often, the data you encounter in the real world won’t have flags attached and won’t provide labeled answers to your question. Finding patterns in this type of data, unlabeled data, is a common theme in many machine learning applications. Unsupervised Learning is how we find patterns and structure in these data.

Clustering is the most well-known unsupervised learning technique. It finds structure in unlabeled data by identifying similar groups, or clusters. Examples of clustering applications are:

Recommendation engines: group products to personalize the user experience
Search engines: group news topics and search results
Market segmentation: group customers based on geography, demography, and behaviors
Image segmentation: medical imaging or road scene segmentation on self-driving cars
Let’s get started!'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np 

from os.path import join, dirname, abspath
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y = iris.target

fignum = 1

# Plot the ground truthd

fig = plt.figure(fignum, figsize=(4, 3))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Zombies', 0),
                    ('Programmers', 1),
                    ('Vampires', 2)]:
    ax.text3D(x[y == label, 3].mean(),
              x[y == label, 0].mean(),
              x[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results

y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(x[:, 3], x[:, 0], x[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_xlabel('hates sunlight')
ax.set_ylabel('likes garlic')
ax.set_zlabel('canine teeth (in)')

ax.set_title('')
ax.dist = 12

# Add code here:
plt.show()

'''
K-MEANS CLUSTERING
K-Means Clustering
The goal of clustering is to separate data so that data similar to one another are in the same group, while data different from one another are in different groups. So two questions arise:

How many groups do we choose?
How do we define similarity?
*K-Means * is the most popular and well-known clustering algorithm, and it tries to address these two questions.

The “K” refers to the number of clusters (groups) we expect to find in a dataset.
The “Means” refers to the average distance of data to each cluster center, also known as the centroid, which we are trying to minimize.
It is an iterative approach:

1.Place k random centroids for the initial clusters.
2.Assign data samples to the nearest centroid.
3.Update centroids based on the above-assigned data samples.
Repeat Steps 2 and 3 until convergence (when points don’t move between clusters and centroids stabilize).

Once we are happy with our clusters, we can take a new unlabeled datapoint and quickly assign it to the appropriate cluster.

In this lesson, we will first implement K-Means the hard way (to help you understand the algorithm) and then the easy way using the sklearn library!

'''
'''
K-MEANS CLUSTERING
Iris Dataset
Before we implement the K-means algorithm, let’s find a dataset. The sklearn package embeds some datasets and sample images. One of them is the Iris dataset.

The Iris dataset consists of measurements of sepals and petals of 3 different plant species:

Iris setosa
Iris versicolor
Iris virginica
Iris
The sepal is the part that encases and protects the flower when it is in the bud stage. A petal is a leaflike part that is often colorful.

From sklearn library, import the datasets module:

from sklearn import datasets
To load the Iris dataset:

iris = datasets.load_iris()
The Iris dataset looks like:

[[ 5.1  3.5  1.4  0.2 ]
 [ 4.9  3.   1.4  0.2 ]
 [ 4.7  3.2  1.3  0.2 ]
 [ 4.6  3.1  1.5  0.2 ]
   . . .
 [ 5.9  3.   5.1  1.8 ]]
We call each piece of data a sample. For example, each flower is one sample.

Each characteristic we are interested in is a feature. For example, petal length is a feature of this dataset.

The features of the dataset are:

Column 0: Sepal length
Column 1: Sepal width
Column 2: Petal length
Column 3: Petal width
The 3 species of Iris plants are what we are going to cluster later in this lesson.'''

'''
Instructions
1.
Import the datasets module and load the Iris data.

From sklearn library, import the datasets module, and load the Iris dataset:

from sklearn import datasets

iris = datasets.load_iris()
2.
Every dataset from sklearn comes with a bunch of different information (not just the data) and is stored in a similar fashion.

First, let’s take a look at the most important thing, the sample data:

print(iris.data)
Each row is a plant!

The Iris dataset looks like:

[[ 5.1  3.5  1.4  0.2 ]
 [ 4.9  3.   1.4  0.2 ]
 [ 4.7  3.2  1.3  0.2 ]
 [ 4.6  3.1  1.5  0.2 ]
   . . .
 [ 5.9  3.   5.1  1.8 ]]
3.
Since the datasets in sklearn datasets are used for practice, they come with the answers (target values) in the target key:

Take a look at the target values:

print(iris.target)
The iris.target values give the ground truth for the Iris dataset. Ground truth, in this case, is the number corresponding to the flower that we are trying to learn.

The ground truth is what’s measured for the target variable for the training and testing examples.

It should look like:

[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
  2 2 ]
Each one is the cluster label for an Iris plant.

4.
It is always a good idea to read the descriptions of the data:

print(iris.DESCR)
Expand the terminal (right panel):

When was the Iris dataset published?
What is the unit of measurement?
DESCR needs to be capitalized.

This dataset was published in 1936, over eighty years ago:

Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936)
The unit of measurement is cm (centimeter):

    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        
        '''
        
import codecademylib3_seaborn
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

print(iris.data)

print(iris.target)

print(iris.DESCR)

'''
K-MEANS CLUSTERING
Visualize Before K-Means
To get a better sense of the data in the iris.data matrix, let’s visualize it!

With Matplotlib, we can create a 2D scatter plot of the Iris dataset using two of its features (sepal length vs. petal length). The sepal length measurements are stored in column 0 of the matrix, and the petal length measurements are stored in column 2 of the matrix.

But how do we get these values?

Suppose we only want to retrieve the values that are in column 0 of a matrix, we can use the NumPy/Pandas notation [:,0] like so:

matrix[:,0]
[:,0] can be translated to [all_rows , column_0]

Once you have the measurements we need, we can make a scatter plot by:

plt.scatter(x, y)
To show the plot:

plt.show()
Let’s try this! But this time, plot the sepal length (column 0) vs. sepal width (column 1) instead.'''
'''

Instructions
1.
Store iris.data in a variable named samples.

samples = iris.data
2.
Create a list named x that contains the column 0 values of samples.

Create a list named y that contains the column 1 values of samples.

x = samples[:, 0]
y = samples[:, 1]
So now, x contains all the sepal length measurements and y contains all the sepal width measurements.

3.
Use the .scatter() function to create a scatter plot of x and y.

Because some of the data samples have the exact same features, let’s add alpha=0.5:

plt.scatter(x, y, alpha=0.5)
Adding alpha=0.5 makes some points look darker than others. The darker spots are where there is overlap.

4.
Call the .show() function to display the graph.

If you didn’t know there are three species of the Iris plant, would you have known just by looking at the visualization?

Answer:

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

plt.scatter(x, y, alpha=0.5)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
We’ve also x-axis label and y-axis label (for good practice!)

Adding alpha=0.5 makes some points look darker than others. The darker spots are where there is overlap.'''


import codecademylib3_seaborn
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

# Store iris.data

samples = iris.data

x = []
x = samples[:,0]

y = []
y = samples[:,1]

plt.scatter(x,y, alpha=0.2)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')


plt.show()

'''
K-MEANS CLUSTERING
Implementing K-Means: Step 1
The K-Means algorithm:

Place k random centroids for the initial clusters.
Assign data samples to the nearest centroid.
Update centroids based on the above-assigned data samples.
Repeat Steps 2 and 3 until convergence.

After looking at the scatter plot and having a better understanding of the Iris data, let’s start implementing the K-Means algorithm.

In this exercise, we will implement Step 1.

Because we expect there to be three clusters (for the three species of flowers), let’s implement K-Means where the k is 3.

Using the NumPy library, we will create 3 random initial centroids and plot them along with our samples.'''
'''
Instructions
1.
First, create a variable named k and set it to 3.

2.
Then, use NumPy’s random.uniform() function to generate random values in two lists:

a centroids_x list that will have k random values between min(x) and max(x)
a centroids_y list that will have k random values between min(y) and max(y)
The random.uniform() function looks like:

np.random.uniform(low, high, size)
The centroids_x will have the x-values for our initial random centroids and the centroids_y will have the y-values for our initial random centroids.

3.
Create an array named centroids and use the zip() function to add centroids_x and centroids_y to it.

The zip() function looks like:

np.array(list(zip(array1, array2)))
Then, print centroids.

The centroids list should now have all the initial centroids.

centroids = np.array(list(zip(centroids_x, centroids_y)))

print(centroids)
The output should look like:

[[5.49815832 3.5073056 ]
 [7.72370927 4.2138989 ]
 [6.64764806 4.10084725]]
Your centroids array will have slightly different values since we are randomly initializing the centroids!

zip() takes two (or more) lists as inputs and returns an object that contains a list of pairs. Each pair contains one element from each of the inputs.

4.
Make a scatter plot of y vs x.

Make a scatter plot of centroids_y vs centroids_x.

Show the plots to see your centroids!

Answer:'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

combined = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

plt.scatter(x, y, alpha=0.5)
plt.scatter(centroids_x, centroids_y)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()'''
Adding alpha=0.5 makes the points look darker than others. This is because some of the points might have the exact the same values. The dots are darker because they are stacked!
'''
'''
K-MEANS CLUSTERING
Implementing K-Means: Step 2
The K-Means algorithm:

Place k random centroids for the initial clusters.
Assign data samples to the nearest centroid.
Update centroids based on the above-assigned data samples.
Repeat Steps 2 and 3 until convergence.

In this exercise, we will implement Step 2.

Now we have the 3 random centroids. Let’s assign data points to their nearest centroids.

To do this we’re going to use the Distance Formula to write a distance() function. Then, we are going to iterate through our data samples and compute the distance from each data point to each of the 3 centroids.

Suppose we have a point and a list of three distances in distances and it looks like [15, 20, 5], then we would want to assign the data point to the 3rd centroid. The argmin(distances) would return the index of the lowest corresponding distance, 2, because the index 2 contains the minimum value.'''
'''
Instructions
1.
Write a distance() function.

It should be able to take in a and b and return the distance between the two points.

For 2D:

def distance(a, b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one+two) ** 0.5
  return distance
2.
Create an array called labels that will hold the cluster labels for each data point. Its size should be the length of the data sample.

It should look something like:

[ 0.  0.  0.  0.  0.  0.  ...  0.]
Create an array called distances that will hold the distances for each centroid. It should have the size of k.

It should look something like:

[ 0.  0.  0.]
# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))

# Distances to each centroid
distances = np.zeros(k)
3.
To assign each data point to the closest centroid, we need to iterate through the whole data sample and calculate each data point’s distance to each centroid.

We can get the index of the smallest distance of distances by doing:

cluster = np.argmin(distances)
Then, assign the cluster to each index of the labels array.

The code should look something like:

for i in range(len(samples)):
  distances[0] = distance(sepal_length_width[i], centroids[0])
  # same as above for distance to centroids[1]
  # same as above for distance to centroids[2]
  cluster = np.argmin(distances)
  labels[i] = cluster
4.
Then, print labels (outside of the for loop).

Awesome! You have just finished Step 2 of the K-means algorithm.

print(labels)
The result labels should look like:

[ 0.  0.  0.  1.  0.  2. 0.  1.  1.  ... ]'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Step 2: Assign samples to nearest centroid




# Distance formula
def distance(a,b):
  distance = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
  return distance

# Cluster labels for each point (either 0, 1, or 2)

#labels = [0 for i in range(len(x))]
labels = np.zeros(len(samples))

# Distances to each centroid

#distances = [0 for i in range(len(centroids_x))]
distances = np.zeros(k)

# Assign to the closest centroid

for i in range(len(x)):
  distances[0] = distance(sepal_length_width[i],centroids[0])
  distances[1] = distance(sepal_length_width[i],centroids[1])
  distances[2] = distance(sepal_length_width[i],centroids[2])
  cluster = np.argmin(distances)
  labels[i]= cluster


# Print labels
print(labels)
'''
K-MEANS CLUSTERING
Implementing K-Means: Step 3
The K-Means algorithm:

Place k random centroids for the initial clusters.
Assign data samples to the nearest centroid.
Update centroids based on the above-assigned data samples.
Repeat Steps 2 and 3 until convergence.

In this exercise, we will implement Step 3.

Find new cluster centers by taking the average of the assigned points. To find the average of the assigned points, we can use the .mean() function.'''

'''
Instructions
1.
Save the old centroids value before updating.

We have already imported deepcopy for you:

from copy import deepcopy
Store centroids into centroids_old using deepcopy():

centroids_old = deepcopy(centroids)
To understand more about the deepcopy() method, read the Python documentation.

2.
Then, create a for loop that iterates k times.

Since k = 3, as we are iterating through the forloop each time, we can calculate mean of the points that have the same cluster label.

Inside the for loop, create an array named points where we get all the data points that have the cluster label i.

There are two ways to do this, check the hints to see both!

One way to do this is:

for i in range(k):
  points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
Another way is to use nested for loop:

for i in range(k):
  points = []
  for j in range(len(sepal_length_width)):
    if labels[j] == i:
      points.append(sepal_length_width[j])
Here, we create an empty list named points first, and use .append() to add values into the list.

3.
Then (still inside the for loop), calculate the mean of those points using .mean() to get the new centroid.

Store the new centroid in centroids[i].

The .mean() fucntion looks like:

np.mean(input, axis=0)
for i in range(k):
  ...
  centroids[i] = np.mean(points, axis=0)
If you don’t have axis=0 parameter, the default is to compute the mean of the flattened array. We need the axis=0 here to specify that we want to compute the means along the rows.

4.
Oustide of the for loop, print centroids_old and centroids to see how centroids changed.

print(centroids_old)
print("- - - - - - - - - - - - - -")
print(centroids)'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()

samples = iris.data
samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Step 2: Assign samples to nearest centroid

def distance(a, b):
  one = (a[0] - b[0]) **2
  two = (a[1] - b[1]) **2
  distance = (one+two) ** 0.5
  return distance

# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))

# Distances to each centroid
distances = np.zeros(k)

for i in range(len(samples)):
  distances[0] = distance(sepal_length_width[i], centroids[0])
  distances[1] = distance(sepal_length_width[i], centroids[1])
  distances[2] = distance(sepal_length_width[i], centroids[2])
  cluster = np.argmin(distances)
  labels[i] = cluster

# Step 3: Update centroids
centroids_old = deepcopy(centroids)

for i in range(k):
  points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
  centroids[i] = np.mean(points, axis=0)

print(centroids_old)

print(centroids)

'''
K-MEANS CLUSTERING
Implementing K-Means: Step 4
The K-Means algorithm:

Place k random centroids for the initial clusters.
Assign data samples to the nearest centroid.
Update centroids based on the above-assigned data samples.
Repeat Steps 2 and 3 until convergence.

In this exercise, we will implement Step 4.

This is the part of the algorithm where we repeatedly execute Step 2 and 3 until the centroids stabilize (convergence).

We can do this using a while loop. And everything from Step 2 and 3 goes inside the loop.

For the condition of the while loop, we need to create an array named errors. In each error index, we calculate the difference between the updated centroid (centroids) and the old centroid (centroids_old).

The loop ends when all three values in errors are 0.'''
'''

Instructions
1.
On line 40 of script.py, initialize error:

error = np.zeros(3)
Then, use the distance() function to calculate the distance between the updated centroid and the old centroid and put them in error:

error[0] = distance(centroids[0], centroids_old[0])
# do the same for error[1]
# do the same for error[2]
error = np.zeros(3)

error[0] = distance(centroids[0], centroids_old[0])
error[1] = distance(centroids[1], centroids_old[1])
error[2] = distance(centroids[2], centroids_old[2])
2.
After that, add a while loop:

while error.all() != 0:
And move everything below (from Step 2 and 3) inside.

And recalculate error again at the end of each iteration of the while loop:

error[0] = distance(centroids[0], centroids_old[0])
# do the same for error[1]
# do the same for error[2]

#hints

while error.all() != 0:

  # Step 2: Assign samples to nearest centroid

  for i in range(len(samples)):
    distances[0] = distance(sepal_length_width[i], centroids[0])
    distances[1] = distance(sepal_length_width[i], centroids[1])
    distances[2] = distance(sepal_length_width[i], centroids[2])
    cluster = np.argmin(distances)
    labels[i] = cluster

  # Step 3: Update centroids

  centroids_old = deepcopy(centroids)

  for i in range(3):
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    centroids[i] = np.mean(points, axis=0)

  # Add this again:

  error[0] = distance(centroids[0], centroids_old[0])
  error[1] = distance(centroids[1], centroids_old[1])
  error[2] = distance(centroids[2], centroids_old[2])
  
3.
Awesome, now you have everything, let’s visualize it.

After the while loop finishes, let’s create an array of colors:

colors = ['r', 'g', 'b']
Then, create a for loop that iterates k times.

Inside the for loop (similar to what we did in the last exercise), create an array named points where we get all the data points that have the cluster label i.

Then we are going to make a scatter plot of points[:, 0] vs points[:, 1] using the scatter() function:

plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

hints


colors = ['r', 'g', 'b']

for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)
  
  
4.
Then, paste the following code at the very end. Here, we are visualizing all the points in each of the labels a different color.

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y)))

# Step 1: Place K random centroids

k = 3

centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

def distance(a, b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one + two) ** 0.5
  return distance

# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)

# Cluster labeles (either 0, 1, or 2)
labels = np.zeros(len(samples))

distances = np.zeros(3)

# Initialize error:
error = np.zeros(3)
error[0] = distance(centroids[0], centroids_old[0])
error[1] = distance(centroids[1], centroids_old[1])
error[2] = distance(centroids[2], centroids_old[2])

while error.all() != 0:


  # Repeat Steps 2 and 3 until convergence:


  # Step 2: Assign samples to nearest centroid

  for i in range(len(samples)):
    distances[0] = distance(sepal_length_width[i], centroids[0])
    distances[1] = distance(sepal_length_width[i], centroids[1])
    distances[2] = distance(sepal_length_width[i], centroids[2])
    cluster = np.argmin(distances)
    labels[i] = cluster

  # Step 3: Update centroids

  centroids_old = deepcopy(centroids)

  for i in range(3):
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]
    centroids[i] = np.mean(points, axis=0)

  error[0] = distance(centroids[0], centroids_old[0])
  error[1] = distance(centroids[1], centroids_old[1])
  error[2] = distance(centroids[2], centroids_old[2])

colors = ['r', 'g', 'b']

for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

  plt.scatter(centroids[:, 0], centroids[:, 1], c=colors[i],marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
 
'''
K-MEANS CLUSTERING
Implementing K-Means: Scikit-Learn
Awesome, you have implemented K-Means clustering from scratch!

Writing an algorithm whenever you need it can be very time-consuming and you might make mistakes and typos along the way. We will now show you how to implement K-Means more efficiently – using the scikit-learn library.

Instead of implementing K-Means from scratch, the sklearn.cluster module has many methods that can do this for you.

To import KMeans from sklearn.cluster:

from sklearn.cluster import KMeans
For Step 1, use the KMeans() method to build a model that finds k clusters. To specify the number of clusters (k), use the n_clusters keyword argument:

model = KMeans(n_clusters = k)
For Steps 2 and 3, use the .fit() method to compute K-Means clustering:

model.fit(X)
After K-Means, we can now predict the closest cluster each sample in X belongs to. Use the .predict() method to compute cluster centers and predict cluster index for each sample:

model.predict(X)

'''
'''
1.
First, import KMeans from sklearn.cluster.

Answer:

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data
2.
Somewhere after samples = iris.data, use KMeans() to create an instance called model to find 3 clusters.

To specify the number of clusters, use the n_clusters keyword argument.

model = KMeans(n_clusters=3)
3.
Next, use the .fit() method of model to fit the model to the array of points samples:

model.fit(samples)
4.
After you have the “fitted” model, determine the cluster labels of samples.

Then, print the labels.

labels = model.predict(samples)

print(labels)'''


import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets

# From sklearn.cluster, import KMeans class
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters = 3)

# Use .fit() to fit the model to samples
model.fit(samples)

# Use .predict() to determine the labels of samples 
labels = model.predict(samples)

# Print the labels
print(labels)

'''
K-MEANS CLUSTERING
New Data?
You used K-Means and found three clusters of the samples data. But it gets cooler!

Since you have created a model that computed K-Means clustering, you can now feed new data samples into it and obtain the cluster labels using the .predict() method.

So, suppose we went to the florist and bought 3 more Irises with the measurements:

[[ 5.1  3.5  1.4  0.2 ]
 [ 3.4  3.1  1.6  0.3 ]
 [ 4.9  3.   1.4  0.2 ]]
We can feed this new data into the model and obtain the labels for them.'''
'''
Instructions
1.
First, store the 2D matrix:

new_samples = np.array([[5.7, 4.4, 1.5, 0.4],
   [6.5, 3. , 5.5, 0.4],
   [5.8, 2.7, 5.1, 1.9]])
To test if it worked, print the new_samples.

2.
Use the model to predict labels for the new_samples, and print the predictions.

The output might look like:

[0 2 2]
Those are the predicted labels for our three new flowers. If you are seeing different labels, don’t worry! Since the cluster centroids are randomly initialized, running the model repeatedly can produce different clusters with the same input data.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

model = KMeans(n_clusters=3)

model.fit(samples)

# Store the new Iris measurements
new_samples = np.array([
  [5.7, 4.4, 1.5, 0.4],
  [6.5, 3. , 5.5, 0.4],
  [5.8, 2.7, 5.1, 1.9]])

# Predict labels for the new_samples
labels = model.predict(new_samples)
print(labels)

'''
Question
In the context of this exercise, does feeding new data when running predict() update the predictor?

Answer
No, when we feed in new data samples using the predict() method of our KMeans model, the predictor does not get changed or updated.

When running the predict() method, it will return its best guess, based on what it learned in the previous steps. When we pass in the new data samples and obtain its predictions, we do not explicitly tell the predictor whether its guess was correct or not, so there is no way for it to update itself based on its guesses. Once it has been fitted to its sample test data, it will not be able to update itself to new data, unless we redo the fitting process.
'''


'''
K-MEANS CLUSTERING
Visualize After K-Means
We have done the following using sklearn library:

Load the embedded dataset
Compute K-Means on the dataset (where k is 3)
Predict the labels of the data samples
And the labels resulted in either 0, 1, or 2.

Let’s finish it by making a scatter plot of the data again!

This time, however, use the labels numbers as the colors.

To edit colors of the scatter plot, we can set c = labels:

plt.scatter(x, y, c=labels, alpha=0.5)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
'''
'''
Instructions
1.
Create an array called x that contains the Column 0 of samples.

Create an array called y that contains the Column 1 of samples.

x = samples[:,0]
y = samples[:,1]
2.
Make a scatter plot of x and y, using labels to define the colors.

plt.scatter(x, y, c=labels, alpha=0.5)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

model = KMeans(n_clusters=3)

model.fit(samples)

# Store the new Iris measurements
new_samples = np.array([
  [5.7, 4.4, 1.5, 0.4],
  [6.5, 3. , 5.5, 0.4],
  [5.8, 2.7, 5.1, 1.9]])

# Predict labels for the new_samples
labels = model.predict(new_samples)
print(labels)

x = samples[:,0]
y = samples[:,1]

plt.scatter(x,y,c=labels, alpha = 0.5)


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()

'''
K-MEANS CLUSTERING
Evaluation
At this point, we have clustered the Iris data into 3 different groups (implemented using Python and using scikit-learn). But do the clusters correspond to the actual species? Let’s find out!

First, remember that the Iris dataset comes with target values:

target = iris.target
It looks like:

[ 0 0 0 0 0 ... 2 2 2]
According to the metadata:

All the 0‘s are Iris-setosa
All the 1‘s are Iris-versicolor
All the 2‘s are Iris-virginica
Let’s change these values into the corresponding species using the following code:

species = np.chararray(target.shape, itemsize=150)

for i in range(len(samples)):
  if target[i] == 0:
    species[i] = 'setosa'
  elif target[i] == 1:
    species[i] = 'versicolor'
  elif target[i] == 2: 
    species[i] = 'virginica'
Then we are going to use the Pandas library to perform a cross-tabulation.

Cross-tabulations enable you to examine relationships within the data that might not be readily apparent when analyzing total survey responses.

The result should look something like:

labels    setosa    versicolor    virginica
0             50             0            0
1              0             2           36
2              0            48           14
(You might need to expand this narrative panel in order to the read the table better.)

The first column has the cluster labels. The second to fourth columns have the Iris species that are clustered into each of the labels.

By looking at this, you can conclude that:

Iris-setosa was clustered with 100% accuracy.
Iris-versicolor was clustered with 96% accuracy.
Iris-virginica didn’t do so well.
Follow the instructions below to learn how to do a cross-tabulation.'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd

iris = datasets.load_iris()

samples = iris.data

target = iris.target

model = KMeans(n_clusters=3)

model.fit(samples)

labels = model.predict(samples)

# Code starts here:
species = np.chararray(target.shape, itemsize=150)

for i in range(len(samples)):
  if target[i] == 0:
    species[i] = 'setosa'
  elif target[i] == 1:
    species[i] = 'versicolor'
  elif target[i] == 2: 
    species[i] = 'virginica'

df = pd.DataFrame({'labels': labels, 'species': species})

print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)

'''
K-MEANS CLUSTERING
The Number of Clusters
At this point, we have grouped the Iris plants into 3 clusters. But suppose we didn’t know there are three species of Iris in the dataset, what is the best number of clusters? And how do we determine that?

Before we answer that, we need to define what is a good cluster?

Good clustering results in tight clusters, meaning that the samples in each cluster are bunched together. How spread out the clusters are is measured by inertia. Inertia is the distance from each sample to the centroid of its cluster. The lower the inertia is, the better our model has done.

You can check the inertia of a model by:

print(model.inertia_)
For the Iris dataset, if we graph all the ks (number of clusters) with their inertias:

Optimal Number of Clusters
Notice how the graph keeps decreasing.

Ultimately, this will always be a trade-off. The goal is to have low inertia and the least number of clusters.

One of the ways to interpret this graph is to use the elbow method: choose an “elbow” in the inertia plot - when inertia begins to decrease more slowly.

In the graph above, 3 is the optimal number of clusters.'''
'''
Instructions
1.
First, create two lists:

num_clusters that has values from 1, 2, 3, … 8
inertias that is empty
Answer:

num_clusters = list(range(1, 9))
inertias = []
2.
Then, iterate through num_clusters and calculate K-means for each number of clusters.

Add each of their inertias into the inertias list.

for k in num_clusters:
  model = KMeans(n_clusters=k)
  model.fit(samples)
  inertias.append(model.inertia_)
3.
Plot the inertias vs num_clusters:

plt.plot(num_clusters, inertias, '-o')

plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.show()'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

# Code Start here:

#num_clusters = [i+1 for i in list(range(8))]
num_clusters = list(range(1, 9))

print(num_clusters)

inertias = []

for k in num_clusters:
  model = KMeans(n_clusters=k)
  moodel.fit(samples)
  inertias.append(model.inertia_)

plt.plot(num_clusters, inertias, '-o')

plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.show()

'''
Question
In the context of this exercise, how is inertia calculated?

Answer
According to the documentation for the KMeans method, the inertia_ attribute is the sum of squared distances of the samples to their nearest centroids.

So, to obtain the value of the inertia, we would obtain each data points’ distance to its nearest centroid, square this distance, and then sum them all together, which gives us the inertia. We will utilize the Euclidean, or geometric, distance formula to calculate this.

The following is a general overview of how we might calculate the inertia.
'''
inertia = 0

for datapoint in dataset:
  # Obtain the nearest centroid of the point.
  centroid = get_centroid(datapoint)

  # Calculate the distance from each datapoint to its centroid
  # using the Euclidean distance formula.
  delta_x = datapoint.x - centroid.x
  delta_y = datapoint.y - centroid.y
  distance = (delta_x ** 2 + delta_y ** 2) ** 0.5

  # Square the distance, and add to the inertia.
  squared_distance = distance ** 2
  inertia += squared_distance
  '''
  '''
'''
K-MEANS CLUSTERING
Try It On Your Own
Now it is your turn!

In this review section, find another dataset from one of the following:

The scikit-learn library (http://scikit-learn.org/stable/datasets/index.html)
UCI Machine Learning Repo (https://archive.ics.uci.edu/ml/index.php)
Codecademy GitHub Repo (coming soon!)
Import the pandas library as pd:

import pandas as pd
Load in the data with read_csv():

digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
Note that if you download the data like this, the data is already split up into a training and a test set, indicated by the extensions .tra and .tes. You’ll need to load in both files.

With the command above, you only load in the training set.

Happy Coding!
'''

import pandas as pd
digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)

labels = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#print(digits)

#find the right clusters
num_clusters = list(range(1,20))

inertias = []

for k in num_clusters:
  model = KMeans(n_clusters=k)
  model.fit(digits)
  inertias.append(model.inertia_)

plt.plot(num_clusters, inertias, '-o')

plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

plt.show()

print(num_clusters, inertias)

'''
Question
In the context of this lesson 2, what are some other examples of clustering applications?

Answer
Generally, clustering applies when we want to separate and group data based on similar features.

Clustering can also apply to applications that match people together, such as dating sites, or services to connect professionals, which can utilize clustering to determine recommended connections.

Clustering can also apply to personality tests, like the Myers Briggs personality test, which take certain responses and group you into a certain category based on those responses.

In addition, clustering can also apply when grouping organisms based on their physical traits, such that if you provide information, say for a dog, it should categorize it as that kind of animal.'''



''' Project
MACHINE LEARNING: UNSUPERVISED LEARNING 🤖
Handwriting Recognition using K-Means
The U.S. Postal Service has been using machine learning and scanning technologies since 1999. Because its postal offices

ATMs can recognize handwritten bank checks
Evernote can recognize handwritten task lists
Expensify can recognize handwritten receipts
But how do they do it?

In this project, you will be using K-means clustering (the algorithm behind this magic) and scikit-learn to cluster images of handwritten digits.

Let’s get started!

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.


1.
The sklearn library comes with a digits dataset for practice.

In script.py, we have already added three lines of code:

import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
From sklearn library, import the datasets module.

Then, load in the digits data using .load_digits() and print digits.


Hint
At this point, your code should look like:

import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
print(digits)
The terminal output should display all the information that comes with the dataset.

2.
When first starting out with a dataset, it’s always a good idea to go through the data description and see what you can already learn.

Instead of printing the digits, print digits.DESCR.

What is the size of an image (in pixel)?
Where is this dataset from?

Hint
Print out the description of the digits data:

print(digits.DESCR)
The result should look like:

Recognition of Handwritten Digits Data Set
==========================================

Notes
-----

Data Set Characteristics:

  :Number of Instances: 5620
  :Number of Attributes: 64
  :Attribute Information: 8x8 image of integer pixels in the range 0-16
  :Missing Attribute Values: None
  :Creator: E. Alpaydin
  :Date: July; 1998
The digit images are 8 x 8. And the dataset is from Bogazici University (Istanbul, Turkey).

3.
Let’s see what the data looks like!

Print digits.data.


Hint
Print out the data:

print(digits.data)
[[ 0.  0.  5. ...,  0.  0.  0. ]
 [ 0.  0.  0. ..., 10. 0.  0. ]
 [ 0.  0.  0. ..., 16.  9.  0. ]
... 
Each list contains 64 values which respent the pixel colors of an image (0-16):

0 is white
16 is black
4.
Next, print out the target values in digits.target.


Hint
Print out the target values:

print(digits.target)
The result should look like:

[ 0 1 2 ..., 8 9 8]
This shows us that the first data point in the set was tagged as a 0 and the last one was tagged as an 8.

5.
To visualize the data images, we need to use Matplotlib. Let’s visualize the image at index 100:

plt.gray() 

plt.matshow(digits.images[100])

plt.show()
The image should look like:

4

Is it a 4? Let’s print out the target label at index 100 to find out!

print(digits.target[100])
Open the hint to see how you can visualize more than one image.


Hint
To take a look at 64 sample images. Copy and paste the code below:

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()
K-Means Clustering:
6.
Now we understand what we are working with. Let’s cluster the 1797 different digit images into groups.

Import KMeans from sklearn.cluster.


Hint
from sklearn.cluster import KMeans
7.
What should be the k, the number of clusters, here?

Use the KMeans() method to build a model that finds k clusters.


Hint
Because there are 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, and 9), there should be 10 clusters.

So k, the number of clusters, is 10:

model = KMeans(n_clusters=10, random_state=42)
The random_state will ensure that every time you run your code, the model is built in the same way. This can be any number. We used random_state = 42.

8.
Use the .fit() method to fit the digits.data to the model.


Hint
model.fit(digits.data)
Visualizing after K-Means:
9.
Let’s visualize all the centroids! Because data samples live in a 64-dimensional space, the centroids have values so they can be images!

First, add a figure of size 8x3 using .figure().

Then, add a title using .suptitle().


Hint
fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')
10.
Scikit-learn sometimes calls centroids “cluster centers”.

Write a for loop to displays each of the cluster_centers_ like so:

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
The cluster centers should be a list with 64 values (0-16). Here, we are making each of the cluster centers into an 8x8 2D array.

11.
Outside of the for loop, use .show() to display the visualization.

It should look like:

8

These are the centroids of handwriting from thirty different people collected by Bogazici University (Istanbul, Turkey):

Index 0 looks like 0
Index 1 looks like 9
Index 2 looks like 2
Index 3 looks like 1
Index 4 looks like 6
Index 5 looks like 8
Index 6 looks like 4
Index 7 looks like 5
Index 8 looks like 7
Index 9 looks like 3
Notice how the centroids that look like 1 and 8 look very similar and 1 and 4 also look very similar.


Hint
plt.show()
12.
Optional:

If you want to see another example that visualizes the data clusters and their centers using K-means, check out the sklearn‘s own example.

K-means clustering example


Hint
In this code, they use k-means++ to place the initial centroids.

Testing Your Model:
13.
Instead of feeding new arrays into the model, let’s do something cooler!

Inside the right panel, go to test.html.


Hint
https://localhost/test.html

14.
What year will robots take over the world?

Use your mouse to write a digit in each of the boxes and click Get Array.


Hint
2020?

15.
Back in script.py, create a new variable named new_samples and copy and paste the 2D array into it.

new_samples = np.array(      )

Hint
Copy and paste the entire code into the parentheses:

new_samples = np.array(      )
Make sure to even copy paste the outer square brackets.

16.
Use the .predict() function to predict new labels for these four new digits. Store those predictions in a variable named new_labels.


Hint
new_labels = model.predict(new_samples)

'''

print(new_labels)

 '''
17.
But wait, because this is a clustering algorithm, we don’t know which label is which.

By looking at the cluster centers, let’s map out each of the labels with the digits we think it represents:
'''
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
    
    '''

Hint
We did print(x, end='') so that all the digits are printed on the same line.

Index 0 looks like 0
Index 1 looks like 9
Index 2 looks like 2
Index 3 looks like 1
Index 4 looks like 6
Index 5 looks like 8
Index 6 looks like 4
Index 7 looks like 5
Index 8 looks like 7
Index 9 looks like 3
18.
Is the model recognizing your handwriting?

Remember, this model is trained on handwritten digits of 30 Turkish people (from the 1990’s).

Try writing your digits similar to these cluster centers:'''

#script.py---------------

import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets 

from sklearn.cluster import KMeans 

digits = datasets.load_digits()

#print(digits.DESCR)
#print(digits.data)
print(digits.target)

#plt.gray()
#plt.matshow(digits.images[100])
#plt.show()

#print(digits.target[100])

model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)

fig = plt.figure(figsize = (8,3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()


new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,2.82,5.65,6.10,3.82,0.84,4.50,5.19,7.32,7.63,6.57,6.87,5.34,1.30,6.03,6.11,4.43,2.07,0.30,7.02,4.81,0.00,0.00,0.00,0.00,0.00,4.20,7.63,1.60,0.00,0.00,0.00,0.00,1.60,7.48,5.12,0.00,0.00,0.00,0.00,0.00,1.52,4.96,0.77,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,5.04,7.55,6.03,1.45,0.00,0.00,0.00,0.00,7.47,5.87,6.72,7.10,0.46,0.00,0.00,0.15,7.63,3.66,1.75,7.63,3.13,0.00,0.00,1.07,7.62,2.75,0.08,5.96,6.71,0.08,0.00,1.52,7.63,2.21,0.00,2.82,7.63,1.45,0.00,0.08,2.75,0.23,0.00,2.90,7.63,1.22,0.00,0.00,0.00,0.00,0.00,4.58,7.17,0.08,0.00,0.00,0.00,0.00,0.00,4.19,7.40,0.00],
[0.00,0.00,0.00,1.15,6.41,6.41,2.98,0.00,0.00,0.00,0.61,6.87,6.87,5.88,7.63,2.60,0.00,0.00,4.66,7.48,1.37,0.08,6.87,5.04,0.00,1.53,7.62,4.27,0.00,0.00,6.72,5.12,0.00,1.52,5.80,0.38,0.00,1.76,7.63,3.14,0.00,0.00,0.00,0.00,0.00,4.28,7.33,0.31,0.00,0.00,0.00,0.00,0.00,5.65,6.11,0.00,0.00,0.00,0.00,0.00,0.15,7.10,4.89,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.46,5.34,7.63,7.55,5.34,1.22,0.00,0.00,5.34,7.40,3.97,4.04,7.17,7.55,0.00,1.60,7.63,3.66,0.00,0.07,2.67,7.40,0.00,0.91,4.27,0.30,0.69,7.02,7.63,6.94,0.00,0.00,0.00,0.00,1.53,7.63,3.82,0.38,0.00,0.00,0.00,0.00,0.61,6.94,5.19,0.00]
])


new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')


# test.html--------------------------

<html>
<head>
</head>
  
<body onload="InitThis();">
  
  <script src="//ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    
  <script type="text/javascript" src="JsCode.js"></script>
    <div align="center">
      
        <canvas id="myCanvas" width="80" height="80" style="border:2px solid black"></canvas>
        
      <canvas id="myCanvas2" width="80" height="80" style="border:2px solid black"></canvas>
      
      <canvas id="myCanvas3" width="80" height="80" style="border:2px solid black"></canvas>
      
      <canvas id="myCanvas4" width="80" height="80" style="border:2px solid black"></canvas>
      
        <br /><br />
        <button onclick="javascript:clearArea();return false;">Clear Area</button>
        Line width : <select id="selWidth">
            <option value="9">9</option>
            <option value="10">10</option>
            <option value="11">11</option>
            <option value="12">12</option>
            <option value="14" selected="selected">14</option>
            <option value="18">18</option>
        </select>
        Color : <select id="selColor">
            <option value="#141c3a">black</option>
            <option value="#6400e4">purple</option>
            <option value="#4b35ef" selected="selected">royal-blue</option>
            <option value="#fa4359">red</option>
            <option value="#37c3be">mint</option>
            <option value="#ffc107">yellow</option>
            <option value="#cccccc">gray</option>
        </select>
         
      <button onclick="javascript:array();return false;">Get Array</button>

      
    </div>
 
    <pre id="opening_bracket">
  </pre>
  
   <pre id="display">
   </pre>
  
  <pre id="display2">
  </pre>
  
   <pre id="display3">
  </pre>
  
   <pre id="display4">
  </pre>
  
  <pre id="closing_bracket">
  </pre>
  
  
  

</body>
</html>

#JsCode.js------------------------

var mousePressed = false;
var lastX, lastY;
var ctx;

function InitThis() {
  
  // ========= 1
  
    ctx = document.getElementById('myCanvas').getContext("2d");

    $('#myCanvas').mousedown(function (e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
        if (mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas').mouseleave(function (e) {
        mousePressed = false;
    });
 
   // =========== 2
  
   ctx2 = document.getElementById('myCanvas2').getContext("2d");

    $('#myCanvas2').mousedown(function (e) {
        mousePressed = true;
        Draw2(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas2').mousemove(function (e) {
        if (mousePressed) {
            Draw2(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas2').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas2').mouseleave(function (e) {
        mousePressed = false;
    });
  
  
  // 3==========
  
   ctx3 = document.getElementById('myCanvas3').getContext("2d");

    $('#myCanvas3').mousedown(function (e) {
        mousePressed = true;
        Draw3(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas3').mousemove(function (e) {
        if (mousePressed) {
            Draw3(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas3').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas3').mouseleave(function (e) {
        mousePressed = false;
    });
  
  
  // 4 =================
  
   ctx4 = document.getElementById('myCanvas4').getContext("2d");

    $('#myCanvas4').mousedown(function (e) {
        mousePressed = true;
        Draw4(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas4').mousemove(function (e) {
        if (mousePressed) {
            Draw4(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

    $('#myCanvas4').mouseup(function (e) {
        mousePressed = false;
    });
	    $('#myCanvas4').mouseleave(function (e) {
        mousePressed = false;
    });
  
  
  
}




function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        ctx.strokeStyle = $('#selColor').val();
        ctx.lineWidth = $('#selWidth').val();
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}

function Draw2(x, y, isDown) {
    if (isDown) {
        ctx2.beginPath();
        ctx2.strokeStyle = $('#selColor').val();
        ctx2.lineWidth = $('#selWidth').val();
        ctx2.lineJoin = "round";
        ctx2.moveTo(lastX, lastY);
        ctx2.lineTo(x, y);
        ctx2.closePath();
        ctx2.stroke();
    }
    lastX = x; lastY = y;
}

function Draw3(x, y, isDown) {
    if (isDown) {
        ctx3.beginPath();
        ctx3.strokeStyle = $('#selColor').val();
        ctx3.lineWidth = $('#selWidth').val();
        ctx3.lineJoin = "round";
        ctx3.moveTo(lastX, lastY);
        ctx3.lineTo(x, y);
        ctx3.closePath();
        ctx3.stroke();
    }
    lastX = x; lastY = y;
}


function Draw4(x, y, isDown) {
    if (isDown) {
        ctx4.beginPath();
        ctx4.strokeStyle = $('#selColor').val();
        ctx4.lineWidth = $('#selWidth').val();
        ctx4.lineJoin = "round";
        ctx4.moveTo(lastX, lastY);
        ctx4.lineTo(x, y);
        ctx4.closePath();
        ctx4.stroke();
    }
    lastX = x; lastY = y;
}
	
function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
    // clear ctx2
   ctx2.setTransform(1, 0, 0, 1, 0, 0);
    ctx2.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
      // clear ctx3
   ctx3.setTransform(1, 0, 0, 1, 0, 0);
    ctx3.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // clear ctx4
  
   ctx4.setTransform(1, 0, 0, 1, 0, 0);
    ctx4.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);


}



function array() {
 
  
   document.getElementById('opening_bracket').innerHTML = "["
  
   var imageData = ctx.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

//document.write(compressed)
  
  // round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  
  
  document.getElementById('display').innerHTML = "[" + compressed + "]" + ","
  
  
  
  
  
  
  
  // part 2
   var imageData = ctx2.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

//document.write(compressed)
  
// round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  


  document.getElementById('display2').innerHTML = "[" + compressed + "]" + ","
  
  
  
  
  
  // =============== part 3
  
  var imageData = ctx3.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

//document.write(compressed)
  
// round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  
  document.getElementById('display3').innerHTML = "[" + compressed + "]" + ","
  
  
  
  
  
  
  
  // =========== 4
  var imageData = ctx4.getImageData(0, 0, 80, 80);
  
   var data = imageData.data;
  
    for (var i = 0; i < data.length; i += 4) {
      
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
      // ctx.putImageData(imageData, 0, 0);
  };
    
   // logs an array of 10 x 10 x 4 (400 items)
   //document.write(data);
   
  var gray = [];
  
  for (var x = 0; x < data.length; x +=4 ) {
      gray.push(data[x]);
  };
  
  // document.write(gray)
  // 122, 122, 124, 0,   121, 122, 122, 122,   
  //document.write(gray.length)
  // 6400
  
  
  var first_digit = [];
  for (var y = 0; y < data.length; y+=4) {
    first_digit.push(data[y])
  }
 
 
 //document.write(first_digit)
 //document.write(first_digit.length)
 // 6400
  
 var compress = []
 var counter = 0;
 var sum = 0;
 var ten = 0;
 
 for (var z = 0; z < first_digit.length; z++) {
   
   sum = sum + first_digit[z];
   
   if (z % 100 === 0) {
     compress.push(sum/100);
     sum = 0;   
   }
       
 };

  function average(list){
 averageVal = 0
 for(var i = 0; i < list.length; i++){
   averageVal = averageVal + list[i]/list.length
 }
 return averageVal
}

squares = []
for(var i = 0; i < 64; i++){
 squares.push([]);

}

for(var y = 0; y < 80; y++) {

  for(var x = 0; x < 80; x++) {
    
    squares[parseInt(y/10) * 8 + parseInt(x/10)].push(first_digit[x + y * 80])
 
  }
  
}

compressed = []

squares.forEach(function(square){
 compressed.push(average(square)/16)
})

// round
for (var k = 0; k < compress.length; k++) {
  compressed[k] = compressed[k].toFixed(2);
}
  
//document.write(compressed)
  
  
  document.getElementById('display4').innerHTML = "[" + compressed + "]"
  
  document.getElementById('closing_bracket').innerHTML = "]"

}










#-----------------------------------------------

'''
K-MEANS++ CLUSTERING
Introduction to K-Means++
The K-Means clustering algorithm is more than half a century old, but it is not falling out of fashion; it is still the most popular clustering algorithm for Machine Learning.

However, there can be some problems with its first step. In the traditional K-Means algorithms, the starting postitions of the centroids are intialized completely randomly. This can result in suboptimal clusters.

In this lesson, we will go over another version of K-Means, known as the K-Means++ algorithm. K-Means++ changes the way centroids are initalized to try to fix this problem.
'''
'''
1.
Run the program in script.py to cluster Codecademy learners into two groups using K-Means and K-Means++.

The only difference between each algorithm is how the cluster centroids are initialized.

It’s hard to see, but the clusters are different. Look at the point at x=0.2 and y=1. On the top graph you should see a purple point, but on the bottom graph a yellow point.

Which one of these clusters is better? We have printed the model of each inertia in the workspace. The model with the lower inertia has more coherent clusters. You can think of the model with the lower inertia as being “better”.

Which model performs better clustering?

Continue to the next exercise to see why random initialization of centroids can result in poorer clusters.

For a recap, the K-Means algorithm is the following:

Place k random centroids for the initial clusters.
Assign data samples to the nearest centroid.
Update centroids based on the above-assigned data samples.
Repeat Steps 2 and 3 until convergence.

Convergence happens when centroids stabilize.

On a Codecademy HQ computer, the result looks something like:

Average Runtime: 0.04122559293
Worst Runtime: 0.14529825281
So why can the random initial placement of centroids result in slow convergence?

When you place initial centroids randomly, there are just way too many possibilities.

For example, what if when all the initial centroids are randomized to be located in the top left corner, but the clusters are around the bottom right corner of the dataset?

Well, the K-Means algorithm would take a little longer than if the initial centroids are placed in a more spaced out way.
'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import random
import timeit

mu = 1
std = 0.5

np.random.seed(100)

xs = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.75,std,100)), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100))

ys = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100)), np.random.normal(0.75,std,100))

values = list(zip(xs, ys))

model = KMeans(init='random', n_clusters=2)
results = model.fit_predict(values)
print("The inertia of model that randomly initialized centroids is " + str(model.inertia_))



colors = ['#6400e4', '#ffc740']
plt.subplot(211)
for i in range(2):
  points = np.array([values[j] for j in range(len(values)) if results[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.6)

plt.title('Codecademy Mobile Feedback - Centroids Initialized Randomly')

plt.xlabel('Learn Python')
plt.ylabel('Learn SQL')

plt.subplot(212)
model = KMeans( n_clusters=2)
results = model.fit_predict(values)
print("The inertia of model that initialized the centroids using KMeans++ is " + str(model.inertia_))



colors = ['#6400e4', '#ffc740']

for i in range(2):
  points = np.array([values[j] for j in range(len(values)) if results[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.6)

plt.title('Codecademy Mobile Feedback - Centroids Initialized Using KMeans++')

plt.xlabel('Learn Python')
plt.ylabel('Learn SQL')

plt.tight_layout()
plt.show()


'''
K-MEANS++ CLUSTERING
Poor Clustering
Suppose we have four data samples that form a rectangle whose width is greater than its height:

Data Points in Rectangle Shape
If you wanted to find two clusters (k = 2) in the data, which points would you cluster together? You might guess the points that align vertically cluster together, since the height of the rectangle is smaller than its width. We end up with a left cluster (purple points) and a right cluster (yellow points).

Optimal Cluster
Let’s say we use the regular K-Means algorithm to cluster the points, where the cluster centroids are initialized randomly. We get unlucky and those randomly initialized cluster centroids happen to be the midpoints of the top and bottom line segments of the rectangle formed by the four data points.

Random Centroid Placement
The algorithm would converge immediately, without moving the cluster centroids. Consequently, the two top data points are clustered together (yellow points) and the two bottom data points are clustered together (purple points).

Suboptimal Clusters
This is a suboptimal clustering because the width of the rectangle is greater than its height. The optimal clusters would be the two left points as one cluster and the two right points as one cluster, as we thought earlier.'''

'''
1.
Suppose we have four data samples with these values:

(1, 1)
(1, 3)
(4, 1)
(4, 3)
And suppose we perform K-means on this data where the k is 2 and the randomized 2 initial centroids are located at the following positions:

(2.5, 1)
(2.5, 3)
What do you think the result clusters would look like?

Run script.py to find out the answer.

The K-Means converged immediately without moving clusters in this example because the centroids are already stabilized (samples won’t move from one cluster to another).'''

#---------------------script.py-------------------------------

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy
from sklearn.cluster import KMeans 

x = [1, 1, 4, 4]
y = [1, 3, 1, 3]

values = np.array(list(zip(x, y)))

centroids_x = [2.5, 2.5]
centroids_y = [1, 3]

centroids = np.array(list(zip(centroids_x, centroids_y)))

model = KMeans(init=centroids, n_clusters=2)

# Initial centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=100)

results = model.fit_predict(values)

plt.scatter(x, y, c=results, alpha=1)

# Cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='v', s=100)

ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4])

plt.title('Unlucky Initialization')
plt.show()

'''
K-MEANS++ CLUSTERING
What is K-Means++?
To recap, the Step 1 of the K-Means algorithm is “Place k random centroids for the initial clusters”.

The K-Means++ algorithm replaces Step 1 of the K-Means algorithm and adds the following:

1.1 The first cluster centroid is randomly picked from the data points.
1.2 For each remaining data point, the distance from the point to its nearest cluster centroid is calculated.
1.3 The next cluster centroid is picked according to a probability proportional to the distance of each point to its nearest cluster centroid. This makes it likely for the next cluster centroid to be far away from the already initialized centroids.
Repeat 1.2 - 1.3 until k centroids are chosen.'''

'''
Instructions
In the web browser you can see the initialization of centroids by regular K-Means (randomly) and by K-Means++.

Notice that the centroids created by K-Means++ are more spaced out.

Make sure to scroll down to see the second graph!'''

''''
K-MEANS++ CLUSTERING
K-Means++ using Scikit-Learn
Using the scikit-learn library and its cluster module , you can use the KMeans() method to build an original K-Means model that finds 6 clusters like so:

model = KMeans(n_clusters=6, init='random')
The init parameter is used to specify the initialization and init='random' specifies that initial centroids are chosen as random (the original K-Means).

But how do you implement K-Means++?

There are two ways and they both require little change to the syntax:

Option 1: You can adjust the parameter to init='k-means++'.

test = KMeans(n_clusters=6, init='k-means++')
Option 2: Simply drop the parameter.

test = KMeans(n_clusters=6)
This is because that init=k-means++ is actually default in scikit-learn.'''

'''
1.
We’ve brought back our small example where we intentionally selected unlucky initial positions for the cluser centroids.

On line 22 where we create the model, change the init parameter to "k-means++" and see how the clusters change. Were we able to find optimal clusters?

Make sure to put "k-means++" in quotes!
'''

import codecademylib3_seaborn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy
from sklearn.cluster import KMeans 

x = [1, 1, 4, 4]
y = [1, 3, 1, 3]

values = np.array(list(zip(x, y)))

centroids_x = [2.5, 2.5]
centroids_y = [1, 3]

centroids = np.array(list(zip(centroids_x, centroids_y)))

model = KMeans(init="k-means++", n_clusters=2)
# Initial centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=100)

results = model.fit_predict(values)

plt.scatter(x, y, c=results, alpha=1)

# Cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='v', s=100)

ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 1, 2, 3, 4])

plt.title('K-Means++ Initialization')
plt.show()
print("The model's inertia is " + str(model.inertia_))

'''
K-MEANS++ CLUSTERING
Review
Congratulations, now your K-Means model is improved and ready to go!

K-Means++ improves K-Means by placing initial centroids more strategically. As a result, it can result in more optimal clusterings than K-Means.

It can also outperform K-Means in speed. If you get very unlucky initial centroids using K-Means, the algorithm can take a long time to converge. K-Means++ will often converge quicker!

You can implement K-Means++ with the scikit-learn library similar to how you implement K-Means.

The KMeans() function has an init parameter, which specifies the method for initialization:

'random'
'k-means++'
Note: scikit-learn’s KMeans() uses 'k-means++' by default, but it is a good idea to be explicit!
'''
'''
Instructions
The code in the workspace performs two clusterings on Codecademy learner data using K-Means. The first algorithm initializes the centroids at the x positions given on line 12 and the y positions given on line 13. The second algorithm initializes the centroids according to the K-Means++ algorithm.

Try changing the positions at which the centroids are initialized on lines 12 and 13. How does changing the initialization position affect the final clustering? And how does the first clustering compare to the K-Means++ clustering?

Make sure to scroll down to see the second graph!'''

import codecademylib3_seaborn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy
from sklearn.cluster import KMeans

std = 0.5

x = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.75,std,100)), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100))

y = np.append(np.append(np.append(np.random.normal(0.25,std,100), np.random.normal(0.25,std,100)), np.random.normal(0.75,std,100)), np.random.normal(0.75,std,100))

values = np.array(list(zip(x, y)))

centroids_x = [2.5, 2.5]
centroids_y = [1, 3]

centroids = np.array(list(zip(centroids_x, centroids_y)))

model_custom = KMeans(init=centroids, n_clusters=2)
results_custom = model_custom.fit_predict(values)

model = KMeans(init='k-means++', n_clusters=2)
results = model.fit_predict(values)

plt.scatter(x, y, c=results_custom, alpha=1)
plt.scatter(model_custom.cluster_centers_[:, 0], model_custom.cluster_centers_[:, 1], marker='v', s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=100)
plt.title('Custom Initialization')
plt.show()
plt.cla()

plt.scatter(x, y, c=results, alpha=1)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='v', s=100)
plt.title('K-Means++ Initialization')
plt.show()

print("The custom model's inertia is " + str(model_custom.inertia_))
print("The K-means++ model's inertia is " + str(model.inertia_))


















'''
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- MACHINE LEARN KNN K-MEANS K-NEAREST MACHINE LEARN   #-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- PROJECT PROJECT PROJECT PROJECT #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Off-Platform Project: Viral Tweets

In this project, we are going to use the K-Nearest Neighbor algorithm to predict whether a tweet will go viral. Before jumping into using the classifier, let's first consider the problem we're trying to solve. Which features of a tweet are most closely linked to its popularity? Maybe the number of hashtags or the number of links in the tweet strongly influences its popularity. Maybe its virality is dependent on how many followers the person has. Maybe it's something more subtle like the specific language used in the tweets.

Let's explore these options by looking at the data we have available to us. We've imported the dataset and printed the following information:
•The total number of tweets in the dataset.
•The columns, or features, of the dataset.
•The text of the first tweet in the dataset.

Some of these features are dictionaries. For example, the feature "user" is a dictionary. We might want to use some of the information found in these sub-dictionaries. Print all_tweets.loc[0]["user"] to see what the "user" feature looks like.

After printing that, try printing just the "location" found in that "user" dictionary. For example, all_tweets.loc[0]["user"]["screen_name"] would give you only the screen name associated with the first tweet.'''

import pandas as pd

all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])
print(all_tweets.loc[0]['text'].count('#'))  #hash tag count
print(len(all_tweets.loc[0]['text'].split()))       #word count
print(all_tweets.loc[0]['user']['location']) #acces to user detail

#Print the user here and the user's location here.

'''result
11099
Index(['created_at', 'id', 'id_str', 'text', 'truncated', 'entities',
       'metadata', 'source', 'in_reply_to_status_id',
       'in_reply_to_status_id_str', 'in_reply_to_user_id',
       'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo',
       'coordinates', 'place', 'contributors', 'retweeted_status',
       'is_quote_status', 'retweet_count', 'favorite_count', 'favorited',
       'retweeted', 'lang', 'possibly_sensitive', 'quoted_status_id',
       'quoted_status_id_str', 'extended_entities', 'quoted_status',
       'withheld_in_countries'],
      dtype='object')
RT @KWWLStormTrack7: We are more than a month into summer but the days are getting shorter. The sunrise is about 25 minutes later on July 3…
0
26
Waterloo, Iowa
'''

'''
Defining Viral Tweets

A K-Nearest Neighbor classifier is a supervised machine learning algorithm, and as a result, we need to have a dataset with tagged labels. For this specific example, we need a dataset where every tweet is marked as viral or not viral. Unfortunately, this isn't a feature of our dataset — we'll need to make it ourselves.

So how do we define a viral tweet? A good place to start is to look at the number of retweets the tweet has. This can be found using the feature "retweet_count". Let's say we wanted to create a column called is_viral that is a 1 if the tweet had more than 5 retweets and 0 otherwise. We could do that like this:
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > 5, 1, 0)

Instead of using 5 as the benchmark for a viral tweet, let's use the median number of retweets. You can find that by calling the median() function on all_tweets["retweet_count"]. Print the median number of retweets to understand what this threshold is.

Print the number of viral tweets and non-viral tweets. You can do this using all_tweets['is_viral'].value_counts().

After finishing this project, consider coming back and playing with this threshold number. How do you think your model would work if it was trying to find incredibly viral tweets? For example, how would it work if it were looking for tweets with 1000 or more retweets?'''

import numpy as np

all_tweets['is_viral'] =  np.where(all_tweets['retweet_count'] > 13, 1, 0)

print(np.median(all_tweets['retweet_count']))

print(all_tweets['is_viral'].value_counts())

print(type(all_tweets))


'''result

13.0
0    5562
1    5537
Name: is_viral, dtype: int64
<class 'pandas.core.frame.DataFrame'>

'''

'''
Making Features¶

Now that we've created a label for every tweet in our dataset, we can begin thinking about which features might determine whether a tweet is viral. We can create new columns in our dataset to represent these features. For example, let's say we think the length of a tweet might be a valuable feature. The following line creates a new column containing the length of the tweet.
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)

Setting axis = 1 creates a new column rather than a new row.

Create a new column called followers_count that contains the number of followers of each user. You can find this information in tweet['user']['followers_count']. Do the same for friends_count.

For the rest of this project, we will be using these three features, but we encourage you to create your own. Here are some potential ideas for more features.
•The number of hashtags in the tweet. You can find this by looking at the text of the tweet and using the .count() function with # as a parameter.
•The number of links in the tweet. Using a similar strategy to the one above, use .count() to count the number of times http appears in the tweet.
•The number of words in the tweet. Call .split() on the text of a tweet. This will give you a list of the words in the tweet. Find the length of that list.
•The average length of the words in the tweet.

'''

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis = 1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis = 1)
all_tweets['hashtags_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis = 1)
all_tweets['link_count'] = all_tweets.apply(lambda tweet: tweet['text'].count("http"), axis = 1)
all_tweets['word_count'] = all_tweets.apply(lambda tweet: len(tweet['text'].split()), axis = 1)

'''
Normalizing The Data

We've now made the columns that we want to feed into our classifier. Let's get rid of all the data that is no longer relevant. Create a variable named labels and set it equal to the 'is_viral' column of all_tweets.

If we had a dataframe named df we could get a single column named A like this:
one_column = df['A']

Create a variable named data and set it equal to all of the columns that you created in the last step. Those columns are tweet_length, followers_count, and friends_count.

When selecting multiple columns, the names of the columns should be in a list. Check out the example below to see how to select column A and B:
features = df[['A', 'B']]

Now create a new variable named scaled_data. scaled_data should be the result of the scale function with data as a parameter. Also include the parameter axis = 0. This scales the columns as opposed to the rows.

The scale function will normalize the data so all of the features will vary within the same range.

Print scaled_data[0] to get a sense of what our data looks like.'''


from sklearn.preprocessing import scale

labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count', 'friends_count', 'hashtags_count', 'word_count', 'link_count']]

scaled_data = scale(data, axis = 0)

print(data['tweet_length'])
print(type(scaled_data))
print(scaled_data[0])

''' result0        140
1         77
2        140
3        140
4        140
        ... 
11094    140
11095     75
11096    140
11097    140
11098     75
Name: tweet_length, Length: 11099, dtype: int64
<class 'numpy.ndarray'>
[ 0.6164054  -0.02878298 -0.14483305 -0.32045057  1.15105133 -0.78415588]
'''

'''
Creating the Training Set and Test Set

To evaluate the effectiveness of our classifier, we now split scaled_data and labels into a training set and test set using scikit-learn's train_test_split function. This function takes two required parameters: It takes the data, followed by the labels. Set the optional parameter test_size to be 0.2. You can also set the random_state parameter so your code will randomly split the data in the same way as our solution code splits the data. We used random_state = 1. Remember, this function returns 4 items in this order:
1.The training data
2.The testing data
3.The training labels
4.The testing labels

Store the results in variables named train_data, test_data, train_labels, and test_labels.'''

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, train_size = 0.8, test_size = 0.2, random_state =1)

print(len(train_data))

print(len(test_data))

'''result
8879
2220

'''

'''

Using the Classifier

We can finally use the K-Nearest Neighbor classifier. Let's test it using k = 5. Begin by creating a KNeighborsClassifier object named classifier with the parameter n_neighbors equal to 5.

Next, train classifier by calling the .fit() method with train_data and train_labels as parameters.

Finally, let's test the model! Call classifier's .score() method using test_data and test_labels as parameters. Print the results.
'''

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(train_data, train_labels)

print(classifier.score(test_data, test_labels))


#0.7175675675675676

'''
Choosing K

We've tested our classifier with k = 5, but maybe there's a k that will work better. Let's test many different values for k and graph the results. 

First, create an empty list called scores. Next, create a for loop that has a variable k that begins at 1 and ends at 200.

Inside the for loop, create a KNeighobrsClassifier object named classifier with the parameter n_neighbors equal to k.

Train classifier by calling the .fit() method with train_data and train_labels as parameters.

Next, let's test the model! Call classifier's .score() method using test_data and test_labels as parameters. append the result to scores.

Finally, let's plot the results. Outside of the loop, use Matplotlib's plot() function. plot() takes two parameters — the data on the x-axis and the data on the y-axis. Data on the x-axis should be the values we used for k. In this case, range(1,200). Data on the y-axis should be scores. Make sure to call the plt.show() function after calling plt.plot(). This should take a couple of seconds to run!
'''

import matplotlib.pyplot as plt

scores = []

for k in range(1,201):
    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))

plt.plot(range(1,201),scores)
plt.show()
    
'''

Explore on your own

Nice work! You can see the classifier gets better as k increases, but as k gets too high, underfitting starts to happen.

By using the features tweet_length, followers_count, and friends_count, we were able to get up to around 63% accuracy. That is better than random, but still not exceptional. Can you find some different features that perform better? Share your graphs with us on Twitter and maybe it will go viral!'''







