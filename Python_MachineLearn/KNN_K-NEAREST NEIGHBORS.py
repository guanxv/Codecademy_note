




'''
K-NEAREST NEIGHBORS
K-Nearest Neighbors Classifier
K-Nearest Neighbors (KNN) is a classification algorithm. The central idea is that data points with similar attributes tend to fall into similar categories.

Consider the image to the right. This image is complicated, but for now, let’s just focus on where the data points are being placed. Every data point — whether its color is red, green, or white — has an x value and a y value. As a result, it can be plotted on this two-dimensional graph.

Next, let’s consider the color of the data. The color represents the class that the K-Nearest Neighbor algorithm is trying to classify. In this image, data points can either have the class green or the class red. If a data point is white, this means that it doesn’t have a class yet. The purpose of the algorithm is to classify these unknown points.

Finally, consider the expanding circle around the white point. This circle is finding the k nearest neighbors to the white point. When k = 3, the circle is fairly small. Two of the three nearest neighbors are green, and one is red. So in this case, the algorithm would classify the white point as green. However, when we increase k to 5, the circle expands, and the classification changes. Three of the nearest neighbors are red and two are green, so now the white point will be classified as red.

This is the central idea behind the K-Nearest Neighbor algorithm. If you have a dataset of points where the class of each point is known, you can take a new point with an unknown class, find it’s nearest neighbors, and classify it.

Instructions
Before moving on to the next exercise, consider the image below:

If k = 1, what would the class of the question mark be?
If k = 5, what would it be?
2D Visualization
Note that rather than using colors, in this image, the class is denoted by the shape of each point.

'''
#cheat sheet ---------------------------------

'''
K-Nearest Neighbors
The K-Nearest Neighbors algorithm is a supervised machine learning algorithm for labeling an unknown data point given existing labeled data.

The nearness of points is typically determined by using distance algorithms such as the Euclidean distance formula based on parameters of the data. The algorithm will classify a point based on the labels of the K nearest neighbor points, where the value of K can be specified.

Elbow Curve Validation Technique in K-Nearest Neighbor Algorithm
Choosing an optimal k value in KNN determines the number of neighbors we look at when we assign a value to any new observation.

For a very low value of k (suppose k=1), the model overfits on the training data, which leads to a high error rate on the validation set. On the other hand, for a high value of k, the model performs poorly on both train and validation set. When k increases, validation error decreases and then starts increasing in a “U” shape. An optimal value of k can be determined from the elbow curve of the validation error.

K-Nearest Neighbors Underfitting and Overfitting
The value of k in the KNN algorithm is related to the error rate of the model. A small value of k could lead to overfitting as well as a big value of k can lead to underfitting. Overfitting imply that the model is well on the training data but has poor performance when new data is coming. Underfitting refers to a model that is not good on the training data and also cannot be generalized to predict new data.

KNN Classification Algorithm in Scikit Learn
Scikit-learn is a very popular Machine Learning library in Python which provides a KNeighborsClassifier object which performs the KNN classification. The n_neighbors parameter passed to the KNeighborsClassifier object sets the desired k value that checks the k closest neighbors for each unclassified point.

The object provides a .fit() method which takes in training data and a .predict() method which returns the classification of a set of data points.

from sklearn.neighbors import KNeighborsClassifier

KNNClassifier = KNeighborsClassifier(n_neighbors=5)
KNNClassifier.fit(X_train, y_train)
KNNClassifier.predict(X_test)
Euclidean Distance
The Euclidean Distance between two points can be computed, knowing the coordinates of those points.

On a 2-D plane, the distance between two points p and q is the square-root of the sum of the squares of the difference between their x and y components. Remember the Pythagorean Theorem: a^2 + b^2 = c^2 ?

We can write a function to compute this distance. Let’s assume that points are represented by tuples of the form (x_coord, y_coord). Also remember that computing the square-root of some value n can be done in a couple of ways: math.sqrt(n), using the math library, or n ** 0.5 (n raised to the power of 1/2).

def distance(p1, p2):
  x_diff_squared = (p1[0] - p2[0]) ** 2
  y_diff_squared = (p1[1] - p2[1]) ** 2
  return (x_diff_squared + y_diff_squared) ** 0.5

distance( (0, 0), (3, 4) )      # => 5.0

'''
'''

Distance Between Points - 2D
In the first exercise, we were able to visualize the dataset and estimate the k nearest neighbors of an unknown point. But a computer isn’t going to be able to do that!

We need to define what it means for two points to be close together or far apart. To do this, we’re going to use the Distance Formula.

For this example, the data has two dimensions:

The length of the movie
The movie’s release date
Consider Star Wars and Raiders of the Lost Ark. Star Wars is 125 minutes long and was released in 1977. Raiders of the Lost Ark is 115 minutes long and was released in 1981.

The distance between the movies is computed below:

distance formula example

K-NEAREST NEIGHBORS
Distance Between Points - 3D
Making a movie rating predictor based on just the length and release date of movies is pretty limited. There are so many more interesting pieces of data about movies that we could use! So let’s add another dimension.

Let’s say this third dimension is the movie’s budget. We now have to find the distance between these two points in three dimensions.

3D graph
What if we’re not happy with just three dimensions? Unfortunately, it becomes pretty difficult to visualize points in dimensions higher than 3. But that doesn’t mean we can’t find the distance between them.

The generalized distance formula between points A and B is as follows:

\sqrt{(A_1-B_1)^2+(A_2-B_2)^2+ \dots+(A_n-B_n)^2} 

​	 
Here, A1-B1 is the difference between the first feature of each point. An-Bn is the difference between the last feature of each point.

Using this formula, we can find the K-Nearest Neighbors of a point in N-dimensional space! We now can use as much information about our movies as we want.

We will eventually use these distances to find the nearest neighbors to an unlabeled point.

Data with Different Scales: Normalization
In the next three lessons, we’ll implement the three steps of the K-Nearest Neighbor Algorithm:

Normalize the data
Find the k nearest neighbors
Classify the new point based on those neighbors
When we added the dimension of budget, you might have realized there are some problems with the way our data currently looks.

Consider the two dimensions of release date and budget. The maximum difference between two movies’ release dates is about 125 years (The Lumière Brothers were making movies in the 1890s). However, the difference between two movies’ budget can be millions of dollars.

The problem is that the distance formula treats all dimensions equally, regardless of their scale. If two movies came out 70 years apart, that should be a pretty big deal. However, right now, that’s exactly equivalent to two movies that have a difference in budget of 70 dollars. The difference in one year is exactly equal to the difference in one dollar of budget. That’s absurd!

Another way of thinking about this is that the budget completely outweighs the importance of all other dimensions because it is on such a huge scale. The fact that two movies were 70 years apart is essentially meaningless compared to the difference in millions in the other dimension.

The solution to this problem is to normalize the data so every value is between 0 and 1. In this lesson, we’re going to be using min-max normalization.

'''
release_dates = [1897, 1998, 2000, 1948, 1962, 1950, 1975, 1960, 2017, 1937, 1968, 1996, 1944, 1891, 1995, 1948, 2011, 1965, 1891, 1978]

def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  for i in range(len(lst)):
    normalized.append((lst[i] - minimum) / (maximum - minimum))
  return normalized

print(min_max_normalize(release_dates))

'''
K-NEAREST NEIGHBORS
Finding the Nearest Neighbors
The K-Nearest Neighbor Algorithm:

Normalize the data
Find the k nearest neighbors
Classify the new point based on those neighbors
Now that our data has been normalized and we know how to find the distance between two points, we can begin classifying unknown data!

To do this, we want to find the k nearest neighbors of the unclassified point. In a few exercises, we’ll learn how to properly choose k, but for now, let’s choose a number that seems somewhat reasonable. Let’s choose 5.

In order to find the 5 nearest neighbors, we need to compare this new unclassified movie to every other movie in the dataset. This means we’re going to be using the distance formula again and again. We ultimately want to end up with a sorted list of distances and the movies associated with those distances.

It might look something like this:

[
  [0.30, 'Superman II'],
  [0.31, 'Finding Nemo'],
  ...
  ...
  [0.38, 'Blazing Saddles']
]
In this example, the unknown movie has a distance of 0.30 to Superman II.

In the next exercise, we’ll use the labels associated with these movies to classify the unlabeled point.
'''
#--------------------------------
'''
Instructions
1.
Begin by running the program. We’ve imported and normalized a movie dataset for you and printed the data for the movie Bruce Almighty. Each movie in the dataset has three features:

the normalized budget (dollars)
the normalized duration (minutes)
the normalized release year.
We’ve also imported the labels associated with every movie in the dataset. The label associated with Bruce Almighty is a 0, indicating that it is a bad movie. Remember, a bad movie had a rating less than 7.0 on IMDb.

Comment out the two print lines after you have run the program.

If you want to see some more of the data, the following line of code will print 20 movies along with their data.

print(list(movie_dataset.items())[:20])
2.
Create a function called classify that has three parameters: the data point you want to classify named unknown, the dataset you are using to classify it named dataset, and k, the number of neighbors you are interested in.

For now put pass inside your function.

def classify(unknown, dataset, k):
'''

from movies import movie_dataset, movie_labels

print(movie_dataset['Bruce Almighty'])
print(movie_labels['Bruce Almighty'])

#print(list(movie_dataset.items())[:20])
#print(list(movie_labels.items())[:20])

#print(len(movie_dataset))

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, k):
  distances = []
  
  for title in dataset:
    distance_to_point = distance(dataset[title], unknown)
    distances.append([distance_to_point, title])
  
  distances.sort()
  
  neighbors = distances[0:k]
  
  return neighbors

print(classify([.4, .2, .9], movie_dataset, 5) )
'''
K-NEAREST NEIGHBORS
Count Neighbors
The K-Nearest Neighbor Algorithm:

Normalize the data
Find the k nearest neighbors
Classify the new point based on those neighbors
We’ve now found the k nearest neighbors, and have stored them in a list that looks like this:

[
  [0.083, 'Lady Vengeance'],
  [0.236, 'Steamboy'],
  ...
  ...
  [0.331, 'Godzilla 2000']
]
Our goal now is to count the number of good movies and bad movies in the list of neighbors. If more of the neighbors were good, then the algorithm will classify the unknown movie as good. Otherwise, it will classify it as bad.

In order to find the class of each of the labels, we’ll need to look at our movie_labels dataset. For example, movie_labels['Akira'] would give us 1 because Akira is classified as a good movie.

You may be wondering what happens if there’s a tie. What if k = 8 and four neighbors were good and four neighbors were bad? There are different strategies, but one way to break the tie would be to choose the class of the closest point.
'''
from movies import movie_dataset, movie_labels

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k ):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]

  print(neighbors)

  num_good = 0
  num_bad = 0
  title = 0

  for movie in neighbors:
    title = movie[1]
    if movie_labels[title] == 1:
      num_good += 1
    else:
      num_bad += 1


  if num_good > num_bad:
    return 1
  else:
    return 0

print(classify([.45, .2, .5], movie_dataset, movie_labels, 5))

print(classify([.4, .2, .9], movie_dataset, movie_labels, 5))

#------------------------------
'''
K-NEAREST NEIGHBORS
Classify Your Favorite Movie
Nice work! Your classifier is now able to predict whether a movie will be good or bad. So far, we’ve only tested this on a completely random point [.4, .2, .9]. In this exercise we’re going to pick a real movie, normalize it, and run it through our classifier to see what it predicts!

In the instructions below, we are going to be testing our classifier using the 2017 movie Call Me By Your Name. Feel free to pick your favorite movie instead!

Instructions
1.
To begin, we want to make sure the movie that we want to classify isn’t already in our database. This is important because we don’t want one of the nearest neighbors to be itself!

You can do this by using the in keyword.

Begin by printing if the title of your movie is in movie_dataset. This should print False.

For Call Me By Your Name, we would do the following:

print("Call Me By Your Name" in movie_dataset)
2.
Once you confirm your movie is not in your database, we need to make a datapoint for your movie. Create a variable named my_movie and set it equal to a list of three numbers. They should be:

The movie’s budget (dollars)
The movie’s runtime (minutes)
The year the movie was released
Make sure to put the information in that order.

If you want to use Call Me By Your Name, the budget was 350,000 dollars, the runtime was 132 minutes, and the movie was released in 2017.

For Call Me By Your Name, our code looks like this:

my_movie = [3500000, 132, 2017]
3.
Next, we want to normalize this datapoint. We’ve included the function normalize_point which takes a datapoint as a parameter and returns the point normalized. Create a variable called normalized_my_movie and set it equal to the normalized value of my_movie. Print the result!

The call to the function should look like this:

normalized_my_movie = normalize_point(my_movie)
4.
Finally, call classify with the following parameters:

normalized_my_movie
movie_dataset
movie_labels
5
Print the result? Did your classifier think your movie was good or bad?

classify(normalized_my_movie, movie_dataset, movie_labels, 5)

'''
from movies import movie_dataset, movie_labels, normalize_point

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

name = "Call Me By Your Name"

print(name in movie_dataset)

my_movie = [350000, 132, 2017]

normalized_my_movie = normalize_point(my_movie)

print(classify(normalized_my_movie,movie_dataset,movie_labels, 5 ))


'''
K-NEAREST NEIGHBORS
Training and Validation Sets
You’ve now built your first K Nearest Neighbors algorithm capable of classification. You can feed your program a never-before-seen movie and it can predict whether its IMDb rating was above or below 7.0. However, we’re not done yet. We now need to report how effective our algorithm is. After all, it’s possible our predictions are totally wrong!

As with most machine learning algorithms, we have split our data into a training set and validation set.

Once these sets are created, we will want to use every point in the validation set as input to the K Nearest Neighbor algorithm. We will take a movie from the validation set, compare it to all the movies in the training set, find the K Nearest Neighbors, and make a prediction. After making that prediction, we can then peek at the real answer (found in the validation labels) to see if our classifier got the answer correct.

If we do this for every movie in the validation set, we can count the number of times the classifier got the answer right and the number of times it got it wrong. Using those two numbers, we can compute the validation accuracy.

Validation accuracy will change depending on what K we use. In the next exercise, we’ll use the validation accuracy to pick the best possible K for our classifier.

Instructions
1.
We’ve imported training_set, training_labels, validation_set, and validation_labels. Let’s take a look at one of the movies in validation_set.

The movie "Bee Movie" is in validation_set. Print out the data associated with Bee Movie. Print Bee Movie ‘s label as well (which can be found in validation_labels).

Is Bee Movie a good or bad movie?

To print the data about Bee Movie, do the following:

print(validation_set["Bee Movie"])
Do the same for Bee Movie‘s label.

2.
Let’s have our classifier predict whether Bee Movie is good or bad using k = 5. Call the classify function using the following parameters:

Bee Movie‘s data
training_set
training_labels
5
Store the results in a variable named guess and print guess.

Bee Movie‘s data can be found using validation_set["Bee Movie"]. Use that as the first parameter of the classify function.

3.
Let’s check to see if our classification got it right. If guess is equal to Bee Movie‘s real class (found in validation_labels), print "Correct!". Otherwise, print "Wrong!".

To check if the guess was right, use:

if guess == validation_labels["Bee Movie"]:

'''

from movies import training_set, training_labels, validation_set, validation_labels

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

name = 'Bee Movie'
print(validation_set[name])
print(validation_labels[name])

guess = classify(validation_set[name], training_set, training_labels, 5)

print (guess)

if guess == validation_labels[name]:
  print('Correct!')
else:
  print('Wrong!') 

'''
K-NEAREST NEIGHBORS
Choosing K
In the previous exercise, we found that our classifier got one point in the training set correct. Now we can test every point to calculate the validation accuracy.

The validation accuracy changes as k changes. The first situation that will be useful to consider is when k is very small. Let’s say k = 1. We would expect the validation accuracy to be fairly low due to overfitting. Overfitting is a concept that will appear almost any time you are writing a machine learning algorithm. Overfitting occurs when you rely too heavily on your training data; you assume that data in the real world will always behave exactly like your training data. In the case of K-Nearest Neighbors, overfitting happens when you don’t consider enough neighbors. A single outlier could drastically determine the label of an unknown point. Consider the image below.

colored dots with a single outlier

The dark blue point in the top left corner of the graph looks like a fairly significant outlier. When k = 1, all points in that general area will be classified as dark blue when it should probably be classified as green. Our classifier has relied too heavily on the small quirks in the training data.

On the other hand, if k is very large, our classifier will suffer from underfitting. Underfitting occurs when your classifier doesn’t pay enough attention to the small quirks in the training set. Imagine you have 100 points in your training set and you set k = 100. Every single unknown point will be classified in the same exact way. The distances between the points don’t matter at all! This is an extreme example, however, it demonstrates how the classifier can lose understanding of the training data if k is too big.'''
'''
Instructions
1.
Begin by creating a function called find_validation_accuracy that takes five parameters. The parameters should be training_set, training_labels, validation_set, validation_labels, and k.

def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
2.
Create a variable called num_correct and have it begin at 0.0. Loop through the movies of validation_set, and call classify using each movie’s data, the training_set, the training_labels, and k. Store the result in a variable called guess. For now, return guess outside of your loop.

Remember, the movie’s data can be found by using validation_set[title].

Your for loop should look like

for title in validation_set:
Inside the for loop, you can call classify using

classify(validation_set[title], training_set, training_labels, k)
3.
Inside the for loop, compare guess to the corresponding label in validation_labels. If they were equal, add 1 to num_correct. For now, outside of the for loop, return num_correct

The label that you want to compare guess to is validation_labels[title].

4.
Outside the for loop return the validation error. This should be num_correct divided by the total number of points in the validation set.

len(validation_set) will give you the number of points in the validation set.

5.
Call find_validation_accuracy with k = 3. Print the results The code should take a couple of seconds to run.

print(find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, 3))'''

from movies import training_set, training_labels, validation_set, validation_labels

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title] == 0:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0

def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
  num_correct = 0.0
  for movie in validation_set:
    guess = classify(validation_set[movie], training_set, training_labels, k)
    if guess == validation_labels[movie]:
      num_correct += 1
  return num_correct / len(validation_set)

print(find_validation_accuracy(training_set, training_labels, validation_set, validation_labels,3))

'''
K-NEAREST NEIGHBORS
Graph of K
The graph to the right shows the validation accuracy of our movie classifier as k increases. When k is small, overfitting occurs and the accuracy is relatively low. On the other hand, when k gets too large, underfitting occurs and accuracy starts to drop.

--------------------------------------------------------------

K-NEAREST NEIGHBORS
Using sklearn
You’ve now written your own K-Nearest Neighbor classifier from scratch! However, rather than writing your own classifier every time, you can use Python’s sklearn library. sklearn is a Python library specifically used for Machine Learning. It has an amazing number of features, but for now, we’re only going to investigate its K-Nearest Neighbor classifier.

There are a couple of steps we’ll need to go through in order to use the library. First, you need to create a KNeighborsClassifier object. This object takes one parameter - k. For example, the code below will create a classifier where k = 3

classifier = KNeighborsClassifier(n_neighbors = 3)
Next, we’ll need to train our classifier. The .fit() method takes two parameters. The first is a list of points, and the second is the labels associated with those points. So for our movie example, we might have something like this

training_points = [
  [0.5, 0.2, 0.1],
  [0.9, 0.7, 0.3],
  [0.4, 0.5, 0.7]
]

training_labels = [0, 1, 1]
classifier.fit(training_points, training_labels)
Finally, after training the model, we can classify new points. The .predict() method takes a list of points that you want to classify. It returns a list of its guesses for those points.

unknown_points = [
  [0.2, 0.1, 0.7],
  [0.4, 0.7, 0.6],
  [0.5, 0.8, 0.1]
]

guesses = classifier.predict(unknown_points)'''

from movies import movie_dataset, labels
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors  = 5)

training_points = movie_dataset
training_labels  = labels

classifier.fit(training_points, training_labels)

unknown_points = [
[.45, .2, .5],
[.25, .8, .9],
[.1, .1, .9]]

print(classifier.predict(unknown_points))

#-----------------------------------------------------
'''
K-NEAREST NEIGHBORS
Review
Congratulations! You just implemented your very own classifier from scratch and used Python’s sklearn library. In this lesson, you learned some techniques very specific to the K-Nearest Neighbor algorithm, but some general machine learning techniques as well. Some of the major takeaways from this lesson include:

Data with n features can be conceptualized as points lying in n-dimensional space.
Data points can be compared by using the distance formula. Data points that are similar will have a smaller distance between them.
A point with an unknown class can be classified by finding the k nearest neighbors
To verify the effectiveness of a classifier, data with known classes can be split into a training set and a validation set. Validation error can then be calculated.
Classifiers have parameters that can be tuned to increase their effectiveness. In the case of K-Nearest Neighbors, k can be changed.
A classifier can be trained improperly and suffer from overfitting or underfitting. In the case of K-Nearest Neighbors, a low k often leads to overfitting and a large k often leads to underfitting.
Python’s sklearn library can be used for many classification and machine learning algorithms.
To the right is an interactive visualization of K-Nearest Neighbors. If you move your mouse over the canvas, the location of your mouse will be classified as either green or blue. The nearest neighbors to your mouse are highlighted in yellow. Use the slider to change k to see how the boundaries of the classification change.

If you find any interesting patterns, share it with us on Twitter!
'''
'''
K-NEAREST NEIGHBOR REGRESSOR
Regression
The K-Nearest Neighbors algorithm is a powerful supervised machine learning algorithm typically used for classification. However, it can also perform regression.

In this lesson, we will use the movie dataset that was used in the K-Nearest Neighbors classifier lesson. However, instead of classifying a new movie as either good or bad, we are now going to predict its IMDb rating as a real number.

This process is almost identical to classification, except for the final step. Once again, we are going to find the k nearest neighbors of the new movie by using the distance formula. However, instead of counting the number of good and bad neighbors, the regressor averages their IMDb ratings.

For example, if the three nearest neighbors to an unrated movie have ratings of 5.0, 9.2, and 6.8, then we could predict that this new movie will have a rating of 7.0.
'''



from movies import movie_dataset, movie_ratings

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(unknown, dataset, movie_ratings, k):
  distances = []
  sum_rating = 0
  avg_rating = 0
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]


  for neighbor in neighbors:
    title = neighbor[1]
    rating = movie_ratings[title]
    sum_rating += rating
  avg_rating = sum_rating / (len(neighbors))
  return avg_rating



print(movie_dataset['Life of Pi'])
print(movie_ratings['Life of Pi'])
print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))

'''
K-NEAREST NEIGHBOR REGRESSOR
Weighted Regression
We’re off to a good start, but we can be even more clever in the way that we compute the average. We can compute a weighted average based on how close each neighbor is.

Let’s say we’re trying to predict the rating of movie X and we’ve found its three nearest neighbors. Consider the following table:

Movie	Rating	Distance to movie X
A	5.0	3.2
B	6.8	11.5
C	9.0	1.1

If we find the mean, the predicted rating for X would be 6.93. However, movie X is most similar to movie C, so movie C’s rating should be more important when computing the average. Using a weighted average, we can find movie X’s rating:

(5.0/3.2 + 6.8 / 11.5 + 9.0 / 1.1) / (1/3.2 + 1 / 11.5 + 1 / 1.1) = 7.9

The numerator is the sum of every rating divided by their respective distances. The denominator is the sum of one over every distance. Even though the ratings are the same as before, the weighted average has now gone up to 7.9.

'''
from movies import movie_dataset, movie_ratings

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def predict(unknown, dataset, movie_ratings, k):
  distances = []
  numerator = 0
  denominator = 0
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  for neighbor in neighbors:
        numerator += (movie_ratings[neighbor[1]] )/ neighbor[0] 
    denominator += 1/neighbor[0]
  return numerator / denominator

print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, k = 5 ))

'''
K-NEAREST NEIGHBOR REGRESSOR
Scikit-learn
Now that you’ve written your own K-Nearest Neighbor regression model, let’s take a look at scikit-learn’s implementation. The KNeighborsRegressor class is very similar to KNeighborsClassifier.

We first need to create the regressor. We can use the parameter n_neighbors to define our value for k.

We can also choose whether or not to use a weighted average using the parameter weights. If weights equals "uniform", all neighbors will be considered equally in the average. If weights equals "distance", then a weighted average is used.
'''
classifier = KNeighborsRegressor(n_neighbors = 3, weights = "distance")'''
Next, we need to fit the model to our training data using the .fit() method. .fit() takes two parameters. The first is a list of points, and the second is a list of values associated with those points.
'''
training_points = [
  [0.5, 0.2, 0.1],
  [0.9, 0.7, 0.3],
  [0.4, 0.5, 0.7]
]

training_labels = [5.0, 6.8, 9.0]
classifier.fit(training_points, training_labels)'''
Finally, we can make predictions on new data points using the .predict() method. .predict() takes a list of points and returns a list of predictions for those points.
'''
unknown_points = [
  [0.2, 0.1, 0.7],
  [0.4, 0.7, 0.6],
  [0.5, 0.8, 0.1]
]

guesses = classifier.predict(unknown_points)'''
'''

#--- code ---
from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
regressor.fit(movie_dataset, movie_ratings)

#print(regressor.predict([0.016, 0.300, 1.022]))
#print(regressor.predict([0.0004092981, 0.283, 1.0112]))
#print(regressor.predict([0.00687649, 0.235, 1.0112]))

print(regressor.predict([
  
  [0.00687649, 0.235, 1.0112],
  [0.0004092981, 0.283, 1.0112],
  [0.00687649, 0.235, 1.0112]
]))



