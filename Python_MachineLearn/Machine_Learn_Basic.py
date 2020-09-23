



#------------------------------
'''c
DISTANCE FORMULA
Representing Points
In this lesson, you will learn three different ways to define the distance between two points:

Euclidean Distance
Manhattan Distance
Hamming Distance
Before diving into the distance formulas, it is first important to consider how to represent points in your code.

In this exercise, we will use a list, where each item in the list represents a dimension of the point. For example, the point (5, 8) could be represented in Python like this:

pt1 = [5, 8]
Points aren’t limited to just two dimensions. For example, a five-dimensional point could be represented as [4, 8, 15, 16, 23].

Ultimately, we want to find the distance between two points. We’ll be writing functions that look like this:

distance([1, 2, 3], [5, 8, 9])
Note that we can only find the difference between two points if they have the same number of dimensions!
'''

'''DISTANCE FORMULA
Euclidean Distance
Euclidean Distance is the most commonly used distance formula. To find the Euclidean distance between two points, we first calculate the squared distance between each dimension. If we add up all of these squared differences and take the square root, we’ve computed the Euclidean distance.

Let’s take a look at the equation that represents what we just learned:

\sqrt{(a_1-b_1)^2+(a_2-b_2)^2+\ldots+(a_n - b_n)^2} 
(a 
1
​	 −b 
1
​	 ) 
2
 +(a 
2
​	 −b 
2
​	 ) 
2
 +…+(a 
n
​	 −b 
n
​	 ) 
2
 
​	 
The image below shows a visual of Euclidean distance being calculated:

The Euclidean distance between two points.

d = \sqrt{(a_1-b_1)^2+(a_2-b_2)^2}d= 
(a 
1
​	 −b 
1
​	 ) 
2
 +(a 
2
​	 −b 
2
​	 ) 
2
 
​	'''

#------------------
def euclidean_distance(pt1, pt2):
  distance = 0
  for n in range(len(pt1)):
    distance += (pt1[n] - pt2[n])**2
  distance = distance ** 0.5
  return distance

print(euclidean_distance(([1,2]),([4,0])))

print(euclidean_distance(([5,4,3]),([1,7,9])))

print(euclidean_distance(([1,2]),([4,0])))

'''

DISTANCE FORMULA
Manhattan Distance
Manhattan Distance is extremely similar to Euclidean distance. Rather than summing the squared difference between each dimension, we instead sum the absolute value of the difference between each dimension. It’s called Manhattan distance because it’s similar to how you might navigate when walking city blocks. If you’ve ever wondered “how many blocks will it take me to get from point A to point B”, you’ve computed the Manhattan distance.

The equation is shown below:

\mid a_1 - b_1 \mid + \mid a_2 - b_2 \mid + \ldots + \mid a_n - b_n \mid∣a 
1
​	 −b 
1
​	 ∣+∣a 
2
​	 −b 
2
​	 ∣+…+∣a 
n
​	 −b 
n
​	 ∣
Note that Manhattan distance will always be greater than or equal to Euclidean distance. Take a look at the image below visualizing Manhattan Distance:

The Manhattan distance between two points.

d = \mid a_1 - b_1 \mid + \mid a_2 - b_2 \midd=∣a 
1
​	 −b 
1
​	 ∣+∣a 
2
​	 −b 
2
​	 ∣

'''
'''
DISTANCE FORMULA
Hamming Distance
Hamming Distance is another slightly different variation on the distance formula. Instead of finding the difference of each dimension, Hamming distance only cares about whether the dimensions are exactly equal. When finding the Hamming distance between two points, add one for every dimension that has different values.

Hamming distance is used in spell checking algorithms. For example, the Hamming distance between the word “there” and the typo “thete” is one. Each letter is a dimension, and each dimension has the same value except for one.
'''
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

def hamming_distance(pt1,pt2):
  distance = 0
  for i in range(len(pt1)):
    if pt1[i] != pt2[i]:
      distance += 1
  return distance
  
 
'''
DISTANCE FORMULA
SciPy Distances
Now that you’ve written these three distance formulas yourself, let’s look at how to use them using Python’s SciPy library:

Euclidean Distance .euclidean()
Manhattan Distance .cityblock()
Hamming Distance .hamming()
There are a few noteworthy details to talk about:

First, the scipy implementation of Manhattan distance is called cityblock(). Remember, computing Manhattan distance is like asking how many blocks away you are from a point.

Second, the scipy implementation of Hamming distance will always return a number between 0 an 1. Rather than summing the number of differences in dimensions, this implementation sums those differences and then divides by the total number of dimensions. For example, in your implementation, the Hamming distance between [1, 2, 3] and [7, 2, -10] would be 2. In scipy‘s version, it would be 2/3.

'''
from scipy.spatial import distance

def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

def hamming_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    if pt1[i] != pt2[i]:
      distance += 1
  return distance

print(euclidean_distance([1, 2], [4, 0]))
print(manhattan_distance([1, 2], [4, 0]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))

print(distance.euclidean([1, 2], [4, 0]))
print(distance.cityblock([1, 2], [4, 0]))
print(distance.hamming([5, 4, 9], [1, 7, 9]))


codecademy.com/courses/machine-learning/articles/normalization
'''
Normalization
This article describes why normalization is necessary. It also demonstrates the pros and cons of min-max normalization and z-score normalization.

Why Normalize?
Many machine learning algorithms attempt to find trends in the data by comparing features of data points. However, there is an issue when the features are on drastically different scales.

For example, consider a dataset of houses. Two potential features might be the number of rooms in the house, and the total age of the house in years. A machine learning algorithm could try to predict which house would be best for you. However, when the algorithm compares data points, the feature with the larger scale will completely dominate the other. Take a look at the image below:

Data points on the y-axis range from 0 to 20. Data points on the x-axis range from 0 to 100
When the data looks squished like that, we know we have a problem. The machine learning algorithm should realize that there is a huge difference between a house with 2 rooms and a house with 20 rooms. But right now, because two houses can be 100 years apart, the difference in the number of rooms contributes less to the overall difference.

As a more extreme example, imagine what the graph would look like if the x-axis was the cost of the house. The data would look even more squished; the difference in the number of rooms would be even less relevant because the cost of two houses could have a difference of thousands of dollars.

The goal of normalization is to make every datapoint have the same scale so each feature is equally important. The image below shows the same house data normalized using min-max normalization.

Data points on the y-axis range from 0 to 1. Data points on the x-axis range from 0 to 1
Min-Max Normalization
Min-max normalization is one of the most common ways to normalize data. For every feature, the minimum value of that feature gets transformed into a 0, the maximum value gets transformed into a 1, and every other value gets transformed into a decimal between 0 and 1.

For example, if the minimum value of a feature was 20, and the maximum value was 40, then 30 would be transformed to about 0.5 since it is halfway between 20 and 40. The formula is as follows:

\frac{value - min}{max - min} 
max−min
value−min
​	 
Min-max normalization has one fairly significant downside: it does not handle outliers very well. For example, if you have 99 values between 0 and 40, and one value is 100, then the 99 values will all be transformed to a value between 0 and 0.4. That data is just as squished as before! Take a look at the image below to see an example of this.

Almost all normalized data points have an x value between 0 and 0.4
Normalizing fixed the squishing problem on the y-axis, but the x-axis is still problematic. Now if we were to compare these points, the y-axis would dominate; the y-axis can differ by 1, but the x-axis can only differ by 0.4.

Z-Score Normalization
Z-score normalization is a strategy of normalizing data that avoids this outlier issue. The formula for Z-score normalization is below:

\frac{value - \mu}{\sigma} 
σ
value−μ
​	 
Here, μ is the mean value of the feature and σ is the standard deviation of the feature. If a value is exactly equal to the mean of all the values of the feature, it will be normalized to 0. If it is below the mean, it will be a negative number, and if it is above the mean it will be a positive number. The size of those negative and positive numbers is determined by the standard deviation of the original feature. If the unnormalized data had a large standard deviation, the normalized values will be closer to 0.

Take a look at the graph below. This is the same data as before, but this time we’re using z-score normalization.

All points have a similar range in both the x and y dimensions
While the data still looks squished, notice that the points are now on roughly the same scale for both features — almost all points are between -2 and 2 on both the x-axis and y-axis. The only potential downside is that the features aren’t on the exact same scale.

With min-max normalization, we were guaranteed to reshape both of our features to be between 0 and 1. Using z-score normalization, the x-axis now has a range from about -1.5 to 1.5 while the y-axis has a range from about -2 to 2. This is certainly better than before; the x-axis, which previously had a range of 0 to 40, is no longer dominating the y-axis.

Review
Normalizing your data is an essential part of machine learning. You might have an amazing dataset with many great features, but if you forget to normalize, one of those features might completely dominate the others. It’s like you’re throwing away almost all of your information! Normalizing solves this problem. In this article, you learned the following techniques to normalize:

Min-max normalization: Guarantees all features will have the exact same scale but does not handle outliers well.
Z-score normalization: Handles outliers, but does not produce normalized data with the exact same scale.

'''
'''
Training Set vs Validation Set vs Test Set
This article teaches the importance of splitting a data set into training, validation and test sets.

Testing Our Model
Supervised machine learning algorithms are amazing tools capable of making predictions and classifications. However, it is important to ask yourself how accurate those predictions are. After all, it’s possible that every prediction your classifier makes is actually wrong! Luckily, we can leverage the fact that supervised machine learning algorithms, by definition, have a dataset of pre-labeled datapoints. In order to test the effectiveness of your algorithm, we’ll split this data into:

training set
validation set
test set
Training Set vs Validation Set
The training set is the data that the algorithm will learn from. Learning looks different depending on which algorithm you are using. For example, when using Linear Regression, the points in the training set are used to draw the line of best fit. In K-Nearest Neighbors, the points in the training set are the points that could be the neighbors.

After training using the training set, the points in the validation set are used to compute the accuracy or error of the classifier. The key insight here is that we know the true labels of every point in the validation set, but we’re temporarily going to pretend like we don’t. We can use every point in the validation set as input to our classifier. We’ll then receive a classification for that point. We can now peek at the true label of the validation point and see whether we got it right or not. If we do this for every point in the validation set, we can compute the validation error!

Validation error might not be the only metric we’re interested in. A better way of judging the effectiveness of a machine learning algorithm is to compute its precision, recall, and F1 score.

How to Split
Figuring out how much of your data should be split into your validation set is a tricky question. If your training set is too small, then your algorithm might not have enough data to effectively learn. On the other hand, if your validation set is too small, then your accuracy, precision, recall, and F1 score could have a large variance. You might happen to get a really lucky or a really unlucky split! In general, putting 80% of your data in the training set, and 20% of your data in the validation set is a good place to start.

N-Fold Cross-Validation
Sometimes your dataset is so small, that splitting it 80/20 will still result in a large amount of variance. One solution to this is to perform N-Fold Cross-Validation. The central idea here is that we’re going to do this entire process N times and average the accuracy. For example, in 10-fold cross-validation, we’ll make the validation set the first 10% of the data and calculate accuracy, precision, recall and F1 score. We’ll then make the validation set the second 10% of the data and calculate these statistics again. We can do this process 10 times, and every time the validation set will be a different chunk of the data. If we then average all of the accuracies, we will have a better sense of how our model does on average.

Cross Validation


Changing The Model / Test Set
Understanding the accuracy of your model is invaluable because you can begin to tune the parameters of your model to increase its performance. For example, in the K-Nearest Neighbors algorithm, you can watch what happens to accuracy as you increase or decrease K. (You can try out all of this in our K-Nearest Neighbors lesson!)

Once you’re happy with your model’s performance, it is time to introduce the test set. This is part of your data that you partitioned away at the very start of your experiment. It’s meant to be a substitute for the data in the real world that you’re actually interested in classifying. It functions very similarly to the validation set, except you never touched this data while building or tuning your model. By finding the accuracy, precision, recall, and F1 score on the test set, you get a good understanding of how well your algorithm will do in the real world.

'''
'''
ACCURACY, RECALL, PRECISION, AND F1 SCORE
Accuracy
After creating a machine learning algorithm capable of making classifications, the next step in the process is to calculate its predictive power. In order to calculate these statistics, we’ll need to split our data into a training set and validation set.

Let’s say you’re using a machine learning algorithm to try to predict whether or not you will get above a B on a test. The features of your data could be something like:

The number of hours you studied this week.
The number of hours you watched Netflix this week.
The time you went to bed the night before the test.
Your average in the class before taking the test.
The simplest way of reporting the effectiveness of an algorithm is by calculating its accuracy. Accuracy is calculated by finding the total number of correctly classified points and dividing by the total number of points.

In other words, accuracy can be defined as:

(True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)

Let’s define those terms in the context of our grade example :

True Positive: The algorithm predicted you would get above a B, and you did.
True Negative: The algorithm predicted you would get below a B, and you did.
False Positive: The algorithm predicted you would get above a B, and you didn’t.
False Negative: The algorithm predicted you would get below a B, and you didn’t.
Let’s calculate the accuracy of a classification algorithm!

ACCURACY, RECALL, PRECISION, AND F1 SCORE
Recall
Accuracy can be an extremely misleading statistic depending on your data. Consider the example of an algorithm that is trying to predict whether or not there will be over 3 feet of snow on the ground tomorrow. We can write a pretty accurate classifier right now: always predict False. This classifier will be incredibly accurate — there are hardly ever many days with that much snow. But this classifier never finds the information we’re actually interested in.

In this situation, the statistic that would be helpful is recall. Recall measures the percentage of relevant items that your classifier found. In this example, recall is the number of snow days the algorithm correctly predicted divided by the total number of snow days. Another way of saying this is:

True Positives / (True Positives + False Negatives)
Our algorithm that always predicts False might have a very high accuracy, but it never will find any True Positives, so its recall is 0. This makes sense; recall should be very low for such an absurd classifier.

Precision
Unfortunately, recall isn’t a perfect statistic either. For example, we could create a snow day classifier that always returns True. This would have low accuracy, but its recall would be 1 because it would be able to accurately find every snow day. But this classifier is just as nonsensical as the one before! The statistic that will help demonstrate that this algorithm is flawed is precision.

In the snow day example, precision is the number of snow days the algorithm correctly predicted divided by the number of times it predicted there would be a snow day. The formula for precision is below:

True Positives / (True Positives + False Positives)
The algorithm that predicts every day is a snow day has recall of 1, but it will have very low precision. It correctly predicts every snow day, but there are tons of false positives as well.

Precision and recall are statistics that are on opposite ends of a scale. If one goes down, the other will go up.

F1 Score
It is useful to consider the precision and recall of an algorithm, however, we still don’t have one number that can sufficiently describe how effective our algorithm is. This is the job of the F1 score — F1 score is the harmonic mean of precision and recall. The harmonic mean of a group of numbers is a way to average them together. The formula for F1 score is below:

F1 = 2 \* (precision \* recall) / (precision + recall)
The F1 score combines both precision and recall into a single statistic. We use the harmonic mean rather than the traditional arithmetic mean because we want the F1 score to have a low value when either precision or recall is 0.

For example, consider a classifier where recall = 1 and precision = 0.01. We know that there is most likely a problem with this classifier since the precision is so low, and so we want the F1 score to reflect that.

If we took the arithmetic mean, we’d get:

(1 + 0.01) / 2 = 0.505
That looks way too high! But if we calculate the harmonic mean, we get:

2 * (1 * 0.01) / (1 + 0.01) = 0.019
That’s much better! The F1 score is now accurately describing the effectiveness of this classifier.

Review
You’ve now learned many different ways to analyze the predictive power of your algorithm. Some of the key insights for this course include:

Classifying a single point can result in a true positive (truth = 1, guess = 1), a true negative (truth = 0, guess = 0), a false positive (truth = 0, guess = 1), or a false negative (truth = 1, guess = 0).
Accuracy measures how many classifications your algorithm got correct out of every classification it made.
Recall measures the percentage of the relevant items your classifier was able to successfully find.
Precision measures the percentage of items your classifier found that were actually relevant.
Precision and recall are tied to each other. As one goes up, the other will go down.
F1 score is a combination of precision and recall.
F1 score will be low if either precision or recall is low.
The decision to use precision, recall, or F1 score ultimately comes down to the context of your classification. Maybe you don’t care if your classifier has a lot of false positives. If that’s the case, precision doesn’t matter as much.

As long as you have an understanding of what question you’re trying to answer, you should be able to determine which statistic is most relevant to you.

The Python library scikit-learn has some functions that will calculate these statistics for you!


'''


#---------------- code 1 -----------------------

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
  #True Positives
  if labels[i] == 1 and guesses[i] == 1:
    true_positives += 1
  #True Negatives
  if labels[i] == 0 and guesses[i] == 0:
    true_negatives += 1
  #False Positives
  if labels[i] == 0 and guesses[i] == 1:
    false_positives += 1
  #False Negatives
  if labels[i] == 1 and guesses[i] == 0:
    false_negatives += 1
    
accuracy = (true_positives + true_negatives) / len(guesses)
print(accuracy)

recall = true_positives / (true_positives + false_negatives)
print(recall)

precision = true_positives / (true_positives + false_positives)
print(precision)

f_1 = 2*precision*recall / (precision + recall)
print(f_1)

'''
Python’s scikit-learn library has functions that will find accuracy, recall, precision, and F1 score for you. They all take two parameters — a list of the true labels and a list of the predicted classifications.

Call accuracy_score() using the correct parameters and print the results.

Call the three other functions and print the results. The name of those functions are:

recall_score()
precision_score()
f1_score()

'''

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

print(accuracy_score(labels, guesses))
print(recall_score(labels, guesses))
print(precision_score(labels, guesses))
print(f1_score(labels, guesses))



'''
The Dangers of Overfitting
Learn about how to recognize when your model is fitting too closely to the training data.

Often in Machine Learning, we feed a huge amount of data to an algorithm that then learns how to classify that input based on rules it creates. The data we feed into this algorithm, the training data, is hugely important. The rules created by the program will be determined by looking at every example in the training data.

Overfitting occurs when we have fit our model’s parameters too closely to the training data:

Image of overfitting

When we overfit, we are assuming that everything we see in the training data is exactly how it will appear in the real world. Instead, we want to be modeling trends that show us the general behavior of a variable:

Image of good fit

That said, when we find trends in training data, all we are doing is replicating trends that already exist. Our model will learn to replicate data from the real world. If that data is part of a system that results in prejudices or injustices, then your machine learning algorithm will produce harmful results as well. Some people say that Machine Learning can be a GIGO process — Garbage In, Garbage Out.

We can imagine an example where an ad agency is creating an algorithm to display the right job recommendations to the right people. If they use a training set of the kinds of people who have high paying jobs to determine which people to show ads for high paying jobs to, the model will probably learn to make decisions that leave out historically underrepresented groups of people.

This problem is fundamentally a problem with overfitting to our training set. If we overfit to training sets with underrepresentation, we only create more underrepresentation. How do we tackle this problem?

Inspect Training Data First
Find the important aggregate statistics for the variables you’re measuring. Find the mean and median of different variables. Use groupby to inspect the aggregate statistics for different groups of data, and see how they differ. These are the trends that your machine learning model will replicate.

Visualize your training data and look for outstanding patterns.

Compare the aggregate statistics from your specific training set to aggregate statistics from other sources. Does your training set seem to follow the trends that are universally present?

Collect Data Thoughtfully
If you have the power to control the way your data is collected, i.e. if you’re the one collecting the data, make sure that you are sampling from all groups.

Imagine for a massively multiplayer app, rewards and hotspots are set by an algorithm that trains on frequencies of user actions. If the people using the app overwhelmingly are in one area, the app will continuously get better and better for people in that area.

Some neighborhoods/areas might be undersampled, or have significantly less datapoints, so the algorithm will fit to the oversampled population. Imagine this clustering forming:

bad clustering

The small cluster in the bottom left would probably be a cluster of its own, if it had a comparable amount of samples to the other two clusters. To solve this, we can specifically oversample areas that are undersampled, and add more datapoints there. Conversely, we can undersample groups that are over-represented in our training set.

Try to Augment the Training Data
In our Bayes’ Theorem lesson we discussed that when we have a small total number of an event, this will affect how reliably we can guess if the event will occur. Many systems built to detect fraud suffer from this problem. Suppose we were creating a machine learning model to detect fraudulent credit card activity. On the aggregate, there are very few fraudulent transactions, so the model can reach a very high accuracy by simply predicting that every transaction is legitimate. This approach would not solve our problem very well.

One technique is to identify a fraudulent transaction and make many copies of it in the training set, with small variations in the feature data. We can imagine that if our training set has only 2 examples of fraud, the algorithm will overfit to only identify a transaction as fraudulent if it has the exact characteristics of those couple of examples. When we augment the training data with more fraudulent examples, mildly altered from the ones we know, we reduce the amount of overfitting.

Data augmentation is used most often in image classification techniques. Often, we add copies of each picture with an added rotation, shearing, or color jittering.

Let’s imagine we have a huge dataset of animals, and we’re trying to classify which animal is which. We may only have one instance of an alpaca:

alpaca

but we know that this image, sheared:

alpaca sheared

and this image rotated:

alpaca upside-down

are all also examples of an alpaca. When we add these examples of augmented data to our training set, the model won’t overfit as much.

Try Restricting the Featureset
If one of your features is more heavily affecting the parameters of the model, try to run your model without that feature.

For example, let’s say you are writing a program to determine if someone’s loan application should be accepted or rejected. Your model shows that the most significant variable is their race — with all other features the same, the model has a much higher chance of producing an “accept” prediction on an application from a white applicant than on a non-white applicant. This parameter weight may be a sign that the training data contained racial bias. We can try to train the model again, with the race data removed from the featureset.

Reflection
Machine Learning algorithms always must introduce a bias as a function of being programs that are trying to make assumptions and rules by looking at data.

Sometimes the best way to deal with the introduction of bias in a training set is to just acknowledge that it is there. As we try to compensate for the bias, our methods of compensation themselves introduce a bias. It is important to find a balance. The most important thing is to mention the existence of bias in your results, and make sure that all stakeholders know that it exists, so that it is taken into consideration with the decisions made from your model’s results.'''








#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-PYTHON MACHINE LEARNING CHEATSHEET PYTHON MACHINE LEARNING CHEATSHEET PYTHON MACHINE LEARNING CHEATSHEET#-#-#-#
#-#-#-#-PYTHON SK LEARN CHEATSHEET PYTHON SK LEARN CHEATSHEET PYTHON SK LEARN CHEATSHEET PYTHON SK LEARN CHEATSHEET   #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Scikit-learn is a library in Python that provides many unsupervised and supervised learning algorithms. It’s built upon some of the technology you might already be familiar with, like NumPy, pandas, and Matplotlib!

As you build robust Machine Learning programs, it’s helpful to have all the sklearn commands all in one place in case you forget.

LINEAR REGRESSION
Import and create the model:
'''

from sklearn.linear_model import LinearRegression

your_model = LinearRegression()

#Fit:

your_model.fit(x_training_data, y_training_data)
#.coef_: contains the coefficients
#.intercept_: contains the intercept

#Predict:

predictions = your_model.predict(your_x_data)

#.score(): returns the coefficient of determination R²

'''
NAIVE BAYES
Import and create the model:
'''

from sklearn.naive_bayes import MultinomialNB

your_model = MultinomialNB()
#Fit:

your_model.fit(x_training_data, y_training_data)
#Predict:

# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)

# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)

'''
K-NEAREST NEIGHBORS
Import and create the model:
'''

from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()

#Fit:

your_model.fit(x_training_data, y_training_data)

#Predict:

# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)

# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)

'''
K-MEANS
Import and create the model:
'''

from sklearn.cluster import KMeans

your_model = KMeans(n_clusters=4, init='random')

#n_clusters: number of clusters to form and number of centroids to generate
#init: method for initialization
    #k-means++: K-Means++ [default]
    #random: K-Means
#random_state: the seed used by the random number generator [optional]

#Fit:

your_model.fit(x_training_data)

#Predict:

predictions = your_model.predict(your_x_data)

'''
VALIDATING THE MODEL
Import and print accuracy, recall, precision, and F1 score:
'''

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(true_labels, guesses))
print(recall_score(true_labels, guesses))
print(precision_score(true_labels, guesses))
print(f1_score(true_labels, guesses))

#Import and print the confusion matrix:

from sklearn.metrics import confusion_matrix

print(confusion_matrix(true_labels, guesses))

#TRAINING SETS AND TEST SETS

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
train_size: the proportion of the dataset to include in the train split

#test_size: the proportion of the dataset to include in the test split
#random_state: the seed used by the random number generator [optional]
#Robot Emoji
#Happy Coding!