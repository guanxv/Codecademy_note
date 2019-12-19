'''#Machine learning can be branched out into the following categories:
#Supervised Learning
#Unsupervised Learning

Supervised Learning is where the data is labeled and the program learns to predict the output from the input data. For instance, a supervised learning algorithm for credit card fraud detection would take as input a set of recorded transactions. For each transaction, the program would predict if it is fraudulent or not.

Supervised learning problems can be further grouped into regression and classification problems.


Regression:

In regression problems, we are trying to predict a continuous-valued output. Examples are:

What is the housing price in New York?
What is the value of cryptocurrencies?
Classification:

In classification problems, we are trying to predict a discrete number of values. Examples are:

Is this a picture of a human or a picture of an AI?
Is this email spam?
For a quick preview, we will show you an example of supervised learning.

Unsupervised Learning
Unsupervised Learning is a type of machine learning where the program learns the inherent structure of the data based on unlabeled examples.

Clustering is a common unsupervised machine learning approach that finds patterns and structures in unlabeled data by grouping them into clusters.

Some examples:

Social networks clustering topics in their news feed
Consumer sites clustering users for recommendations
Search engines to group similar objects in one cluster

Summary
We have gone over the difference between supervised and unsupervised learning:

Supervised Learning: data is labeled and the program learns to predict the output from the input data
Unsupervised Learning: data is unlabeled and the program learns to recognize the inherent structure in the input data


'''

#Linear Regression----------------------------
#Import and create the model:

from sklearn.linear_model import LinearRegression

your_model = LinearRegression()
#Fit:

your_model.fit(x_training_data, y_training_data)
#.coef_: contains the coefficients
#.intercept_: contains the intercept
#Predict:

predictions = your_model.predict(your_x_data)
#.score(): returns the coefficient of determination R²




# Naive -------------------
# Import and create the model:

from sklearn.naive_bayes import MultinomialNB

your_model = MultinomialNB()
# Fit:

your_model.fit(x_training_data, y_training_data)
# Predict:

# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)

# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data



# K-Nearest Neighbors-------------------------
# Import and create the model:

from sklearn.neigbors import KNeighborsClassifier

your_model = KNeighborsClassifier()
# Fit:

your_model.fit(x_training_data, y_training_data)
# Predict:

# Returns a list of predicted classes - one prediction for every data point
predictions = your_model.predict(your_x_data)

# For every data point, returns a list of probabilities of each class
probabilities = your_model.predict_proba(your_x_data)




# K-Means------------------------------------
# Import and create the model:

from sklearn.cluster import KMeans

your_model = KMeans(n_clusters=4, init='random')
# n_clusters: number of clusters to form and number of centroids to generate
# init: method for initialization
# k-means++: K-Means++ [default]
# random: K-Means
# random_state: the seed used by the random number generator [optional]
# Fit:

your_model.fit(x_training_data)
# Predict:

predictions = your_model.predict(your_x_data)





# Validating the Model---------------------
Import and print accuracy, recall, precision, and F1 score:

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print(accuracy_score(true_labels, guesses))
print(recall_score(true_labels, guesses))
print(precision_score(true_labels, guesses))
print(f1_score(true_labels, guesses))
# Import and print the confusion matrix:

from sklearn.metrics import confusion_matrix

print(confusion_matrix(true_labels, guesses))



# Training Sets and Test Sets--------------------------------
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
# train_size: the proportion of the dataset to include in the train split
# test_size: the proportion of the dataset to include in the test split
# random_state: the seed used by the random number generator [optional]

# ################################Linear Regression########################


# Points and Lines
# In the last exercise, you were probably able to make a rough estimate about the next data point for Sandra’s lemonade stand without thinking too hard about it. For our program to make the same level of guess, we have to determine what a line would look like through those data points.

# A line is determined by its slope and its intercept. In other words, for each point y on a line we can say:

# y = m x + by=mx+b
# where m is the slope, and b is the intercept. y is a given point on the y-axis, and it corresponds to a given x on the x-axis.

# The slope is a measure of how steep the line is, while the intercept is a measure of where the line hits the y-axis.

# When we perform Linear Regression, the goal is to get the “best” m and b for our data. We will determine what “best” means in the next exercises.

# Loss
# When we think about how we can assign a slope and intercept to fit a set of points, we have to define what the best fit is.

# For each data point, we calculate loss, a number that measures how bad the model’s (in this case, the line’s) prediction was. You may have seen this being referred to as error.

# We can think about loss as the squared distance from the point to the line. We do the squared distance (instead of just the distance) so that points above and below the line both contribute to total loss in the same way:


# In this example:

# For point A, the squared distance is 9 (3²)
# For point B, the squared distance is 1 (1²)
# So the total loss, with this model, is 10. If we found a line that had less loss than 10, that line would be a better model for this data.

# Gradient Descent for Intercept
# As we try to minimize loss, we take each parameter we are changing, and move it as long as we are decreasing loss. It’s like we are moving down a hill, and stop once we reach the bottom:


# The process by which we do this is called gradient descent. We move in the direction that decreases our loss the most. Gradient refers to the slope of the curve at any point.

# For example, let’s say we are trying to find the intercept for a line. We currently have a guess of 10 for the intercept. At the point of 10 on the curve, the slope is downward. Therefore, if we increase the intercept, we should be lowering the loss. So we follow the gradient downwards.


# We derive these gradients using calculus. It is not crucial to understand how we arrive at the gradient equation. To find the gradient of loss as intercept changes, the formula comes out to be:

# \frac{2}{N}\sum_{i=1}^{N}-(y_i-(mx_i+b)) 
# N
# 2
# ​	  
# i=1
# ∑
# N
# ​	 −(y 
# i
# ​	 −(mx 
# i
# ​	 +b))
# N is the number of points we have in our dataset
# m is the current gradient guess
# b is the curr


#------------------------------------------------

#---------------------------------------------------
good video to recap all this 

https://www.youtube.com/watch?v=sDv4f4s2SB8

and a good video to review "how to calculate derivative"

https://www.youtube.com/watch?v=54KiyZy145Y&t=162s

and the chain rule for derivative

https://www.youtube.com/watch?v=H-ybCx8gt-8&t=372s

#-------------------------------------------------------

#---------------- sample of a Gradient Descent 1 -----------------------

import codecademylib3_seaborn
import matplotlib.pyplot as plt

def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

#Your step_gradient function here
def step_gradient(b_current, m_current, x, y, learning_rate ):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]
  
#Your gradient_descent function here:  
def gradient_descent(x, y, learning_rate, num_iterations):
  b = 0
  m = 0
  i = range(num_iterations)
  for a in i:
    b, m = step_gradient (b, m, x, y, learning_rate)
  return [b, m]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

#Uncomment the line below to run your gradient_descent function
b, m = gradient_descent(months, revenue, 0.01, 1000)

#Uncomment the lines below to see the line you've settled upon!
y = [m*x + b for x in months]

plt.plot(months, revenue, "o")
plt.plot(months, y)

#---------------------------------------------------------

import codecademylib3_seaborn
from gradient_descent_funcs import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")

X = df["height"]
y = df["weight"]

plt.plot(X, y, 'o')
#plot your line here:
b, m = gradient_descent(X, y, 0.0001, 1000)

y_predictions = [x*m + b for x in X]

plt.plot(X, y_predictions, 'o')

plt.show

#-------------------------------------------


# Scikit-Learn
# Congratulations! You’ve now built a linear regression algorithm from scratch.

# Luckily, we don’t have to do this every time we want to use linear regression. We can use Python’s scikit-learn library. Scikit-learn, or sklearn, is used specifically for Machine Learning. Inside the linear_model module, there is a LinearRegression() function we can use:

from sklearn.linear_model import LinearRegression
You can first create a LinearRegression model, and then fit it to your x and y data:

line_fitter = LinearRegression()
# line_fitter.fit(X, y)
# The .fit() method gives the model two variables that are useful to us:

# the line_fitter.coef_, which contains the slope
# the line_fitter.intercept_, which contains the intercept
# We can also use the .predict() function to pass in x-values and receive the y-values that this line would predict:

y_predicted = line_fitter.predict(X)
# Note: the num_iterations and the learning_rate that you learned about in your own implementation have default values within scikit-learn, so you don’t need to worry about setting them specifically!

#--------------------------------------------

import codecademylib3_seaborn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

temperature = np.array(range(60, 100, 2))
temperature = temperature.reshape(-1, 1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]
         
line_fitter = LinearRegression()
line_fitter.fit(temperature, sales )
sales_predict = line_fitter.predict(temperature)

plt.plot(temperature, sales, 'o')
plt.plot(temperature, sales_predict, 'o')

plt.show()

#------------------------------------------------

Find another dataset, maybe in scikit-learn’s example datasets. Or on Kaggle, a great resource for tons of interesting data.

Try to perform linear regression on your own! If you find any cool linear correlations, make sure to share them!

As a starter, we’ve loaded in the Boston housing dataset. We made the X values the nitrogen oxides concentration (parts per 10 million), and the y values the housing prices. See if you can perform regression on these houses!

#----------------------------------------

Sklearn sample and real source of data for practice:

https://scikit-learn.org/stable/datasets/index.html

https://www.kaggle.com/datasets

#-------------------------------------

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Boston housing dataset
boston = load_boston()

df = pd.DataFrame(boston.data, columns = boston.feature_names)

# Set the x-values to the nitrogen oxide concentration:
X = df[['NOX']]
# Y-values are the prices:
y = boston.target

# Can we do linear regression on this?

line_fitter = LinearRegression()
line_fitter.fit(X, y )
yy_predict = line_fitter.predict(X)


plt.scatter(X, y, alpha=0.4)
# Plot line here:
plt.plot(X, yy_predict)

plt.title("Boston Housing Dataset")
plt.xlabel("Nitric Oxides Concentration")
plt.ylabel("House Price ($)")
plt.show()

