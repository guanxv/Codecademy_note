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

#----------------------------------------------

import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

#print(df.head())
prod_per_year = df.groupby("year").totalprod.mean().reset_index()

#print(prod_per_year.head())
X = prod_per_year['year']
X = X.values.reshape(-1 ,1)

y = prod_per_year['totalprod']
#print(type(y))

regr = linear_model.LinearRegression()
regr.fit(X, y)

y_predict = regr.predict(X)

X_future = np.array(range(2013,2051))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)

plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.show()
#--------------------------------------------
Multiple Linear Regression
#----------------------------------------------

import codecademylib3_seaborn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['size_sqft','building_age_yrs']]
y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

ols = LinearRegression()

ols.fit(x_train, y_train)

# Plot the figure

fig = plt.figure(1, figsize=(6, 4))
plt.clf()

elev = 43.5
azim = -110

ax = Axes3D(fig, elev=elev, azim=azim)

ax.scatter(x_train[['size_sqft']], x_train[['building_age_yrs']], y_train, c='k', marker='+')

ax.plot_surface(np.array([[0, 0], [4500, 4500]]), np.array([[0, 140], [0, 140]]), ols.predict(np.array([[0, 0, 4500, 4500], [0, 140, 0, 140]]).T).reshape((2, 2)), alpha=.7)

ax.set_xlabel('Size (ft$^2$)')
ax.set_ylabel('Building Age (Years)')
ax.set_zlabel('Rent ($)')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# Add the code below:

plt.show()


# split Train_test set 
                                         
import codecademylib3_seaborn
import pandas as pd

# import train_test_split
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state = 6)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)
 
 
 #---------------------------------------------------
 
#MULTIPLE LINEAR REGRESSION
#Multiple Linear Regression: Scikit-Learn
#Now we have the training set and the test set, let’s use scikit-learn to build the linear regression model!

#The steps for multiple linear regression in scikit-learn are identical to the steps for simple linear regression. Just like simple linear regression, we need to import LinearRegression from the linear_model module:

from sklearn.linear_model import LinearRegression
#Then, create a LinearRegression model, and then fit it to your x_train and y_train data:

mlr = LinearRegression()

mlr.fit(x_train, y_train) 
# finds the coefficients and the intercept value
#We can also use the .predict() function to pass in x-values. It returns the y-values that this plane would predict:

y_predicted = mlr.predict(x_test)
# takes values calculated by `.fit()` and the `x` values, plugs them into the multiple linear regression equation, and calculates the predicted y values. 
We will start by using two of these columns to teach you how to predict the values of the dependent variable, prices.


'''
Now we have:

x_test
x_train
y_test
y_train
and y_predict!
4.
To see this model in action, let’s test it on Sonny’s apartment in Greenpoint, Brooklyn!

Or if you reside in New York, plug in your own apartment’s values and see if you are over or underpaying!

This is a 1BR/1Bath apartment that is 620 ft². We have pulled together the data for you:

Features	Sonny’s Apartment
bedrooms	1
bathrooms	1
size_sqft	620 ft²
min_to_subway	16 min
floor	1
building_age_yrs	98 (built in 1920)
no_fee	1
has_roofdeck	0
has_washer_dryer	Yas
has_doorman	0
has_elevator	0
has_dishwasher	1
has_patio	1
has_gym	0

'''
# Sonny doesn't have an elevator so the 11th item in the list is a 0
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]

predict = mlr.predict(sonny_apartment)

print("Predicted rent: $%.2f" % predict)
#The result is:

#Predicted rent: $2393.58
#cAnd Sonny is only paying $2,000. Yay!

# -------------- code -----------------------
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

# Add the code here:
mlr = LinearRegression()

mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# test of sonny_apartment

sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]

predict = mlr.predict(sonny_apartment)

print("Predicted rent: $%.2f" % predict)

#plot the result

# Create a scatter plot
plt.scatter(y_test, y_predict, alpha=0.4)

# Create x-axis label and y-axis label

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")


# Create a title

plt.title("Actual Rent vs Predicted Rent")

# show the plot
plt.show()

#---------------------------
'''
MULTIPLE LINEAR REGRESSION
Multiple Linear Regression Equation
Now that we have implemented Multiple Linear Regression, we will learn how to tune and evaluate the model. Before we do that, however, it’s essential to learn the equation behind it.

Equation 6.1 The equation for multiple linear regression that uses two independent variables is this:

y = b + m_{1}x_{1} + m_{2}x_{2}y=b+m 
1
​	 x 
1
​	 +m 
2
​	 x 
2
​	 
Equation 6.2 The equation for multiple linear regression that uses three independent variables is this:

y = b + m_{1}x_{1} + m_{2}x_{2} + m_{3}x_{3}y=b+m 
1
​	 x 
1
​	 +m 
2
​	 x 
2
​	 +m 
3
​	 x 
3
​	 
Equation 6.3 As a result, since multiple linear regression can use any number of independent variables, its general equation becomes:

y = b + m_{1}x_{1} + m_{2}x_{2} + ... + m_{n}x_{n}y=b+m 
1
​	 x 
1
​	 +m 
2
​	 x 
2
​	 +...+m 
n
​	 x 
n
​	 
Here, m1, m2, m3, … mn refer to the coefficients, and b refers to the intercept that you want to find. You can plug these values back into the equation to compute the predicted y values.

Remember, with sklearn‘s LinearRegression() method, we can get these values with ease.

The .fit() method gives the model two variables that are useful to us:

.coef_, which contains the coefficients
.intercept_, which contains the intercept
After performing multiple linear regression, you can print the coefficients using .coef_.

Coefficients are most helpful in determining which independent variable carries more weight. For example, a coefficient of -1.345 will impact the rent more than a coefficient of 0.238, with the former impacting prices negatively and latter positively.
'''
# code ---

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Input code here:

print(mlr.coef_)

print(mlr.intercept_)

'''MULTIPLE LINEAR REGRESSION
Correlations
In our Manhattan model, we used 14 variables, so there are 14 coefficients:

[ -302.73009383  1199.3859951  4.79976742  -24.28993151  24.19824177  -7.58272473  -140.90664773  48.85017415  191.4257324  -151.11453388  89.408889  -57.89714551  -19.31948556  -38.92369828 ]]
bedrooms - number of bedrooms
bathrooms - number of bathrooms
size_sqft - size in square feet
min_to_subway - distance from subway station in minutes
floor - floor number
building_age_yrs - building’s age in years
no_fee - has no broker fee (0 for fee, 1 for no fee)
has_roofdeck - has roof deck (0 for no, 1 for yes)
has_washer_dryer - has in-unit washer/dryer (0/1)
has_doorman - has doorman (0/1)
has_elevator - has elevator (0/1)
has_dishwasher - has dishwasher (0/1)
has_patio - has patio (0/1)
has_gym - has gym (0/1)
To see if there are any features that don’t affect price linearly, let’s graph the different features against rent.

Interpreting graphs

In regression, the independent variables will either have a positive linear relationship to the dependent variable, a negative linear relationship, or no relationship. A negative linear relationship means that as X values increase, Y values will decrease. Similarly, a positive linear relationship means that as X values increase, Y values will also increase.

Graphically, when you see a downward trend, it means a negative linear relationship exists. When you find an upward trend, it indicates a positive linear relationship. Here are two graphs indicating positive and negative linear relationships:

'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

# Input code here:

plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)

plt.show()

#------------------------
"""
MULTIPLE LINEAR REGRESSION
Evaluating the Model's Accuracy
When trying to evaluate the accuracy of our multiple linear regression model, one technique we can use is Residual Analysis.

The difference between the actual value y, and the predicted value ŷ is the residual e. The equation is:

e = y - ŷ
​	 
In the StreetEasy dataset, y is the actual rent and the ŷ is the predicted rent. The real y values should be pretty close to these predicted y values.

sklearn‘s linear_model.LinearRegression comes with a .score() method that returns the coefficient of determination R² of the prediction.

The coefficient R² is defined as:

1 - u / v
​	 
where u is the residual sum of squares:

((y - y_predict) ** 2).sum()
and v is the total sum of squares (TSS):

((y - y.mean()) ** 2).sum()
The TSS tells you how much variation there is in the y variable.

R² is the percentage variation in y explained by all the x variables together.

For example, say we are trying to predict rent based on the size_sqft and the bedrooms in the apartment and the R² for our model is 0.72 — that means that all the x variables (square feet and number of bedrooms) together explain 72% variation in y (rent).

Now let’s say we add another x variable, building’s age, to our model. By adding this third relevant x variable, the R² is expected to go up. Let say the new R² is 0.95. This means that square feet, number of bedrooms and age of the building together explain 95% of the variation in the rent.
w
The best possible R² is 1.00 (and it can be negative because the model can be arbitrarily worse). Usually, a R² of 0.70 is considered good.

"""
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Input code here:
print("Train score:")
print(mlr.score(x_train, y_train ))

print("Test score:")
print(mlr.score(x_test, y_test))

residuals = y_predict - y_test

plt.scatter(y_predict, residuals, alpha=0.4)
plt.title('Residual Analysis')

plt.show()

#-----------------------------------
"""
MULTIPLE LINEAR REGRESSION
Rebuild the Model
Now let’s rebuild the model using the new features as well as evaluate the new model to see if we improved!

For Manhattan, the scores returned:

Train score: 0.772546055982
Test score:  0.805037197536
For Brooklyn, the scores returned:

Train score: 0.613221453798
Test score:  0.584349923873
For Queens, the scores returned:

Train score: 0.665836031009
Test score:  0.665170319781
For whichever borough you used, let’s see if we can improve these scores!

Instructions
1.
Print the coefficients again to see which ones are strongest.

2.
Currently the x should look something like:

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
Remove some of the features that don’t have strong correlations and see if your scores improved!

Post your best model in the Slack channel!

There is no right answer! Try building a model using different features!

"""

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft',  'floor', 'building_age_yrs', 'has_roofdeck', 'has_washer_dryer',  'has_elevator', 'has_gym']]

#x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")

plt.show()

# zoe_apartment = [[1, 1, 620, 16, 1, 98, 0, 0, 1, 0, 0, 0, 1, 0]]
# predict = model.predict(zoe_apartment)
# print("Predicted rent: $%.2f" % predict)

#-------Review-------------

"""
MULTIPLE LINEAR REGRESSION
Review
Great work! Let’s review the concepts before you move on:

Multiple Linear Regression uses two or more variables to make predictions about another variable:
y = b + m_{1}x_{1} + m_{2}x_{2} + ... + m_{n}x_{n}y=b+m 
1
​	 x 
1
​	 +m 
2
​	 x 
2
​	 +...+m 
n
​	 x 
n
​	 
Multiple linear regression uses a set of independent variables and a dependent variable. It uses these variables to learn how to find optimal parameters. It takes a labeled dataset and learns from it. Once we confirm that it’s learned correctly, we can then use it to make predictions by plugging in new x values.
We can use scikit-learn’s LinearRegression() to perform multiple linear regression.
Residual Analysis is used to evaluate the regression model’s accuracy. In other words, it’s used to see if the model has learned the coefficients correctly.
Scikit-learn’s linear_model.LinearRegression comes with a .score() method that returns the coefficient of determination R² of the prediction. The best score is 1.0.

"""

#------------------------------------------

'''
Regression vs. Classification
Learn about the two types of Supervised Learning algorithms.

Machine Learning is a set of many different techniques that are each suited to answering different types of questions.

We have previously divided algorithms into two groups — Supervised Learning vs Unsupervised Learning. Supervised learning algorithms use labeled data as input while unsupervised learning algorithms use unlabeled data. However, we can further distinguish machine learning algorithms by the output they produce. In terms of output, two main types of machine learning models exist: those for regression and those for classification.

Regression
Regression is used to predict outputs that are continuous. The outputs are quantities that can be flexibly determined based on the inputs of the model rather than being confined to a set of possible labels.

For example:

Predict the height of a potted plant from the amount of rainfall
Predict salary based on someone’s age and availability of high-speed internet
Predict a car’s MPG (miles per gallon) based on size and model year
Regression GIF


Linear regression is the most popular regression algorithm. It is often underrated because of its relative simplicity. In a business setting, it could be used to predict the likelihood that a customer will churn or the revenue a customer will generate. More complex models may fit this data better, at the cost of losing simplicity.

Classification
Classification is used to predict a discrete label. The outputs fall under a finite set of possible outcomes. Many situations have only two possible outcomes. This is called binary classification (True/False, 0 or 1, Hotdog / not Hotdog).

For example:

Predict whether an email is spam or not
Predict whether it will rain or not
Predict whether a user is a power user or a casual user
Multi-label classification is when there are multiple possible outcomes. It is useful for customer segmentation, image categorization, and sentiment analysis for understanding text. To perform these classifications, we use models like Naive Bayes, K-Nearest Neighbors, and SVMs.

Classification GIF


Choosing a model is a critical step in the Machine Learning process. It is important that the model fits the question at hand. When you choose the right model, you are already one step closer to getting meaningful and interesting results.
'''


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
LOGISTIC REGRESSION
Introduction
When an email lands in your inbox, how does your email service know whether it’s a real email or spam? This evaluation is made billions of times per day, and one way it can be done is with Logistic Regression. Logistic Regression is a supervised machine learning algorithm that uses regression to predict the continuous probability, ranging from 0 to 1, of a data sample belonging to a specific category, or class. Then, based on that probability, the sample is classified as belonging to the more probable class, ultimately making Logistic Regression a classification algorithm.

In our spam filtering example, a Logistic Regression model would predict the probability of an incoming email being spam. If that predicted probability is greater than or equal to 0.5, the email is classified as spam. We would call spam the positive class, with the label 1, since the positive class is the class our model is looking to detect. If the predicted probability is less than 0.5, the email is classified as ham (a real email). We would call ham the negative class, with the label 0. This act of deciding which of two classes a data sample belongs to is called binary classification.

Some other examples of what we can classify with Logistic Regression include:

Disease survival —Will a patient, 5 years after treatment for a disease, still be alive?
Customer conversion —Will a customer arriving on a sign-up page enroll in a service?
In this lesson you will learn how to perform Logistic Regression and use it to make classifications on your own data!

If you are unfamiliar with Linear Regression, we recommend you go check out our Linear Regression course before proceeding to Logistic Regression. If you are familiar, let’s dive in!
'''
import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from exam import hours_studied, passed_exam, math_courses_taken

# Scatter plot of exam passage vs number of hours studied
plt.scatter(hours_studied.ravel(), passed_exam, color='black', zorder=20)
plt.ylabel('passed/failed')
plt.xlabel('hours studied')

plt.show()

'''
LOGISTIC REGRESSION
Linear Regression Approach
With the data from Codecademy University, we want to predict whether each student will pass their final exam. And the first step to making that prediction is to predict the probability of each student passing. Why not use a Linear Regression model for the prediction, you might ask? Let’s give it a try.

Recall that in Linear Regression, we fit a regression line of the following form to the data:

y = b_{0} + b_{1}x_{1} + b_{2}x_{2} +\cdots + b_{n}x_{n}y=b 
0
​	 +b 
1
​	 x 
1
​	 +b 
2
​	 x 
2
​	 +⋯+b 
n
​	 x 
n
​	 
where

y is the value we are trying to predict
b_0 is the intercept of the regression line
b_1, b_2, … b_n are the coefficients of the features x_1, x_2, … x_n of the regression line
For our data points y is either 1 (passing), or 0 (failing), and we have one feature, num_hours_studied. Below we fit a Linear Regression model to our data and plotted the results, with the line of best fit in red.

Linear Regression Model on Exam Data
A problem quickly arises. For low values of num_hours_studied the regression line predicts negative probabilities of passing, and for high values of num_hours_studied the regression line predicts probabilities of passing greater than 1. These probabilities are meaningless! We get these meaningless probabilities since the output of a Linear Regression model ranges from -∞ to +∞.'''

'''
LOGISTIC REGRESSION
Logistic Regression
We saw that the output of a Linear Regression model does not provide the probabilities we need to predict whether a student passes the final exam. Step in Logistic Regression!

In Logistic Regression we are also looking to find coefficients for our features, but this time we are fitting a logistic curve to the data so that we can predict probabilities. Described below is an overview of how Logistic Regression works. Don’t worry if something does not make complete sense right away, we will dig into each of these steps in further detail in the remaining exercises!

To predict the probability of a data sample belonging to a class, we:

initialize all feature coefficients and intercept to 0
multiply each of the feature coefficients by their respective feature value to get what is known as the log-odds
place the log-odds into the sigmoid function to link the output to the range [0,1], giving us a probability
By comparing the predicted probabilities to the actual classes of our data points, we can evaluate how well our model makes predictions and use gradient descent to update the coefficients and find the best ones for our model.

To then make a final classification, we use a classification threshold to determine whether the data sample belongs to the positive class or the negative class.'''

import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from exam import hours_studied, passed_exam
from plotter import plot_data

# Create logistic regression model
model = LogisticRegression()
model.fit(hours_studied,passed_exam)

# Plug sample data into fitted model
sample_x = np.linspace(-16.65, 33.35, 300).reshape(-1,1)
probability = model.predict_proba(sample_x)[:,1]

# Function to plot exam data and logistic regression curve
plot_data(model)

# Show the plot
plt.show()

# Lowest and highest probabilities
lowest = 0

highest = 1


'''
LOGISTIC REGRESSION
Log-Odds
In Linear Regression we multiply the coefficients of our features by their respective feature values and add the intercept, resulting in our prediction, which can range from -∞ to +∞. In Logistic Regression, we make the same multiplication of feature coefficients and feature values and add the intercept, but instead of the prediction, we get what is called the log-odds.

The log-odds are another way of expressing the probability of a sample belonging to the positive class, or a student passing the exam. In probability, we calculate the odds of an event occurring as follows:

Odds = \frac{P(event\ occurring)}{P(event\ not\ occurring)}Odds= 
P(event not occurring)
P(event occurring)
​	 
The odds tell us how many more times likely an event is to occur than not occur. If a student will pass the exam with probability 0.7, they will fail with probability 1 - 0.7 = 0.3. We can then calculate the odds of passing as:

Odds\ of\ passing = \frac{0.7}{0.3} = 2.\overline{33}Odds of passing= 
0.3
0.7
​	 =2. 
33
 
The log-odds are then understood as the logarithm of the odds!

Log\ odds\ of\ passing = log(2.\overline{33}) = 0.847Log odds of passing=log(2. 
33
 )=0.847
For our Logistic Regression model, however, we calculate the log-odds, represented by z below, by summing the product of each feature value by its respective coefficient and adding the intercept. This allows us to map our feature values to a measure of how likely it is that a data sample belongs to the positive class.

z = b_{0}+b_{1}x_{1} + \cdots + b_{n}x_{n}z=b 
0
​	 +b 
1
​	 x 
1
​	 +⋯+b 
n
​	 x 
n
​	 
b_0 is the intercept
b_1, b_2, … b_n are the coefficients of the features x_1, x_2, … x_n
This kind of multiplication and summing is known as a dot product.

We can perform a dot product using numpy‘s np.dot() method! Given feature matrix features, coefficient vector coefficients, and an intercept, we can calculate the log-odds in numpy as follows:

log_odds = np.dot(features, coefficients) + intercept
np.dot() will take each row, or student, in features and multiply each individual feature value by its respective coefficient in coefficients, summing the result, as shown below.

Matrix Multiplication
We then add in the intercept to get the log-odds!'''

import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

# Create your log_odds() function here

def log_odds(features, coefficients, intercept):
  return np.dot(features, coefficients) + intercept

# Calculate the log-odds for the Codecademy University data here
calculated_log_odds = log_odds(hours_studied, calculated_coefficients, intercept)

print(calculated_log_odds)

'''
LOGISTIC REGRESSION
Sigmoid Function
How did our Logistic Regression model create the S-shaped curve we previously saw? The answer is the Sigmoid Function.

Sigmoid Function
The Sigmoid Function is a special case of the more general Logistic Function, where Logistic Regression gets its name. Why is the Sigmoid Function so important? By plugging the log-odds into the Sigmoid Function, defined below, we map the log-odds z to the range [0,1].

h(z)=\frac{1}{1+e^{-z}}h(z)= 
1+e 
−z
 
1
​	 
e^(-z) is the exponential function, which can be written in numpy as np.exp(-z)
This enables our Logistic Regression model to output the probability of a sample belonging to the positive class, or in our case, a student passing the final exam!'''

import codecademylib3_seaborn
import numpy as np
from exam import calculated_log_odds

# Create your sigmoid function here
def sigmoid(z):
  denominator = 1 + np.exp(-z)
  return 1/denominator

# Calculate the sigmoid of the log-odds here
probabilities = sigmoid(calculated_log_odds)
print(probabilities)

'''
LOGISTIC REGRESSION
Log-Loss I
Now that we understand how a Logistic Regression model makes its probability predictions, what coefficients and intercept should we use in our model to best predict whether a student will pass the exam? To answer this question we need a way to evaluate how well a given model fits the data we have.

The function used to evaluate the performance of a machine learning model is called a loss function, or a cost function. To evaluate how “good a fit” a model is, we calculate the loss for each data sample (how wrong the model’s prediction was) and then average the loss across all samples. The loss function for Logistic Regression, known as Log Loss, is given below:

-\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}log(h(z^{(i)})) + (1-y^{(i)})log(1-h(z^{(i)}))]− 
m
1
​	  
i=1
∑
m
​	 [y 
(i)
 log(h(z 
(i)
 ))+(1−y 
(i)
 )log(1−h(z 
(i)
 ))]
m is the total number of data samples
y_i is the class of data sample i
z_i is the log-odds of sample i
h(z_i) is the sigmoid of the log-odds of sample i, which is the probability of sample i belonging to the positive class
The log-loss function might seem scary, but don’t worry, we are going to break it down in the next exercise!

The goal of our Logistic Regression model is to find the feature coefficients and intercept, which shape the logistic function, that minimize log-loss for our training data!

'''
'''
LOGISTIC REGRESSION
Log Loss II
J(\mathbf{b}) = -\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}log(h(z^{(i)})) + (1-y^{(i)})log(1-h(z^{(i)}))]J(b)=− 
m
1
​	  
i=1
∑
m
​	 [y 
(i)
 log(h(z 
(i)
 ))+(1−y 
(i)
 )log(1−h(z 
(i)
 ))]
Let’s go ahead and break down our log-loss function into two separate parts so it begins to make more sense. Consider the case when a data sample has class y = 1, or for our data when a student passed the exam. The right-side of the equation drops out because we end up with 1 - 1 (or 0) multiplied by some value. The loss for that individual student becomes:

loss_{y=1} = -log(h(z^{(i)}))loss 
y=1
​	 =−log(h(z 
(i)
 ))
The loss for a student who passed the exam is just the log of the probability the student passed the exam!

And for a student who fails the exam, where a sample has class y = 0, the left-side of the equation drops out and the loss for that student becomes:

loss_{y = 0} = -log(1-h(z^{(i)}))loss 
y=0
​	 =−log(1−h(z 
(i)
 ))
The loss for a student who failed the exam is the log of one minus the probability the student passed the exam, which is just the log of the probability the student failed the exam!

Let’s take a closer look at what is going on with our loss function by graphing the loss of individual samples when the class label is y = 1 and y = 0.

Log Loss for Positive and Negative Samples
Let’s go back to our Codecademy University data and consider four possible cases:

Class	Model Probability y = 1	Correct?	Loss
y = 1	High	Yes	Low
y = 1	Low	No	High
y = 0	High	No	High
y = 0	Low	Yes	Low

From the graphs and the table you can see that confident correct predictions result in small losses, while confident incorrect predictions result in large losses that approach infinity. This makes sense! We want to punish our model with an increasing loss as it makes progressively incorrect predictions, and we want to reward the model with a small loss as it makes correct predictions.

Just like in Linear Regression, we can then use gradient descent to find the coefficients that minimize log-loss across all of our training data.
'''

import numpy as np
from exam import passed_exam, probabilities, probabilities_2

# Function to calculate log-loss
def log_loss(probabilities,actual_class):
  return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities)))

# Print passed_exam here
print(passed_exam)


# Calculate and print loss_1 here
loss_1 = log_loss(probabilities,passed_exam )
print(loss_1)

# Calculate and print loss_2 here
loss_2 = log_loss(probabilities_2,passed_exam )
print(loss_2)

'''

Classification Thresholding
Many machine learning algorithms, including Logistic Regression, spit out a classification probability as their result. Once we have this probability, we need to make a decision on what class the sample belongs to. This is where the classification threshold comes in!

The default threshold for many algorithms is 0.5. If the predicted probability of an observation belonging to the positive class is greater than or equal to the threshold, 0.5, the classification of the sample is the positive class. If the predicted probability of an observation belonging to the positive class is less than the threshold, 0.5, the classification of the sample is the negative class.

Threshold at 0.5
We can choose to change the threshold of classification based on the use-case of our model. For example, if we are creating a Logistic Regression model that classifies whether or not an individual has cancer, we want to be more sensitive to the positive cases, signifying the presence of cancer, than the negative cases.

In order to ensure that most patients with cancer are identified, we can move the classification threshold down to 0.3 or 0.4, increasing the sensitivity of our model to predicting a positive cancer classification. While this might result in more overall misclassifications, we are now missing fewer of the cases we are trying to detect: actual cancer patients.

Threshold at 0.4

'''
'''
Instructions
1.
Let’s use all the knowledge we’ve gathered to create a function that performs thresholding and makes class predictions! Define a function predict_class() that takes a features matrix, a coefficients vector, an intercept, and a threshold as parameters. Return threshold.

2.
In predict_class(), calculate the log-odds using the log_odds() function we defined earlier. Store the result in calculated_log_odds, and return calculated_log_odds.

3.
Still in predict_class(), find the probabilities that the samples belong to the positive class. Create a variable probabilities, and give it the value returned by calling sigmoid() on calculated_log_odds. Return probabilities.

4.
Return 1 for all values within probabilities equal to or above threshold, and 0 for all values below threshold.

Since we are working with numpy objects, we can compare all the values in an array with some threshold using the following syntax:

np.where(array_to_check >= threshold, 1, 0)
If a value in array_to_check is above threshold, the output is 1. If a value in array_to_check is below threshold, the output is 0.

5.
Let’s make final classifications on our Codecademy University data to see which students passed the exam. Use the predict_class() function with hours_studied, calculated_coefficients, intercept, and a threshold of 0.5 as parameters. Store the results in final_results, and print final_results.
'''

import numpy as np
from exam import hours_studied, calculated_coefficients, intercept

def log_odds(features, coefficients,intercept):
  return np.dot(features,coefficients) + intercept

def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1/denominator

# Create predict_class() function here

def predict_class(features, coefficients, intercept, threshold):
  calculated_log_odds = log_odds(features, coefficients, intercept)
  probabilities = sigmoid(calculated_log_odds)
  
  return np.where(probabilities >= threshold, 1, 0) 
  
# Make final classifications on Codecademy University data here

final_results = predict_class(hours_studied, calculated_coefficients,intercept, 0.5) 

print(final_results)


'''
LOGISTIC REGRESSION
Scikit-Learn
Now that you know the inner workings of how Logistic Regression works, let’s learn how to easily and quickly create Logistic Regression models with sklearn! sklearn is a Python library that helps build, train, and evaluate Machine Learning models.

To take advantage of sklearn‘s abilities, we can begin by creating a LogisticRegression object.

model = LogisticRegression()
After creating the object, we need to fit our model on the data. When we fit the model with sklearn it will perform gradient descent, repeatedly updating the coefficients of our model in order to minimize the log-loss. We train — or fit — the model using the .fit() method, which takes two parameters. The first is a matrix of features, and the second is a matrix of class labels.

model.fit(features, labels)
Now that the model is trained, we can access a few useful attributes of the LogisticRegression object.

model.coef_ is a vector of the coefficients of each feature
model.intercept_ is the intercept b_0
With our trained model we are able to predict whether new data points belong to the positive class using the .predict() method! .predict() takes a matrix of features as a parameter and returns a vector of labels 1 or 0 for each sample. In making its predictions, sklearn uses a classification threshold of 0.5.

model.predict(features)
If we are more interested in the predicted probability of the data samples belonging to the positive class than the actual class, we can use the .predict_proba() method. predict_proba() also takes a matrix of features as a parameter and returns a vector of probabilities, ranging from 0 to 1, for each sample.

model.predict_proba(features)
Before proceeding, one important note is that sklearn‘s Logistic Regression implementation requires feature data to be normalized. Normalization scales all feature data to vary over the same range. sklearn‘s Logistic Regression requires normalized feature data due to a technique called Regularization that it uses under the hood. Regularization is out of the scope of this lesson, but in order to ensure the best results from our model, we will be using a normalized version of the data from our Codecademy University example.

'''
'''
Instructions
1.
Let’s build, train and evaluate a Logistic Regression model in sklearn for our Codecademy University data! We’ve imported sklearn and the LogisiticRegression classifier for you. Create a Logistic Regression model named model.

2.
Train the model using hours_studied_scaled as the training features and passed_exam as the training labels.

3.
Save the coefficients of the model to the variable calculated_coefficients, and the intercept of the model to intercept. Print calculated_coefficients and intercept.

4.
The next semester a group of students in the Introductory Machine Learning course want to predict their final exam scores based on how much they intended to study for the exam. The number of hours each student thinks they will study, normalized, is given in guessed_hours_scaled. Use model to predict the probability that each student will pass the final exam, and save the probabilities to passed_predictions.

5.
That same semester, the Data Science department decides to update the final exam passage model to consider two features instead of just one. During the final exam, students were asked to estimate how much time they spent studying, as well as how many previous math courses they have taken. The student responses, along with their exam results, were split into training and test sets. The training features, normalized, are given to you in exam_features_scaled_train, and the students’ results on the final are given in passed_exam_2_train.

Create a new Logistic Regression model named model_2 and train it on exam_features_scaled_train and passed_exam_2_train.

6.
Use the model you just trained to predict whether each student in the test set, exam_features_scaled_test, will pass the exam and save the predictions to passed_predictions_2. Print passed_predictions_2.

Compare the predictions to the actual student performance on the exam in the test set. How well did your model do?

To make predictions, call model_2.predict() on exam_features_scaled_test.

Print passed_exam_2_test to see how well your model performed!

'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from exam import hours_studied_scaled, passed_exam, exam_features_scaled_train, exam_features_scaled_test, passed_exam_2_train, passed_exam_2_test, guessed_hours_scaled

# Create and fit logistic regression model here
model = LogisticRegression()
model.fit(hours_studied_scaled, passed_exam)

# Save the model coefficients and intercept here
calculated_coefficients = model.coef_
intercept = model.intercept_
print(calculated_coefficients)
print(intercept)

# Predict the probabilities of passing for next semester's students here
passed_predictions = model.predict_proba(guessed_hours_scaled)
print(passed_predictions)

passed_predictions1 = model.predict(guessed_hours_scaled)
print(passed_predictions1)

# Create a new model on the training data with two features here
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled_train , passed_exam_2_train)

# Predict whether the students will pass here
passed_predictions_2 = model_2.predict(exam_features_scaled_test)
'''
LOGISTIC REGRESSION
Feature Importance
One of the defining features of Logistic Regression is the interpretability we have from the feature coefficients. How to handle interpreting the coefficients depends on the kind of data you are working with (normalized or not) and the specific implementation of Logistic Regression you are using. We’ll discuss how to interpret the feature coefficients from a model created in sklearn with normalized feature data.

Since our data is normalized, all features vary over the same range. Given this understanding, we can compare the feature coefficients’ magnitudes and signs to determine which features have the greatest impact on class prediction, and if that impact is positive or negative.

Features with larger, positive coefficients will increase the probability of a data sample belonging to the positive class
Features with larger, negative coefficients will decrease the probability of a data sample belonging to the positive class
Features with small, positive or negative coefficients have minimal impact on the probability of a data sample belonging to the positive class
Given cancer data, a logistic regression model can let us know what features are most important for predicting survival after, for example, five years from diagnosis. Knowing these features can lead to a better understanding of outcomes, and even lives saved!
'''
'''
Instructions
1.
Let’s revisit the sklearn Logistic Regression model we fit to our exam data in the last exercise. Remember, the two features in the new model are the number of hours studied and the number of previous math courses taken.

Using the model, given to you as model_2 in the code editor, save the feature coefficients to the variable coefficients.

2.
In order to visualize the coefficients, let’s pull them out of the numpy array in which they are currently stored. With numpys tolist() method we can convert the array into a list and grab the values we want to visualize.

Below your original assignment of coefficients, update coefficients to equal coefficients.tolist()[0].

3.
Create a bar graph comparing the feature coefficients with matplotlib‘s plt.bar() method. Which feature appears to be more important in determining whether or not a student will pass the Introductory Machine Learning final exam?
'''

import codecademylib3_seaborn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from exam import exam_features_scaled, passed_exam_2

# Train a sklearn logistic regression model on the normalized exam data
model_2 = LogisticRegression()
model_2.fit(exam_features_scaled,passed_exam_2)

# Assign and update coefficients
coefficients = model_2.coef_
coefficients = coefficients.tolist()[0]

# Plot bar graph
#plt.bar(range(len(coefficients)),coefficients)
#plt.show()

plt.bar([1,2],coefficients)
plt.xticks([1,2],['hours studied','math courses taken'])
plt.xlabel('feature')
plt.ylabel('coefficient')

plt.show()

'''
LOGISTIC REGRESSION
Review
Congratulations! You just learned how a Logistic Regression model works and how to fit one to a dataset. Class is over, and the final exam for Codecademy University’s Introductory Machine Learning is around the corner. Do you predict that you will pass? Let’s do some review to make sure.

Logistic Regression is used to perform binary classification, predicting whether a data sample belongs to a positive (present) class, labeled 1 and the negative (absent) class, labeled 0.
The Sigmoid Function bounds the product of feature values and their coefficients, known as the log-odds, to the range [0,1], providing the probability of a sample belonging to the positive class.
A loss function measures how well a machine learning model makes predictions. The loss function of Logistic Regression is log-loss.
A Classification Threshold is used to determine the probabilistic cutoff for where a data sample is classified as belonging to a positive or negative class. The standard cutoff for Logistic Regression is 0.5, but the threshold can be higher or lower depending on the nature of the data and the situation.
Scikit-learn has a Logistic Regression implementation that allows you to fit a model to your data, find the feature coefficients, and make predictions on new data samples.
The coefficients determined by a Logistic Regression model can be used to interpret the relative importance of each feature in predicting the class of a data sample.

'''
'''
DECISION TREES
Decision Trees
Decision trees are machine learning models that try to find patterns in the features of data points. Take a look at the tree on this page. This tree tries to predict whether a student will get an A on their next test.

By asking questions like “What is the student’s average grade in the class” the decision tree tries to get a better understanding of their chances on the next test.

In order to make a classification, this classifier needs a data point with four features:

The student’s average grade in the class.
The number of hours the student plans on studying for the test.
The number of hours the student plans on sleeping the night before the test.
Whether or not the student plans on cheating.
For example, let’s say that somebody has a “B” average in the class, studied for more than 3 hours, slept less than 5 hours before the test, and doesn’t plan to cheat. If we start at the top of the tree and take the correct path based on that data, we’ll arrive at a leaf node that predicts the person will not get an A on the next test.

In this course, you’ll learn how to create a tree like this!

DECISION TREES
Making Decision Trees
If we’re given this magic tree, it seems relatively easy to make classifications. But how do these trees get created in the first place? Decision trees are supervised machine learning models, which means that they’re created from a training set of labeled data. Creating the tree is where the learning in machine learning happens.

Take a look at the gif on this page. We begin with every point in the training set at the top of the tree. These training points have labels — the red points represent students that didn’t get an A on a test and the green points represent students that did get an A on a test .

We then decide to split the data into smaller groups based on a feature. For example, that feature could be something like their average grade in the class. Students with an A average would go into one set, students with a B average would go into another subset, and so on.

Once we have these subsets, we repeat the process — we split the data in each subset again on a different feature.

Eventually, we reach a point where we decide to stop splitting the data into smaller groups. We’ve reached a leaf of the tree. We can now count up the labels of the data in that leaf. If an unlabeled point reaches that leaf, it will be classified as the majority label.

We can now make a tree, but how did we know which features to split the data set with? After all, if we started by splitting the data based on the number of hours they slept the night before the test, we’d end up with a very different tree that would produce very different results. How do we know which tree is best? We’ll tackle this question soon!

'''

'''

DECISION TREES
Cars
In this lesson, we’ll create a decision tree build off of a dataset about cars. When considering buying a car, what factors go into making that decision?

Each car can fall into four different classes which represent how satisfied someone would be with purchasing the car — unacc (unacceptable), acc (acceptable), good, vgood.

Each car has 6 features:

The price of the car which can be "vhigh", "high", "med", or "low".
The cost of maintaining the car which can be "vhigh", "high", "med", or "low".
The number of doors which can be "2", "3", "4", "5more".
The number of people the car can hold which can be "2", "4", or "more".
The size of the trunk which can be "small", "med", or "big".
The safety rating of the car which can be "low", "med", or "high".
We’ve imported a dataset of cars behind the scenes and created a decision tree using that data. In this lesson, you’ll learn how to build that tree yourself, but for now, let’s see what the tree can do!

Instructions

Create a variable named car. We’re going to be feeding car into tree, the decision tree we’ve made behind the scenes. car should be a list of six items — one value for each feature.

Try to make is a car that you think would have the label vgood and we’ll see if the decision tree agrees with you!

Make sure your features are in the order listed above.

Here’s the start of the definition of a car.

car = ["low", "med", "3", ____, ____, ____]
2.
Call classify() using car and tree as parameters. Print the result.

Did the decision tree classify car as you expected?

Feel free to change the features of car to see how tree reacts.

'''
'''
DECISION TREES
Gini Impurity
Consider the two trees below. Which tree would be more useful as a model that tries to predict whether someone would get an A in a class?

A tree where the leaf nodes have different types of classificationA tree where the leaf nodes have only one type of classification
Let’s say you use the top tree. You’ll end up at a leaf node where the label is up for debate. The training data has labels from both classes! If you use the bottom tree, you’ll end up at a leaf where there’s only one type of label. There’s no debate at all! We’d be much more confident about our classification if we used the bottom tree.

This idea can be quantified by calculating the Gini impurity of a set of data points. To find the Gini impurity, start at 1 and subtract the squared percentage of each label in the set. For example, if a data set had three items of class A and one item of class B, the Gini impurity of the set would be

1 - \bigg(\frac{3}{4}\bigg)^2 - \bigg(\frac{1}{4}\bigg)^2 = 0.3751−( 
4
3
​	 ) 
2
 −( 
4
1
​	 ) 
2
 =0.375
If a data set has only one class, you’d end up with a Gini impurity of 0. The lower the impurity, the better the decision tree!

'''
'''
Instructions
1.
Let’s find the Gini impurity of the set of labels we’ve given you.

Let’s start by creating a variable named impurity and set it to 1.

2.
We now want to count up how many times every unique label is in the dataset. Python’s Counter object can do this quickly.

For example, given the following code:

lst = ["A", "A", "B"]
counts = Counter(lst)
would result in counts storing this object:

Counter({"A": 2, "B": 1})
Create a counter object of labels‘ items named label_counts.

Print out label_counts to see if it matches what you expect.

Fill in labels as the parameter:

label_counts = Counter(___)
3.
Let’s find the probability of each label given the dataset. Loop through each label in label_counts.

Inside the for loop, create a variable named probability_of_label. Set it equal to the label count divided by the total number of labels in the dataset.

For every label, the count associated with that label can be found at label_counts[label].

We can find the total number of labels in the dataset with len(labels).

Your for loop should look something like this:

for label in label_counts:
  probability_of_label = ____/____
4.
We now want to take probability_of_label, square it, and subtract it from impurity.

Inside the for loop, subtract probability_of_label squared from impurity.

In Python, you can square x by using x ** 2.

You can use -= to subtract from impurity:

impurity -= _____
5.
Outside of the for loop, print impurity.

Test out some of the other labels that we’ve given you by uncommenting them. Which one do you expect to have the lowest impurity?

In the next exercise, we’ll put all of your code into a function. If you want a challenge, try creating the function yourself! Ours is named gini(), takes labels as a parameter, and returns impurity.

The dataset that has only one type of label should have an impurity of 0.'''

from collections import Counter

labels = ["unacc", "unacc", "acc", "acc", "good", "good"]
#labels = ["unacc","unacc","unacc", "good", "vgood", "vgood"]
#labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc"]

impurity = 1

label_counts = Counter(labels)

print(label_counts)

for lable in label_counts:
  probability_of_label = label_counts[lable] / len(labels)
  print(lable,probability_of_label)
  impurity += -1*(probability_of_label**2)
  
print(impurity)

'''
DECISION TREES
Information Gain
We know that we want to end up with leaves with a low Gini Impurity, but we still need to figure out which features to split on in order to achieve this. For example, is it better if we split our dataset of students based on how much sleep they got or how much time they spent studying?

To answer this question, we can calculate the information gain of splitting the data on a certain feature. Information gain measures difference in the impurity of the data before and after the split. For example, let’s say you had a dataset with an impurity of 0.5. After splitting the data based on a feature, you end up with three groups with impurities 0, 0.375, and 0. The information gain of splitting the data in that way is 0.5 - 0 - 0.375 - 0 = 0.125.


Not bad! By splitting the data in that way, we’ve gained some information about how the data is structured — the datasets after the split are purer than they were before the split. The higher the information gain the better — if information gain is 0, then splitting the data on that feature was useless! Unfortunately, right now it’s possible for information gain to be negative. In the next exercise, we’ll calculate weighted information gain to fix that problem.
'''
'''
Instructions
1.
We’ve given you a set of labels named unsplit_labels and two different ways of splitting those labels into smaller subsets. Let’s calculate the information gain of splitting the labels in this way.

At the bottom of your code, begin by creating a variable named info_gain. info_gain should start at the Gini impurity of the unsplit_labels.

Call the gini() function we’ve given you with unsplit_labels as a parameter. Store the result in info_gain.

2.
We now want to subtract the impurity of each subset in split_labels_1 from info_gain.

Loop through every subset in split_labels_1. We want to change the value of info_gain.

For every subset, calculate the Gini impurity and subtract it from info_gain.

Your for loop might look something like this:

for subset in split_labels_1:
  info_gain -= _______
3.
Outside of your loop, print info_gain.

We’ve given you a second way to split the data. Instead of looping through the subsets in split_labels_1, loop through the subsets in split_labels_2.

Which split resulted in more information gain?

Once again, in the next exercise, we’ll put the code you wrote into a function named information_gain that takes unsplit_labels and split_labels as parameters.

The second method of splitting the data should have slightly more information gain.'''

from collections import Counter

unsplit_labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good", "vgood", "vgood", "vgood"]

split_labels_1 = [
  ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "vgood"], 
  [ "good", "good"], 
  ["vgood", "vgood"]
]

split_labels_2 = [
  ["unacc", "unacc", "unacc", "unacc","unacc", "unacc", "good", "good", "good", "good"], 
  ["vgood", "vgood", "vgood"]
]

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

info_gain = gini(unsplit_labels)

print(info_gain)
for subset in split_labels_1:
  print(gini(subset))
  info_gain -= gini(subset)
print(info_gain)

info_gain2 = gini(unsplit_labels)

print(info_gain2)
for subset in split_labels_2:
  print(gini(subset))
  info_gain2 -= gini(subset)
print(info_gain2)

'''
DECISION TREES
Weighted Information Gain
We’re not quite done calculating the information gain of a set of objects. The sizes of the subset that get created after the split are important too! For example, the image below shows two sets with the same impurity. Which set would you rather have in your decision tree?


Both of these sets are perfectly pure, but the purity of the second set is much more meaningful. Because there are so many items in the second set, we can be confident that whatever we did to produce this set wasn’t an accident.

It might be helpful to think about the inverse as well. Consider these two sets with the same impurity:


Both of these sets are completely impure. However, that impurity is much more meaningful in the set with more instances. We know that we are going to have to do a lot more work in order to completely separate the two classes. Meanwhile, the impurity of the set with two items isn’t as important. We know that we’ll only need to split the set one more time in order to make two pure sets.

Let’s modify the formula for information gain to reflect the fact that the size of the set is relevant. Instead of simply subtracting the impurity of each set, we’ll subtract the weighted impurity of each of the split sets. If the data before the split contained 20 items and one of the resulting splits contained 2 items, then the weighted impurity of that subset would be 2/20 * impurity. We’re lowering the importance of the impurity of sets with few elements.


Now that we can calculate the information gain using weighted impurity, let’s do that for every possible feature. If we do this, we can find the best feature to split the data on.'''

'''
Instructions
1.
Let’s update the information_gain function to make it calculate weighted information gain.

When subtracting the impurity of a subset from info_gain, first multiply the impurity by the correct percentage.

The percentage should be the number of labels in the subset, len(subset), divided by the number of labels before the split, len(starting_labels).

Multiply gini(subset) by len(subset)/len(starting_labels).

2.
We’ve given you a split() function along with ten cars and the car_labels associated with those cars.

After your information_gain() function, call split() using cars, car_labels and 3 as a parameter. This will split the data based on the third index (That feature was the number of people the car could hold).

split() returns two lists. Create two variables named split_data and split_labels and set them equal to the result of the split function.

We’ll explore what these variables contain in a second!

In Python, functions can return more than one value. When this happens, you can do something like this:

a, b = function_that_returns_two_things()
Do this with your split() function.

3.
Take a look at what these variables are. Begin by printing split_data. It’s kind of hard to tell what’s going on there! There are so many lists of lists!

Try printing the length of split_data. What do you think this is telling you?

Also try printing split_data[0]. What do you notice about the items at index 3 of all these lists? (Remember, when we called split, we used 3 as the split index).

Try printing split_data[1]. What do you notice about the items at index 3 of these lists?

len(split_data) is telling you how many subsets the original data set was split into. In this case, when we split the dataset using index 3, we split it into 3 subsets.

When you print each subset, you’ll see that the value at index 3 of each car in the subset is the same. We’ve basically created three subsets — cars could hold "2" people, cars that could hold "4" people, and cars that could hold "more" people.

4.
We now know that split_data contains the cars split into different subsets. split_labels contains the labels of those cars split into different subsets.

Use those split labels to find the information gain of splitting on index 3! Remember, the information_gain() function takes a list of the labels before the split (car_labels), and a list of the subsets of labels after the split (split_labels).

Call this function and print the result! How did we do when we split the function on index 3?

Print the results of information_gain(car_labels, split_labels)

5.
We found the information gain when splitting on feature 3. Let’s do the same for every possible feature.

Loop through all of the features of our data to find the best one to split on! Each car has six features, so we want to loop through the indices 0 through 5.

Inside your for loop, call split() using the unsplit data, the unsplit labels, and the index that you’re looping through.

Call information_gain() using the resulting split labels and print the results. Which feature produces the most information gain?

Your for loop might look something like this:

for i in range(0, 6):
  split_data, split_labels = split(____, ____, i)
  print(information_gain(____, ____)
  '''
from collections import Counter

cars = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    # Multiply gini(subset) by the correct percentage below
    info_gain -= gini(subset)*len(subset)/len(starting_labels)
  return info_gain

split_data , split_labels = split(cars, car_labels, 3) 
#print(len(split_data))

#print(split_data[0])

#print(split_data[1])

#print(split_labe   `ls)

print(information_gain(car_labe      QQQQ   qlswrsdsswqawdsxawqewsadsaxszals , split_labels))

for i in range(len(cars[0])):
  split_data_a , split_labels_a = split(cars, car_labels, i)
    #print(i)  
  
'''
DECISION TREES
Recursive Tree Building
Now that we can find the best feature to split the dataset, we can repeat this process again and again to create the full tree. This is a recursive algorithm! We start with every data point from the training set, find the best feature to split the data, split the data based on that feature, and then recursively repeat the process again on each subset that was created from the split.

We’ll stop the recursion when we can no longer find a feature that results in any information gain. In other words, we want to create a leaf of the tree when we can’t find a way to split the data that makes purer subsets.

The leaf should keep track of the classes of the data points from the training set that ended up in the leaf. In our implementation, we’ll use a Counter object to keep track of the counts of labels.

We’ll use these counts to make predictions about new data that we give the tree.
'''
#script.py-------------------------------

from tree import *

car_data = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


def build_tree(data, labels):
  best_feature, best_gain = find_best_split(data, labels)
  
  if best_gain == 0:
    return Counter(labels)
  
  data_subsets, label_subsets = split(data, labels, best_feature)

  branches = []

  for i in range(len(data_subsets)):
    branch = build_tree(data_subsets[i], label_subsets[i])
    branches.append(branch)

  return branches

tree = build_tree(car_data, car_labels)
print_tree(tree)

#tree.py-------------------------------
from collections import Counter

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset)/len(starting_labels)
  return info_gain  

class Leaf:
    def __init__(self, labels):
        self.predictions = Counter(labels)

class Internal_Node:
    def __init__(self,
                 feature,
                 branches):
        self.feature = feature
        self.branches = branches

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1:"Price of maintenance", 2:"Number of doors", 3:"Person Capacity", 4:"Size of luggage boot", 5:"Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Counter):
        print (spacing + str(node))
        return

    # Print the question at this node
    print (spacing + "Splitting")

    # Call this function recursively on the true branch
    for i in range(len(node)):
        print (spacing + '--> Branch ' + str(i)+':')
        print_tree(node[i], spacing + "  ")

'''
DECISION TREES
Classifying New Data
We can finally use our tree as a classifier! Given a new data point, we start at the top of the tree and follow the path of the tree until we hit a leaf. Once we get to a leaf, we’ll use the classes of the points from the training set to make a classification.

We’ve slightly changed the way our build_tree() function works. Instead of returning a list of branches or a Counter object, the build_tree() function now returns a Leaf object or an Internal_Node object. We’ll explain how to use these objects in the instructions!

Let’s write a function that will use our tree to classify new points!
'''
'''
Instructions
1.
We’ve created a tree named tree using a lot of car data. Use the print_tree() function with tree as a parameter to see it.

Notice that the tree now knows which feature was used to split the data. This new information is contained in the Leaf and Internal_Node classes. This will come in handy when we write our classify function!

Comment out printing the tree once you get a sense of how large it is!

Call print_tree(tree).

2.
Let’s start writing the classify() function. classify() should take a datapoint and a tree as a parameter.

The first thing classify should do is check to see if we’re at a leaf.

Check to see if tree is a Leaf by using the isinstance() function.

For example, isinstance(a, list) will be True if a is a list. You should check if tree is a Leaf.

If we’ve found a Leaf, that means we want to return the label with the highest count. The label counts are stored in tree.labels.

You could find the label with the largest count by using a for loop, or by using this rather complicated line of code:

return max(tree.labels.items(), key=operator.itemgetter(1))[0]
Your if statement should look like this:

if isinstance(tree, Counter):
Then return the label with the highest count.

3.
If we’re not at a leaf, we want to find the branch that corresponds to our data point. For example, if we’re splitting on index 0 and our data point is ['med', 'low', '4', '2', 'big', 'low'], we want to find the branch that contains all of the points with med at index 0.

To start, let’s find datapoint‘s value of the feature we’re looking for. If datapoint were the example above, and the feature we’re interested is 0, this would be med.

Outside the if statement, create a variable named value and set it equal to datapoint[tree.feature]. tree.feature contains the index of the feature that we’re splitting on, so datapoint[tree.feature] is the value at that index.

To help us check your code, return value.

4.
Start by deleting return value.

Let’s now loop through all of the branches in the tree to find the one that has all the data points with value at the correct index.

Your loop should look like this:

for branch in tree.branches:
Next, inside the loop, check to see if branch.value is equal to value. If it is, we’ve found the branch that we’re looking for! We want to now recursively call classify() on that branch:

return classify(datapoint, branch)
We know that one of these branches will be the one we’re looking for, so we know that this return statement will happen once.

Your final function should look something like this. Fill in the if statement near the bottom of the function.

def classify(datapoint, tree):
  if isinstance(tree, Leaf):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]
  answer = datapoint[tree.feature]
  for branch in tree.branches:
    if ____ == ____:
      return classify(datapoint, branch)
5.
Finally, outside of your function, call classify() using test_point and tree as parameters. Print the results. You should see a classification for this new point.
'''
#--------------------- script.py-------------------------------
from tree import *
import operator

test_point = ['vhigh', 'low', '3', '4', 'med', 'med']

#print_tree(tree)

def classify(datapoint, tree):
  if isinstance(tree, Leaf) == True:
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]
  
  value = datapoint[tree.feature]
  
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)

test = classify(test_point, tree)
print(test)

#---------------------tree.py-------------------------------

from collections import Counter

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset)/len(starting_labels)
  return info_gain  

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value

class Internal_Node:
    def __init__(self,
                 feature,
                 branches,
                 value):
        self.feature = feature
        self.branches = branches
        self.value = value

        
def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def build_tree(data, labels, value = ""):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain == 0:
    return Leaf(Counter(labels), value)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
    branches.append(branch)
  return Internal_Node(best_feature, branches, value)
        
        
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1:"Price of maintenance", 2:"Number of doors", 3:"Person Capacity", 4:"Size of luggage boot", 5:"Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + str(node.labels))
        return

    # Print the question at this node
    print (spacing + "Splitting on " + question_dict[node.feature])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print (spacing + '--> Branch ' + node.branches[i].value+':')
        print_tree(node.branches[i], spacing + "  ")
        
cars = [['high', 'low', '5more', '4', 'big', 'high'], ['high', 'vhigh', '4', 'more', 'med', 'med'], ['high', 'med', '4', '2', 'med', 'high'], ['low', 'vhigh', '4', '2', 'med', 'med'], ['vhigh', 'low', '5more', '2', 'small', 'low'], ['vhigh', 'high', '5more', '4', 'small', 'low'], ['med', 'med', '2', 'more', 'small', 'med'], ['med', 'med', '2', 'more', 'small', 'high'], ['med', 'low', '2', '4', 'med', 'high'], ['high', 'vhigh', '4', '4', 'small', 'low'], ['vhigh', 'low', '5more', 'more', 'med', 'med'], ['vhigh', 'vhigh', '3', 'more', 'big', 'med'], ['high', 'med', '4', '4', 'small', 'high'], ['med', 'med', '5more', 'more', 'med', 'high'], ['low', 'vhigh', '4', 'more', 'small', 'med'], ['high', 'med', '4', '2', 'big', 'low'], ['vhigh', 'vhigh', '5more', '2', 'med', 'med'], ['low', 'vhigh', '2', '2', 'big', 'high'], ['med', 'high', '2', '4', 'med', 'low'], ['vhigh', 'med', '2', '4', 'med', 'low'], ['low', 'high', '3', '4', 'med', 'high'], ['med', 'low', '5more', '4', 'med', 'high'], ['high', 'high', '3', '2', 'big', 'low'], ['low', 'vhigh', '2', '4', 'big', 'low'], ['high', 'low', '4', '2', 'small', 'high'], ['vhigh', 'med', '5more', 'more', 'big', 'high'], ['vhigh', 'med', '5more', '2', 'small', 'low'], ['vhigh', 'med', '5more', '2', 'big', 'low'], ['med', 'vhigh', '4', 'more', 'med', 'high'], ['low', 'high', '2', 'more', 'small', 'low'], ['med', 'vhigh', '2', 'more', 'med', 'high'], ['low', 'vhigh', '5more', '2', 'small', 'high'], ['med', 'med', '4', '2', 'med', 'low'], ['med', 'low', '4', '4', 'big', 'high'], ['high', 'vhigh', '3', 'more', 'big', 'high'], ['high', 'high', '4', 'more', 'med', 'low'], ['vhigh', 'high', '5more', '2', 'small', 'low'], ['high', 'high', '3', '4', 'med', 'med'], ['high', 'low', '5more', '4', 'small', 'low'], ['low', 'vhigh', '5more', '2', 'med', 'high'], ['med', 'high', '3', '4', 'big', 'high'], ['med', 'low', '4', '2', 'big', 'low'], ['med', 'high', '2', '2', 'med', 'low'], ['low', 'vhigh', '3', 'more', 'med', 'high'], ['vhigh', 'low', '3', '4', 'big', 'low'], ['high', 'high', '2', 'more', 'big', 'med'], ['low', 'vhigh', '3', '2', 'med', 'low'], ['low', 'med', '5more', 'more', 'med', 'med'], ['high', 'med', '3', 'more', 'small', 'high'], ['high', 'med', '5more', 'more', 'big', 'high'], ['med', 'vhigh', '2', '2', 'med', 'high'], ['low', 'high', '3', '2', 'big', 'high'], ['vhigh', 'high', '3', 'more', 'big', 'low'], ['vhigh', 'med', '2', '2', 'big', 'low'], ['high', 'vhigh', '4', '4', 'big', 'med'], ['low', 'med', '4', 'more', 'med', 'low'], ['high', 'high', '3', '4', 'small', 'med'], ['med', 'low', '3', '4', 'small', 'high'], ['vhigh', 'vhigh', '5more', '2', 'big', 'low'], ['vhigh', 'med', '3', 'more', 'med', 'high'], ['high', 'low', '2', '4', 'med', 'high'], ['low', 'high', '4', '2', 'small', 'med'], ['high', 'med', '2', '4', 'med', 'high'], ['low', 'med', '3', '4', 'big', 'high'], ['high', 'low', '4', 'more', 'big', 'low'], ['high', 'low', '5more', '2', 'med', 'low'], ['low', 'high', '2', 'more', 'small', 'high'], ['med', 'high', '2', '2', 'big', 'high'], ['med', 'high', '3', '4', 'small', 'high'], ['high', 'high', '3', '4', 'med', 'high'], ['vhigh', 'med', '5more', '4', 'med', 'high'], ['vhigh', 'med', '4', '4', 'small', 'high'], ['high', 'low', '4', 'more', 'big', 'med'], ['high', 'med', '2', 'more', 'big', 'low'], ['low', 'vhigh', '3', '2', 'med', 'high'], ['vhigh', 'vhigh', '5more', '2', 'big', 'high'], ['low', 'high', '4', '4', 'med', 'high'], ['high', 'low', '4', 'more', 'big', 'high'], ['med', 'vhigh', '5more', '2', 'small', 'low'], ['high', 'med', '4', '4', 'med', 'low'], ['med', 'med', '3', '2', 'small', 'med'], ['vhigh', 'low', '3', 'more', 'med', 'high'], ['high', 'low', '2', '2', 'small', 'med'], ['med', 'med', '5more', 'more', 'big', 'high'], ['high', 'vhigh', '5more', '4', 'small', 'high'], ['med', 'med', '5more', 'more', 'small', 'high'], ['high', 'low', '4', '2', 'med', 'high'], ['low', 'high', '4', '2', 'big', 'high'], ['low', 'vhigh', '2', '4', 'med', 'med'], ['low', 'med', '5more', '2', 'big', 'high'], ['vhigh', 'vhigh', '5more', '4', 'big', 'low'], ['vhigh', 'med', '4', '2', 'small', 'high'], ['med', 'high', '4', '2', 'med', 'med'], ['high', 'vhigh', '3', '4', 'small', 'low'], ['low', 'low', '5more', 'more', 'big', 'low'], ['vhigh', 'high', '3', '2', 'big', 'med'], ['high', 'high', '3', '4', 'big', 'med'], ['low', 'high', '5more', '4', 'small', 'med'], ['vhigh', 'med', '4', '4', 'med', 'high'], ['med', 'vhigh', '4', '2', 'small', 'med'], ['med', 'med', '3', '2', 'big', 'high'], ['low', 'high', '4', '2', 'small', 'high'], ['vhigh', 'med', '2', '4', 'med', 'high'], ['high', 'med', '2', '2', 'small', 'med'], ['vhigh', 'low', '4', '2', 'big', 'med'], ['low', 'vhigh', '4', 'more', 'big', 'high'], ['low', 'high', '2', '2', 'big', 'low'], ['vhigh', 'low', '5more', '4', 'big', 'med'], ['med', 'vhigh', '5more', '4', 'med', 'med'], ['med', 'med', '2', '2', 'small', 'low'], ['med', 'med', '2', '2', 'med', 'med'], ['low', 'med', '3', 'more', 'med', 'low'], ['med', 'high', '4', '4', 'big', 'high'], ['vhigh', 'vhigh', '2', '2', 'big', 'med'], ['high', 'med', '5more', '2', 'small', 'high'], ['low', 'high', '5more', '2', 'small', 'high'], ['low', 'med', '2', 'more', 'small', 'low'], ['low', 'high', '5more', '2', 'med', 'med'], ['high', 'med', '5more', '4', 'big', 'low'], ['vhigh', 'low', '3', 'more', 'big', 'high'], ['med', 'vhigh', '5more', 'more', 'med', 'low'], ['vhigh', 'med', '5more', '2', 'small', 'high'], ['low', 'vhigh', '2', '4', 'med', 'high'], ['med', 'low', '2', 'more', 'med', 'low'], ['high', 'low', '3', '2', 'med', 'high'], ['low', 'med', '4', 'more', 'big', 'low'], ['low', 'vhigh', '2', '4', 'big', 'high'], ['low', 'med', '3', '4', 'small', 'low'], ['low', 'med', '4', 'more', 'small', 'high'], ['med', 'low', '3', 'more', 'med', 'med'], ['high', 'med', '2', 'more', 'small', 'low'], ['med', 'vhigh', '4', 'more', 'med', 'low'], ['med', 'vhigh', '5more', '2', 'med', 'high'], ['med', 'vhigh', '3', '2', 'big', 'low'], ['vhigh', 'high', '5more', '2', 'big', 'high'], ['low', 'high', '3', '4', 'big', 'med'], ['high', 'high', '2', '2', 'med', 'low'], ['high', 'vhigh', '5more', '2', 'med', 'low'], ['vhigh', 'high', '5more', 'more', 'small', 'high'], ['high', 'low', '5more', 'more', 'big', 'low'], ['vhigh', 'low', '2', '4', 'med', 'high'], ['vhigh', 'vhigh', '3', 'more', 'small', 'high'], ['high', 'low', '2', 'more', 'med', 'low'], ['high', 'high', '3', 'more', 'small', 'med'], ['low', 'vhigh', '2', '2', 'big', 'low'], ['low', 'vhigh', '5more', '4', 'med', 'low'], ['med', 'vhigh', '4', '4', 'med', 'high'], ['vhigh', 'low', '2', 'more', 'small', 'med'], ['low', 'low', '5more', '4', 'small', 'low'], ['high', 'vhigh', '4', '4', 'med', 'med'], ['low', 'vhigh', '2', 'more', 'small', 'med'], ['high', 'high', '4', '4', 'small', 'med'], ['low', 'low', '4', '4', 'small', 'low'], ['high', 'high', '3', 'more', 'med', 'med'], ['high', 'med', '3', 'more', 'small', 'low'], ['med', 'vhigh', '3', '4', 'small', 'med'], ['high', 'high', '4', '2', 'med', 'med'], ['med', 'med', '3', '2', 'med', 'med'], ['vhigh', 'med', '5more', '2', 'big', 'high'], ['low', 'high', '3', '4', 'med', 'low'], ['low', 'low', '4', '4', 'big', 'med'], ['low', 'high', '2', 'more', 'med', 'high'], ['high', 'low', '4', '4', 'med', 'med'], ['low', 'vhigh', '2', '2', 'big', 'med'], ['high', 'vhigh', '4', '2', 'big', 'low'], ['vhigh', 'high', '4', '4', 'med', 'med'], ['vhigh', 'high', '2', 'more', 'small', 'high'], ['vhigh', 'vhigh', '2', 'more', 'med', 'med'], ['vhigh', 'high', '4', 'more', 'small', 'med'], ['high', 'high', '3', '2', 'med', 'high'], ['high', 'high', '4', 'more', 'big', 'low'], ['low', 'med', '4', '2', 'small', 'med'], ['med', 'vhigh', '3', 'more', 'big', 'low'], ['low', 'vhigh', '2', 'more', 'big', 'high'], ['high', 'high', '4', '2', 'med', 'high'], ['low', 'med', '4', '2', 'med', 'med'], ['vhigh', 'low', '5more', '4', 'big', 'low'], ['high', 'vhigh', '4', '4', 'small', 'high'], ['med', 'med', '2', '2', 'big', 'high'], ['high', 'med', '3', 'more', 'med', 'low'], ['low', 'med', '3', '2', 'small', 'high'], ['vhigh', 'med', '4', 'more', 'small', 'low'], ['med', 'vhigh', '3', '4', 'big', 'med'], ['low', 'low', '2', '2', 'med', 'low'], ['med', 'high', '4', '2', 'small', 'med'], ['high', 'high', '4', '4', 'med', 'high'], ['med', 'low', '5more', 'more', 'big', 'low'], ['vhigh', 'med', '2', '4', 'small', 'low'], ['vhigh', 'low', '3', '4', 'big', 'med'], ['vhigh', 'vhigh', '5more', '4', 'big', 'high'], ['med', 'med', '5more', '4', 'med', 'high'], ['vhigh', 'high', '2', 'more', 'small', 'med'], ['med', 'med', '4', 'more', 'small', 'med'], ['high', 'low', '2', '4', 'big', 'med'], ['high', 'low', '2', 'more', 'big', 'high'], ['high', 'high', '2', '4', 'small', 'high'], ['high', 'high', '4', '2', 'big', 'med'], ['low', 'vhigh', '5more', '2', 'small', 'med'], ['high', 'med', '4', '2', 'small', 'low'], ['low', 'med', '4', '4', 'med', 'high'], ['vhigh', 'high', '5more', '4', 'med', 'low'], ['high', 'med', '5more', '4', 'med', 'high'], ['vhigh', 'med', '3', 'more', 'med', 'med'], ['med', 'low', '3', '4', 'med', 'low'], ['vhigh', 'med', '3', '2', 'big', 'med'], ['vhigh', 'low', '2', '4', 'small', 'high'], ['high', 'high', '3', 'more', 'big', 'med'], ['high', 'med', '3', 'more', 'med', 'med'], ['vhigh', 'high', '5more', 'more', 'big', 'med'], ['vhigh', 'low', '4', 'more', 'small', 'high'], ['med', 'med', '3', '2', 'small', 'high'], ['vhigh', 'low', '4', '4', 'small', 'low'], ['med', 'high', '2', '4', 'small', 'low'], ['high', 'high', '2', 'more', 'med', 'med'], ['vhigh', 'low', '4', 'more', 'small', 'med'], ['med', 'low', '3', '4', 'big', 'med'], ['med', 'high', '2', '2', 'med', 'high'], ['low', 'vhigh', '3', 'more', 'big', 'med'], ['vhigh', 'high', '2', '4', 'small', 'med'], ['med', 'low', '3', '2', 'med', 'med'], ['high', 'low', '5more', '2', 'small', 'med'], ['high', 'vhigh', '3', '2', 'med', 'low'], ['vhigh', 'low', '2', '2', 'big', 'med'], ['high', 'vhigh', '3', 'more', 'small', 'high'], ['vhigh', 'low', '3', '4', 'med', 'med'], ['high', 'vhigh', '4', '4', 'small', 'med'], ['high', 'low', '5more', '4', 'med', 'high'], ['high', 'low', '4', '2', 'med', 'low'], ['low', 'med', '5more', '4', 'small', 'low'], ['vhigh', 'vhigh', '3', '2', 'big', 'low'], ['vhigh', 'low', '4', '4', 'big', 'high'], ['med', 'low', '5more', '2', 'small', 'med'], ['med', 'vhigh', '5more', 'more', 'small', 'high'], ['med', 'med', '2', 'more', 'big', 'med'], ['vhigh', 'high', '2', '2', 'small', 'low'], ['vhigh', 'vhigh', '5more', '2', 'med', 'high'], ['med', 'high', '3', 'more', 'small', 'med'], ['low', 'high', '2', 'more', 'med', 'med'], ['vhigh', 'med', '3', 'more', 'med', 'low'], ['vhigh', 'med', '3', '4', 'big', 'med'], ['low', 'low', '4', '4', 'big', 'high'], ['high', 'high', '3', '4', 'big', 'high'], ['med', 'high', '5more', '4', 'big', 'low'], ['vhigh', 'high', '3', '4', 'small', 'low'], ['high', 'vhigh', '3', 'more', 'small', 'med'], ['med', 'low', '4', '4', 'big', 'low'], ['low', 'vhigh', '5more', '2', 'big', 'high'], ['med', 'high', '4', '2', 'med', 'high'], ['med', 'med', '4', '2', 'big', 'low'], ['vhigh', 'low', '4', '2', 'med', 'high'], ['vhigh', 'vhigh', '4', 'more', 'big', 'high'], ['vhigh', 'vhigh', '3', '2', 'small', 'low'], ['low', 'vhigh', '5more', '4', 'small', 'low'], ['med', 'med', '2', 'more', 'small', 'low'], ['high', 'med', '4', 'more', 'med', 'low'], ['vhigh', 'low', '4', '4', 'big', 'med'], ['vhigh', 'low', '2', '4', 'big', 'low'], ['med', 'high', '3', '2', 'med', 'low'], ['low', 'vhigh', '5more', '4', 'small', 'med'], ['low', 'med', '3', '2', 'big', 'med'], ['vhigh', 'high', '3', 'more', 'big', 'med'], ['vhigh', 'med', '5more', '4', 'big', 'low'], ['med', 'low', '5more', '2', 'med', 'high'], ['high', 'high', '2', 'more', 'small', 'low'], ['low', 'vhigh', '3', '2', 'big', 'low'], ['vhigh', 'vhigh', '3', '4', 'big', 'high'], ['high', 'med', '5more', 'more', 'big', 'low'], ['vhigh', 'high', '3', '2', 'small', 'high'], ['med', 'high', '5more', '2', 'big', 'low'], ['med', 'low', '5more', 'more', 'small', 'med'], ['low', 'med', '3', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'small', 'high'], ['low', 'vhigh', '3', '2', 'small', 'low'], ['med', 'low', '3', '2', 'small', 'med'], ['vhigh', 'low', '2', '4', 'med', 'med'], ['low', 'low', '5more', '2', 'small', 'low'], ['high', 'vhigh', '2', '4', 'big', 'high'], ['low', 'vhigh', '4', 'more', 'med', 'med'], ['vhigh', 'med', '3', '4', 'small', 'high'], ['high', 'low', '5more', 'more', 'big', 'high'], ['high', 'high', '4', 'more', 'small', 'med'], ['vhigh', 'vhigh', '2', 'more', 'small', 'med'], ['vhigh', 'high', '5more', 'more', 'med', 'low'], ['med', 'med', '4', 'more', 'big', 'low'], ['vhigh', 'med', '2', '2', 'big', 'med'], ['low', 'med', '4', '4', 'small', 'med'], ['med', 'vhigh', '3', '2', 'small', 'med'], ['vhigh', 'high', '4', '4', 'small', 'low'], ['med', 'high', '2', '4', 'med', 'med'], ['low', 'low', '2', 'more', 'small', 'low'], ['high', 'med', '2', '4', 'small', 'med'], ['med', 'vhigh', '3', '2', 'med', 'med'], ['high', 'med', '3', '2', 'med', 'med'], ['low', 'low', '2', '4', 'med', 'high'], ['med', 'med', '3', '4', 'small', 'med'], ['vhigh', 'low', '2', '4', 'small', 'med'], ['vhigh', 'high', '4', 'more', 'small', 'low'], ['vhigh', 'low', '5more', '2', 'med', 'med'], ['med', 'low', '2', '2', 'med', 'high'], ['med', 'high', '2', '4', 'small', 'high'], ['vhigh', 'vhigh', '4', '2', 'med', 'med'], ['vhigh', 'vhigh', '4', 'more', 'med', 'high'], ['high', 'med', '4', 'more', 'med', 'high'], ['vhigh', 'high', '3', '4', 'big', 'high'], ['low', 'vhigh', '5more', 'more', 'med', 'low'], ['high', 'vhigh', '3', '4', 'small', 'med'], ['vhigh', 'high', '4', 'more', 'med', 'low'], ['med', 'low', '5more', 'more', 'small', 'high'], ['low', 'low', '4', '4', 'med', 'med'], ['vhigh', 'vhigh', '4', '4', 'big', 'med'], ['high', 'high', '2', 'more', 'big', 'high'], ['med', 'vhigh', '2', '2', 'small', 'med'], ['vhigh', 'vhigh', '3', '4', 'small', 'med'], ['low', 'vhigh', '3', '2', 'big', 'med'], ['low', 'vhigh', '2', '4', 'small', 'med'], ['high', 'med', '2', '2', 'big', 'low'], ['high', 'med', '3', 'more', 'small', 'med'], ['low', 'low', '5more', '2', 'big', 'high'], ['low', 'vhigh', '2', '2', 'med', 'low'], ['vhigh', 'low', '5more', '4', 'med', 'low'], ['low', 'low', '5more', '4', 'big', 'low'], ['vhigh', 'vhigh', '4', '4', 'med', 'med'], ['low', 'low', '2', 'more', 'med', 'low'], ['med', 'med', '4', '2', 'med', 'med'], ['low', 'high', '4', '4', 'med', 'med'], ['vhigh', 'med', '3', '4', 'big', 'low'], ['low', 'high', '5more', 'more', 'small', 'high'], ['high', 'vhigh', '2', '2', 'big', 'med'], ['high', 'high', '4', '4', 'big', 'med'], ['high', 'med', '5more', '4', 'small', 'high'], ['low', 'high', '5more', '2', 'big', 'med'], ['med', 'low', '2', 'more', 'small', 'low'], ['vhigh', 'vhigh', '2', '4', 'med', 'high'], ['high', 'high', '5more', '2', 'med', 'med'], ['vhigh', 'vhigh', '4', 'more', 'big', 'med'], ['vhigh', 'vhigh', '5more', '4', 'big', 'med'], ['high', 'med', '4', 'more', 'big', 'med'], ['low', 'med', '5more', 'more', 'med', 'low'], ['vhigh', 'low', '2', 'more', 'big', 'high'], ['med', 'med', '4', 'more', 'small', 'low'], ['med', 'med', '3', '4', 'med', 'med'], ['med', 'low', '5more', 'more', 'med', 'med'], ['low', 'high', '5more', '4', 'big', 'low'], ['high', 'med', '2', '2', 'small', 'high'], ['med', 'vhigh', '4', '4', 'big', 'med'], ['low', 'med', '4', '2', 'med', 'high'], ['low', 'vhigh', '3', 'more', 'small', 'high'], ['high', 'low', '2', '4', 'small', 'med'], ['high', 'high', '5more', 'more', 'big', 'low'], ['low', 'vhigh', '5more', '4', 'small', 'high'], ['med', 'med', '3', '2', 'med', 'low'], ['vhigh', 'low', '5more', 'more', 'small', 'low'], ['med', 'med', '2', '2', 'med', 'low'], ['med', 'high', '3', '4', 'med', 'med'], ['low', 'high', '3', '4', 'small', 'low'], ['med', 'vhigh', '3', '4', 'med', 'med'], ['low', 'low', '2', '4', 'big', 'high'], ['low', 'low', '3', 'more', 'big', 'low'], ['vhigh', 'med', '4', '4', 'small', 'med'], ['vhigh', 'vhigh', '2', '4', 'med', 'med'], ['vhigh', 'high', '3', '4', 'small', 'high'], ['high', 'low', '4', 'more', 'med', 'low'], ['low', 'med', '5more', 'more', 'med', 'high'], ['high', 'vhigh', '4', '4', 'med', 'low'], ['vhigh', 'low', '4', 'more', 'big', 'low'], ['med', 'vhigh', '3', 'more', 'med', 'med'], ['low', 'med', '5more', '4', 'med', 'low'], ['vhigh', 'vhigh', '4', 'more', 'med', 'low'], ['vhigh', 'low', '5more', 'more', 'small', 'med'], ['med', 'med', '4', '4', 'small', 'high'], ['low', 'low', '3', '4', 'small', 'high'], ['high', 'low', '2', '4', 'med', 'low'], ['high', 'low', '3', '4', 'med', 'med'], ['vhigh', 'vhigh', '5more', '4', 'small', 'low'], ['high', 'med', '4', 'more', 'small', 'high'], ['vhigh', 'vhigh', '3', '4', 'small', 'high'], ['med', 'high', '3', 'more', 'big', 'low'], ['med', 'low', '5more', '2', 'small', 'low'], ['vhigh', 'med', '2', 'more', 'big', 'med'], ['low', 'med', '2', '4', 'big', 'med'], ['vhigh', 'med', '4', 'more', 'med', 'med'], ['high', 'low', '3', '4', 'small', 'low'], ['low', 'vhigh', '5more', '4', 'big', 'low'], ['med', 'low', '5more', 'more', 'big', 'high'], ['vhigh', 'high', '4', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'high'], ['low', 'low', '5more', '2', 'med', 'high'], ['high', 'low', '4', '2', 'med', 'med'], ['high', 'low', '3', 'more', 'med', 'high'], ['high', 'med', '3', '4', 'med', 'low'], ['med', 'vhigh', '3', '2', 'med', 'low'], ['high', 'med', '5more', '2', 'med', 'med'], ['high', 'low', '4', '2', 'small', 'low'], ['med', 'high', '5more', '4', 'med', 'low'], ['vhigh', 'med', '2', '4', 'big', 'high'], ['low', 'vhigh', '2', '2', 'med', 'high'], ['vhigh', 'med', '5more', 'more', 'med', 'low'], ['med', 'low', '4', '2', 'small', 'high'], ['vhigh', 'high', '2', '2', 'med', 'low'], ['low', 'high', '5more', 'more', 'med', 'low'], ['low', 'low', '2', '4', 'small', 'low'], ['low', 'high', '5more', '4', 'med', 'med'], ['med', 'low', '3', 'more', 'med', 'low'], ['high', 'low', '3', '4', 'small', 'med'], ['high', 'high', '2', '2', 'small', 'high'], ['high', 'low', '3', '4', 'med', 'high'], ['low', 'med', '2', 'more', 'med', 'med'], ['low', 'med', '3', '4', 'med', 'med'], ['med', 'high', '4', 'more', 'small', 'high'], ['high', 'med', '2', 'more', 'small', 'med'], ['low', 'low', '4', 'more', 'med', 'low'], ['med', 'high', '5more', '2', 'med', 'low'], ['high', 'low', '2', 'more', 'med', 'high'], ['high', 'high', '3', '4', 'small', 'low'], ['med', 'vhigh', '5more', '4', 'big', 'med'], ['high', 'low', '4', '4', 'big', 'med'], ['low', 'high', '5more', '2', 'small', 'low'], ['med', 'vhigh', '2', 'more', 'med', 'med'], ['low', 'med', '5more', '4', 'small', 'high'], ['vhigh', 'vhigh', '2', '2', 'small', 'low'], ['vhigh', 'vhigh', '5more', '2', 'small', 'med'], ['low', 'med', '2', '2', 'big', 'med'], ['low', 'low', '5more', '4', 'big', 'med'], ['high', 'low', '5more', 'more', 'big', 'med'], ['low', 'vhigh', '3', '4', 'small', 'med'], ['low', 'low', '2', '2', 'small', 'high'], ['vhigh', 'high', '2', '4', 'med', 'med'], ['med', 'low', '5more', '4', 'big', 'high'], ['med', 'high', '4', '2', 'big', 'high'], ['low', 'low', '4', 'more', 'big', 'high'], ['low', 'low', '5more', 'more', 'big', 'high'], ['med', 'low', '3', '2', 'small', 'high'], ['high', 'med', '4', '4', 'med', 'high'], ['med', 'vhigh', '2', 'more', 'med', 'low'], ['med', 'vhigh', '4', '4', 'big', 'low'], ['med', 'low', '3', '4', 'small', 'low'], ['low', 'med', '4', '4', 'big', 'low'], ['high', 'vhigh', '5more', 'more', 'big', 'high'], ['high', 'med', '2', '2', 'big', 'med'], ['med', 'high', '3', '2', 'big', 'med'], ['high', 'vhigh', '2', '2', 'small', 'low'], ['high', 'high', '5more', '4', 'med', 'high'], ['low', 'med', '4', 'more', 'small', 'low'], ['high', 'high', '4', '2', 'small', 'high'], ['vhigh', 'med', '4', 'more', 'med', 'high'], ['high', 'med', '2', 'more', 'med', 'med'], ['med', 'med', '3', 'more', 'big', 'med'], ['low', 'high', '3', '2', 'big', 'low'], ['high', 'med', '3', '4', 'small', 'low'], ['vhigh', 'low', '4', '4', 'med', 'low'], ['low', 'vhigh', '2', '2', 'small', 'high'], ['med', 'low', '2', '2', 'big', 'med'], ['low', 'low', '3', '2', 'big', 'low'], ['med', 'vhigh', '2', 'more', 'small', 'high'], ['vhigh', 'med', '4', 'more', 'small', 'high'], ['med', 'med', '3', '4', 'big', 'low'], ['med', 'vhigh', '2', '4', 'med', 'low'], ['high', 'high', '4', '4', 'big', 'low'], ['med', 'med', '2', '4', 'med', 'high'], ['vhigh', 'high', '4', '2', 'big', 'high'], ['high', 'low', '2', 'more', 'small', 'med'], ['vhigh', 'high', '4', '2', 'med', 'high'], ['vhigh', 'med', '5more', '4', 'big', 'med'], ['low', 'low', '4', '2', 'small', 'med'], ['vhigh', 'high', '2', '2', 'big', 'low'], ['low', 'med', '4', 'more', 'med', 'high'], ['med', 'high', '5more', 'more', 'big', 'high'], ['low', 'med', '5more', '2', 'small', 'low'], ['vhigh', 'low', '2', '2', 'med', 'low'], ['med', 'vhigh', '2', '4', 'small', 'med'], ['low', 'vhigh', '5more', '4', 'med', 'high'], ['vhigh', 'vhigh', '2', '2', 'small', 'high'], ['low', 'med', '2', '4', 'big', 'high'], ['high', 'vhigh', '3', '2', 'small', 'low'], ['vhigh', 'low', '2', '4', 'small', 'low'], ['med', 'high', '3', '2', 'big', 'low'], ['high', 'vhigh', '4', 'more', 'small', 'high'], ['vhigh', 'high', '4', '2', 'big', 'low'], ['vhigh', 'vhigh', '2', '2', 'med', 'med'], ['high', 'low', '2', '2', 'small', 'low'], ['vhigh', 'low', '3', '4', 'small', 'high'], ['vhigh', 'low', '3', '4', 'med', 'high'], ['med', 'high', '3', 'more', 'med', 'med'], ['med', 'med', '5more', '2', 'small', 'low'], ['med', 'vhigh', '5more', 'more', 'big', 'low'], ['vhigh', 'vhigh', '2', '4', 'big', 'low'], ['high', 'med', '4', 'more', 'small', 'med'], ['low', 'high', '4', '4', 'small', 'high'], ['med', 'low', '4', '4', 'big', 'med'], ['low', 'med', '3', '2', 'big', 'high'], ['high', 'vhigh', '2', 'more', 'med', 'low'], ['low', 'med', '2', 'more', 'small', 'high'], ['low', 'med', '5more', '2', 'big', 'low'], ['high', 'high', '4', '2', 'big', 'low'], ['high', 'med', '3', '2', 'big', 'med'], ['med', 'vhigh', '5more', 'more', 'big', 'high'], ['high', 'high', '5more', 'more', 'med', 'med'], ['vhigh', 'med', '5more', '4', 'small', 'med'], ['low', 'high', '5more', '4', 'med', 'low'], ['high', 'high', '2', '4', 'med', 'high'], ['high', 'med', '5more', 'more', 'small', 'low'], ['high', 'high', '3', 'more', 'big', 'low'], ['high', 'vhigh', '3', '4', 'med', 'high'], ['low', 'high', '4', '4', 'big', 'low'], ['vhigh', 'low', '4', '2', 'med', 'med'], ['vhigh', 'vhigh', '5more', '4', 'small', 'med'], ['low', 'vhigh', '4', '4', 'small', 'low'], ['vhigh', 'low', '3', 'more', 'big', 'low'], ['vhigh', 'high', '4', '2', 'small', 'low'], ['high', 'high', '3', '2', 'small', 'low'], ['vhigh', 'high', '4', '2', 'med', 'low'], ['high', 'low', '2', '2', 'med', 'low'], ['low', 'med', '4', 'more', 'big', 'high'], ['vhigh', 'high', '2', '4', 'small', 'low'], ['low', 'low', '5more', '2', 'small', 'high'], ['low', 'low', '3', '2', 'small', 'high'], ['med', 'med', '2', '2', 'big', 'med'], ['high', 'high', '5more', '4', 'small', 'high'], ['vhigh', 'low', '5more', '2', 'med', 'high'], ['vhigh', 'vhigh', '2', '4', 'small', 'high'], ['med', 'low', '4', '2', 'med', 'low'], ['low', 'high', '4', '4', 'big', 'med'], ['low', 'low', '2', 'more', 'big', 'low'], ['vhigh', 'low', '5more', '4', 'small', 'low'], ['high', 'low', '5more', '4', 'small', 'med'], ['vhigh', 'med', '4', '2', 'small', 'low'], ['high', 'low', '2', '2', 'small', 'high'], ['low', 'vhigh', '3', '4', 'small', 'low'], ['low', 'med', '4', '4', 'small', 'low'], ['low', 'med', '2', '4', 'big', 'low'], ['med', 'med', '2', '4', 'big', 'low'], ['vhigh', 'high', '4', '4', 'big', 'med'], ['vhigh', 'med', '2', 'more', 'med', 'high'], ['low', 'high', '4', 'more', 'big', 'med'], ['low', 'med', '4', '2', 'big', 'low'], ['high', 'med', '2', '2', 'med', 'high'], ['low', 'high', '2', 'more', 'big', 'high'], ['high', 'vhigh', '3', '2', 'med', 'med'], ['vhigh', 'low', '4', 'more', 'med', 'low'], ['low', 'vhigh', '4', '4', 'med', 'low'], ['high', 'low', '5more', '2', 'big', 'high'], ['high', 'vhigh', '5more', 'more', 'small', 'high'], ['high', 'med', '5more', '2', 'small', 'med'], ['med', 'low', '4', 'more', 'big', 'high'], ['med', 'high', '2', 'more', 'big', 'high'], ['high', 'med', '4', 'more', 'big', 'low'], ['low', 'high', '2', '2', 'med', 'high'], ['high', 'vhigh', '5more', '2', 'med', 'med'], ['vhigh', 'high', '2', '2', 'med', 'med'], ['med', 'vhigh', '2', 'more', 'big', 'high'], ['vhigh', 'low', '3', 'more', 'small', 'med'], ['vhigh', 'med', '4', 'more', 'big', 'med'], ['med', 'low', '3', '4', 'med', 'med'], ['med', 'low', '3', '4', 'med', 'high'], ['med', 'med', '5more', '2', 'big', 'med'], ['med', 'med', '3', 'more', 'med', 'low'], ['low', 'low', '4', '4', 'med', 'low'], ['high', 'vhigh', '5more', 'more', 'med', 'low'], ['med', 'high', '4', '4', 'med', 'low'], ['low', 'high', '4', 'more', 'med', 'low'], ['low', 'high', '2', '4', 'small', 'high'], ['vhigh', 'med', '3', '4', 'small', 'med'], ['med', 'med', '4', '4', 'small', 'low'], ['low', 'med', '2', 'more', 'big', 'med'], ['high', 'vhigh', '4', '2', 'small', 'high'], ['low', 'low', '5more', '2', 'med', 'low'], ['med', 'vhigh', '4', '2', 'med', 'low'], ['low', 'med', '4', '4', 'big', 'med'], ['high', 'vhigh', '2', '2', 'med', 'med'], ['vhigh', 'vhigh', '3', '2', 'small', 'med'], ['med', 'med', '5more', '2', 'small', 'high'], ['low', 'high', '2', '2', 'med', 'med'], ['high', 'med', '5more', 'more', 'small', 'med'], ['med', 'vhigh', '2', 'more', 'small', 'med'], ['vhigh', 'med', '4', '2', 'med', 'high'], ['high', 'high', '5more', 'more', 'big', 'med'], ['high', 'vhigh', '2', '2', 'small', 'med'], ['low', 'high', '2', 'more', 'big', 'med'], ['med', 'vhigh', '3', '2', 'small', 'low'], ['high', 'low', '3', '4', 'small', 'high'], ['high', 'vhigh', '2', 'more', 'small', 'high'], ['vhigh', 'med', '3', '4', 'med', 'med'], ['med', 'vhigh', '2', '4', 'med', 'med'], ['high', 'low', '2', '4', 'big', 'low'], ['low', 'med', '2', '4', 'med', 'high'], ['vhigh', 'med', '3', '4', 'med', 'high'], ['low', 'high', '4', 'more', 'small', 'med'], ['med', 'low', '4', '2', 'small', 'med'], ['vhigh', 'low', '3', '2', 'big', 'high'], ['vhigh', 'high', '2', 'more', 'med', 'med'], ['med', 'med', '4', '2', 'med', 'high'], ['med', 'low', '5more', '4', 'small', 'med'], ['high', 'vhigh', '2', 'more', 'big', 'low'], ['med', 'low', '4', 'more', 'big', 'med'], ['high', 'vhigh', '2', 'more', 'small', 'low'], ['med', 'med', '3', '4', 'big', 'high'], ['low', 'low', '5more', 'more', 'big', 'med'], ['low', 'med', '3', '2', 'med', 'med'], ['med', 'high', '2', 'more', 'small', 'high'], ['med', 'med', '3', '4', 'med', 'low'], ['high', 'vhigh', '3', '4', 'small', 'high'], ['low', 'med', '3', '4', 'small', 'med'], ['med', 'med', '2', '2', 'small', 'med'], ['low', 'low', '2', '2', 'small', 'med'], ['low', 'vhigh', '4', '2', 'big', 'low'], ['med', 'vhigh', '5more', '4', 'med', 'high'], ['med', 'vhigh', '4', '2', 'med', 'med'], ['med', 'vhigh', '5more', '2', 'small', 'med'], ['high', 'vhigh', '5more', '4', 'big', 'med'], ['low', 'med', '3', 'more', 'med', 'med'], ['vhigh', 'vhigh', '3', 'more', 'big', 'high'], ['low', 'vhigh', '3', '2', 'small', 'med'], ['low', 'vhigh', '4', '4', 'med', 'med'], ['med', 'med', '4', '4', 'small', 'med'], ['med', 'med', '3', 'more', 'big', 'low'], ['vhigh', 'vhigh', '5more', '2', 'small', 'low'], ['vhigh', 'low', '5more', '4', 'small', 'med'], ['med', 'high', '3', '4', 'med', 'high'], ['vhigh', 'vhigh', '5more', 'more', 'big', 'low'], ['med', 'med', '2', '4', 'big', 'high'], ['high', 'high', '2', 'more', 'med', 'high'], ['low', 'med', '5more', '2', 'med', 'high'], ['vhigh', 'med', '3', '2', 'med', 'high'], ['med', 'vhigh', '2', '4', 'med', 'high'], ['high', 'high', '4', '4', 'big', 'high'], ['vhigh', 'high', '2', '2', 'med', 'high'], ['low', 'med', '2', '2', 'small', 'med'], ['low', 'high', '5more', '2', 'med', 'low'], ['vhigh', 'low', '5more', '2', 'big', 'med'], ['vhigh', 'med', '2', '2', 'big', 'high'], ['high', 'high', '3', '4', 'med', 'low'], ['low', 'med', '2', 'more', 'med', 'high'], ['vhigh', 'vhigh', '3', '4', 'med', 'high'], ['vhigh', 'vhigh', '2', 'more', 'small', 'high'], ['vhigh', 'med', '5more', '2', 'med', 'med'], ['med', 'low', '2', '2', 'med', 'low'], ['low', 'low', '4', '4', 'small', 'med'], ['low', 'high', '3', '2', 'small', 'high'], ['med', 'vhigh', '2', '2', 'small', 'low'], ['vhigh', 'vhigh', '3', 'more', 'small', 'med'], ['high', 'high', '5more', '2', 'big', 'med'], ['high', 'low', '3', '2', 'small', 'high'], ['vhigh', 'high', '4', 'more', 'big', 'low'], ['vhigh', 'med', '3', '2', 'small', 'high'], ['high', 'low', '3', '4', 'med', 'low'], ['high', 'vhigh', '2', '2', 'big', 'low'], ['low', 'high', '4', 'more', 'small', 'low'], ['high', 'high', '5more', '2', 'med', 'low'], ['low', 'high', '5more', '2', 'med', 'high'], ['med', 'med', '2', '2', 'small', 'high'], ['vhigh', 'vhigh', '4', 'more', 'big', 'low'], ['med', 'high', '4', '4', 'small', 'low'], ['high', 'high', '2', '2', 'big', 'med'], ['med', 'med', '3', 'more', 'small', 'low'], ['low', 'med', '3', '4', 'small', 'high'], ['high', 'low', '2', 'more', 'big', 'low'], ['high', 'vhigh', '2', '4', 'med', 'low'], ['med', 'med', '3', 'more', 'big', 'high'], ['vhigh', 'vhigh', '3', 'more', 'small', 'low'], ['vhigh', 'vhigh', '2', 'more', 'big', 'high'], ['vhigh', 'high', '3', 'more', 'small', 'low'], ['high', 'high', '4', 'more', 'small', 'high'], ['high', 'vhigh', '5more', '2', 'big', 'high'], ['high', 'low', '3', '2', 'big', 'high'], ['high', 'vhigh', '4', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'vhigh', '2', '2', 'med', 'med'], ['low', 'high', '3', 'more', 'big', 'med'], ['vhigh', 'low', '3', '2', 'med', 'high'], ['high', 'high', '5more', '2', 'small', 'high'], ['med', 'low', '5more', 'more', 'big', 'med'], ['vhigh', 'low', '3', '4', 'big', 'high'], ['high', 'high', '4', 'more', 'big', 'high'], ['vhigh', 'vhigh', '5more', 'more', 'small', 'low'], ['med', 'vhigh', '5more', '4', 'big', 'high'], ['med', 'high', '5more', 'more', 'big', 'med'], ['high', 'high', '3', '2', 'small', 'high'], ['med', 'vhigh', '3', 'more', 'med', 'high'], ['low', 'high', '4', 'more', 'big', 'high'], ['med', 'med', '4', 'more', 'med', 'high'], ['high', 'med', '3', '2', 'small', 'med'], ['med', 'high', '2', '2', 'small', 'med'], ['vhigh', 'med', '5more', 'more', 'small', 'high'], ['med', 'vhigh', '2', '4', 'small', 'low'], ['med', 'vhigh', '3', '4', 'small', 'low'], ['high', 'vhigh', '2', '4', 'big', 'low'], ['vhigh', 'high', '3', '2', 'med', 'med'], ['high', 'med', '3', '4', 'small', 'high'], ['low', 'vhigh', '4', '4', 'big', 'low'], ['med', 'high', '3', '2', 'small', 'low'], ['low', 'low', '3', '2', 'med', 'med'], ['low', 'vhigh', '2', 'more', 'med', 'med'], ['low', 'high', '3', '2', 'med', 'low'], ['vhigh', 'med', '5more', '2', 'med', 'high'], ['high', 'med', '2', '4', 'med', 'med'], ['med', 'med', '3', 'more', 'med', 'high'], ['low', 'high', '2', '4', 'med', 'high'], ['med', 'high', '3', 'more', 'small', 'low'], ['low', 'low', '5more', 'more', 'small', 'high'], ['vhigh', 'med', '3', '2', 'med', 'med'], ['vhigh', 'low', '2', '2', 'med', 'high'], ['vhigh', 'high', '5more', 'more', 'med', 'med'], ['low', 'vhigh', '3', '4', 'big', 'med'], ['low', 'low', '2', '4', 'med', 'med'], ['med', 'high', '4', '2', 'small', 'low'], ['vhigh', 'vhigh', '3', '4', 'big', 'low'], ['med', 'high', '3', '4', 'med', 'low'], ['vhigh', 'vhigh', '3', '2', 'med', 'low'], ['vhigh', 'vhigh', '2', 'more', 'big', 'med'], ['med', 'vhigh', '4', 'more', 'big', 'med'], ['vhigh', 'med', '2', '4', 'small', 'med'], ['high', 'vhigh', '3', '4', 'med', 'low'], ['vhigh', 'vhigh', '4', '4', 'big', 'high'], ['med', 'high', '5more', '4', 'small', 'high'], ['med', 'med', '2', '2', 'med', 'high'], ['high', 'vhigh', '5more', 'more', 'small', 'med'], ['low', 'vhigh', '2', '2', 'small', 'med'], ['med', 'low', '3', '4', 'small', 'med'], ['vhigh', 'low', '3', 'more', 'med', 'med'], ['vhigh', 'vhigh', '5more', 'more', 'med', 'med'], ['low', 'med', '4', 'more', 'med', 'med'], ['high', 'vhigh', '5more', '4', 'med', 'high'], ['vhigh', 'med', '2', 'more', 'small', 'high'], ['vhigh', 'low', '5more', '2', 'big', 'low'], ['high', 'low', '5more', '4', 'big', 'low'], ['low', 'vhigh', '3', 'more', 'med', 'med'], ['vhigh', 'low', '4', 'more', 'small', 'low'], ['vhigh', 'vhigh', '4', '2', 'small', 'med'], ['med', 'low', '2', 'more', 'big', 'low'], ['low', 'med', '3', '2', 'med', 'low'], ['med', 'high', '5more', 'more', 'med', 'low'], ['high', 'high', '4', 'more', 'med', 'high'], ['vhigh', 'vhigh', '5more', '4', 'small', 'high'], ['med', 'high', '2', '2', 'big', 'med'], ['high', 'high', '2', '2', 'med', 'high'], ['med', 'low', '3', '4', 'big', 'low'], ['med', 'vhigh', '3', 'more', 'small', 'low'], ['vhigh', 'med', '3', '2', 'med', 'low'], ['med', 'low', '2', '4', 'med', 'med'], ['med', 'vhigh', '5more', '2', 'small', 'high'], ['vhigh', 'low', '4', '4', 'small', 'high'], ['med', 'vhigh', '3', 'more', 'big', 'med'], ['vhigh', 'low', '4', 'more', 'big', 'high'], ['med', 'vhigh', '4', '2', 'small', 'high'], ['med', 'vhigh', '2', '2', 'big', 'high'], ['low', 'high', '2', '2', 'small', 'high'], ['high', 'vhigh', '2', '4', 'med', 'high'], ['low', 'high', '5more', '4', 'big', 'med'], ['high', 'high', '5more', '4', 'big', 'low'], ['med', 'vhigh', '4', '2', 'med', 'high'], ['vhigh', 'med', '3', '4', 'med', 'low'], ['high', 'med', '3', '2', 'big', 'low'], ['low', 'med', '2', 'more', 'big', 'low'], ['low', 'med', '3', '2', 'small', 'med'], ['med', 'med', '4', '2', 'small', 'high'], ['vhigh', 'med', '2', '2', 'small', 'high'], ['high', 'med', '2', '4', 'small', 'high'], ['vhigh', 'med', '2', 'more', 'big', 'low'], ['vhigh', 'low', '5more', '4', 'small', 'high'], ['low', 'high', '5more', 'more', 'big', 'low'], ['high', 'low', '5more', 'more', 'small', 'high'], ['low', 'vhigh', '2', 'more', 'med', 'low'], ['high', 'low', '4', '4', 'small', 'med'], ['high', 'high', '4', '4', 'small', 'high'], ['med', 'med', '3', '2', 'med', 'high'], ['high', 'med', '4', 'more', 'small', 'low'], ['low', 'low', '2', '2', 'big', 'low'], ['low', 'high', '2', '4', 'big', 'low'], ['vhigh', 'med', '2', '4', 'big', 'med'], ['high', 'low', '4', '2', 'small', 'med'], ['low', 'low', '3', 'more', 'small', 'low'], ['med', 'high', '5more', '2', 'med', 'high'], ['vhigh', 'med', '3', '2', 'small', 'low'], ['high', 'vhigh', '4', '2', 'med', 'med'], ['low', 'med', '3', '4', 'med', 'high'], ['vhigh', 'vhigh', '5more', '2', 'med', 'low'], ['med', 'high', '4', 'more', 'big', 'low'], ['low', 'high', '3', '2', 'big', 'med'], ['high', 'vhigh', '2', 'more', 'big', 'med'], ['high', 'high', '4', '2', 'big', 'high'], ['med', 'high', '5more', '4', 'small', 'low'], ['vhigh', 'vhigh', '4', '4', 'med', 'low'], ['med', 'med', '2', '4', 'small', 'med'], ['med', 'med', '5more', 'more', 'big', 'med'], ['low', 'low', '2', '2', 'med', 'high'], ['med', 'high', '2', '2', 'small', 'high'], ['low', 'med', '5more', '4', 'med', 'high'], ['low', 'high', '3', '4', 'big', 'high'], ['vhigh', 'high', '3', '2', 'small', 'low'], ['high', 'high', '3', 'more', 'med', 'low'], ['med', 'vhigh', '4', '2', 'big', 'high'], ['med', 'med', '2', '4', 'med', 'low'], ['med', 'low', '2', '2', 'small', 'med'], ['high', 'med', '4', 'more', 'big', 'high'], ['high', 'vhigh', '3', 'more', 'med', 'med'], ['vhigh', 'low', '5more', 'more', 'big', 'low'], ['low', 'low', '4', 'more', 'big', 'low'], ['med', 'high', '4', '4', 'small', 'high'], ['vhigh', 'low', '3', '2', 'small', 'low'], ['high', 'med', '3', '2', 'med', 'high'], ['low', 'low', '3', '4', 'big', 'med'], ['med', 'high', '4', 'more', 'med', 'high'], ['med', 'low', '3', 'more', 'small', 'low'], ['vhigh', 'low', '2', '2', 'small', 'high'], ['vhigh', 'vhigh', '4', '2', 'med', 'high'], ['med', 'med', '4', 'more', 'med', 'med'], ['vhigh', 'high', '5more', '4', 'med', 'med'], ['vhigh', 'vhigh', '3', '2', 'small', 'high'], ['high', 'low', '3', 'more', 'big', 'low'], ['vhigh', 'vhigh', '2', '4', 'med', 'low'], ['low', 'med', '5more', '2', 'med', 'low'], ['low', 'med', '3', '2', 'big', 'low'], ['high', 'high', '2', '2', 'big', 'high'], ['vhigh', 'high', '5more', 'more', 'med', 'high'], ['vhigh', 'med', '5more', 'more', 'small', 'low'], ['med', 'high', '3', '4', 'small', 'low'], ['high', 'low', '5more', '4', 'med', 'med'], ['high', 'high', '3', 'more', 'med', 'high'], ['med', 'med', '5more', '2', 'med', 'low'], ['high', 'med', '2', 'more', 'med', 'low'], ['med', 'med', '3', 'more', 'small', 'med'], ['high', 'low', '3', '2', 'med', 'low'], ['low', 'high', '4', '2', 'med', 'high'], ['high', 'vhigh', '3', 'more', 'med', 'high'], ['med', 'high', '2', '4', 'big', 'high'], ['low', 'vhigh', '3', 'more', 'small', 'med'], ['vhigh', 'low', '4', '2', 'small', 'low'], ['high', 'low', '5more', '4', 'small', 'high'], ['low', 'high', '4', '4', 'small', 'low'], ['vhigh', 'med', '5more', 'more', 'small', 'med'], ['med', 'high', '3', '4', 'small', 'med'], ['low', 'vhigh', '2', 'more', 'big', 'med'], ['low', 'low', '5more', '2', 'big', 'med'], ['high', 'low', '2', '4', 'big', 'high'], ['low', 'vhigh', '2', 'more', 'small', 'high'], ['high', 'vhigh', '2', 'more', 'med', 'high'], ['med', 'med', '4', '4', 'big', 'med'], ['high', 'high', '4', '2', 'small', 'low'], ['vhigh', 'high', '5more', '2', 'big', 'low'], ['high', 'high', '5more', '2', 'big', 'high'], ['low', 'vhigh', '3', '4', 'med', 'med'], ['high', 'high', '5more', '2', 'big', 'low'], ['med', 'vhigh', '5more', '2', 'med', 'med'], ['low', 'vhigh', '5more', 'more', 'med', 'med'], ['med', 'high', '4', '4', 'small', 'med'], ['high', 'vhigh', '3', '2', 'big', 'low'], ['high', 'vhigh', '2', 'more', 'big', 'high'], ['low', 'low', '5more', '2', 'big', 'low'], ['vhigh', 'high', '4', '4', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'med'], ['vhigh', 'high', '2', 'more', 'med', 'high'], ['low', 'low', '3', '4', 'big', 'low'], ['high', 'vhigh', '2', '2', 'big', 'high'], ['med', 'low', '4', 'more', 'med', 'low'], ['low', 'low', '4', '4', 'big', 'low'], ['high', 'vhigh', '5more', 'more', 'big', 'med'], ['low', 'vhigh', '4', '2', 'small', 'high'], ['vhigh', 'low', '4', 'more', 'med', 'high'], ['low', 'low', '3', 'more', 'small', 'high'], ['med', 'low', '2', 'more', 'big', 'high'], ['vhigh', 'med', '3', 'more', 'big', 'low'], ['vhigh', 'low', '2', '4', 'big', 'high'], ['vhigh', 'low', '5more', 'more', 'med', 'high'], ['vhigh', 'med', '2', '2', 'med', 'low'], ['vhigh', 'vhigh', '2', '4', 'big', 'med'], ['low', 'vhigh', '5more', '2', 'big', 'med'], ['high', 'med', '5more', 'more', 'med', 'med'], ['low', 'med', '2', 'more', 'big', 'high'], ['med', 'vhigh', '3', '2', 'big', 'high'], ['vhigh', 'high', '2', '4', 'big', 'med'], ['high', 'med', '3', '2', 'small', 'low'], ['low', 'vhigh', '4', '4', 'big', 'med'], ['med', 'high', '5more', '4', 'med', 'high'], ['vhigh', 'vhigh', '3', '2', 'big', 'med'], ['med', 'low', '5more', 'more', 'small', 'low'], ['med', 'low', '2', '2', 'big', 'low'], ['low', 'med', '5more', 'more', 'small', 'high'], ['vhigh', 'low', '5more', '4', 'big', 'high'], ['low', 'low', '5more', '2', 'med', 'med'], ['med', 'med', '2', 'more', 'big', 'low'], ['low', 'high', '5more', 'more', 'big', 'med'], ['med', 'vhigh', '2', '2', 'big', 'low'], ['vhigh', 'med', '4', '4', 'med', 'med'], ['high', 'low', '5more', '2', 'small', 'high'], ['low', 'low', '5more', '4', 'med', 'med'], ['med', 'low', '3', '2', 'big', 'med'], ['low', 'low', '3', '2', 'small', 'med'], ['vhigh', 'high', '3', 'more', 'big', 'high'], ['low', 'low', '5more', '2', 'small', 'med'], ['vhigh', 'med', '5more', '2', 'small', 'med'], ['med', 'med', '3', '4', 'small', 'high'], ['med', 'med', '5more', '4', 'big', 'med'], ['med', 'low', '4', '4', 'small', 'low'], ['high', 'med', '4', '2', 'small', 'med'], ['low', 'low', '4', '2', 'med', 'low'], ['med', 'low', '3', '2', 'med', 'high'], ['low', 'high', '3', '2', 'small', 'low'], ['high', 'high', '2', '4', 'big', 'high'], ['high', 'med', '4', '2', 'big', 'high'], ['high', 'med', '2', '2', 'med', 'low'], ['low', 'vhigh', '5more', '4', 'med', 'med'], ['low', 'low', '2', '4', 'big', 'med'], ['vhigh', 'high', '5more', '4', 'small', 'high'], ['high', 'med', '4', '2', 'med', 'low'], ['low', 'med', '3', 'more', 'small', 'med'], ['low', 'vhigh', '4', '4', 'big', 'high'], ['high', 'high', '4', 'more', 'small', 'low'], ['med', 'med', '2', 'more', 'med', 'high'], ['high', 'low', '5more', 'more', 'small', 'low'], ['med', 'med', '5more', '4', 'small', 'high'], ['high', 'low', '5more', '2', 'med', 'high'], ['med', 'vhigh', '2', '4', 'big', 'high'], ['low', 'med', '2', '2', 'small', 'high'], ['high', 'med', '5more', 'more', 'big', 'med'], ['low', 'med', '4', '2', 'big', 'med'], ['high', 'high', '2', '4', 'med', 'low'], ['high', 'vhigh', '4', '2', 'small', 'low'], ['low', 'low', '5more', 'more', 'med', 'high'], ['med', 'high', '3', '4', 'big', 'low'], ['vhigh', 'med', '3', 'more', 'big', 'med'], ['high', 'low', '2', '2', 'med', 'med'], ['vhigh', 'vhigh', '2', 'more', 'big', 'low'], ['low', 'med', '5more', '4', 'big', 'low'], ['low', 'vhigh', '3', 'more', 'big', 'low'], ['high', 'med', '2', '4', 'med', 'low'], ['low', 'high', '3', 'more', 'big', 'high'], ['low', 'high', '2', '4', 'big', 'med'], ['vhigh', 'low', '4', 'more', 'med', 'med'], ['vhigh', 'high', '5more', '4', 'small', 'med'], ['low', 'low', '2', '2', 'small', 'low'], ['med', 'vhigh', '5more', 'more', 'small', 'med'], ['high', 'low', '2', '4', 'med', 'med'], ['high', 'high', '2', 'more', 'big', 'low'], ['high', 'high', '4', 'more', 'med', 'med'], ['vhigh', 'vhigh', '3', '2', 'med', 'med'], ['vhigh', 'vhigh', '5more', 'more', 'big', 'med'], ['low', 'vhigh', '3', '2', 'small', 'high'], ['high', 'high', '2', 'more', 'small', 'high'], ['high', 'med', '4', '4', 'med', 'med'], ['vhigh', 'high', '3', '2', 'med', 'high'], ['high', 'med', '4', '4', 'big', 'high'], ['low', 'high', '3', '4', 'small', 'med'], ['vhigh', 'med', '2', 'more', 'small', 'low'], ['low', 'vhigh', '5more', '2', 'big', 'low'], ['high', 'vhigh', '4', '4', 'med', 'high'], ['med', 'low', '2', 'more', 'small', 'high'], ['low', 'med', '2', 'more', 'med', 'low'], ['low', 'low', '2', '2', 'med', 'med'], ['vhigh', 'med', '4', '2', 'big', 'high'], ['med', 'med', '2', 'more', 'big', 'high'], ['vhigh', 'vhigh', '5more', 'more', 'med', 'low'], ['high', 'high', '3', '2', 'big', 'high'], ['med', 'med', '4', '2', 'small', 'med'], ['high', 'low', '4', 'more', 'small', 'high'], ['med', 'med', '5more', 'more', 'big', 'low'], ['high', 'low', '4', '2', 'big', 'low'], ['low', 'low', '3', 'more', 'med', 'low'], ['vhigh', 'low', '5more', '2', 'small', 'high'], ['vhigh', 'high', '2', 'more', 'big', 'med'], ['med', 'med', '5more', '2', 'big', 'high'], ['vhigh', 'high', '3', '4', 'med', 'low'], ['med', 'low', '4', 'more', 'med', 'med'], ['vhigh', 'low', '2', 'more', 'big', 'low'], ['vhigh', 'med', '5more', '2', 'big', 'med'], ['vhigh', 'high', '5more', '4', 'big', 'high'], ['vhigh', 'low', '4', '2', 'big', 'high'], ['vhigh', 'high', '4', '4', 'big', 'low'], ['low', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'high', '5more', 'more', 'med', 'low'], ['vhigh', 'high', '3', 'more', 'small', 'high'], ['low', 'high', '4', '2', 'big', 'med'], ['low', 'med', '5more', 'more', 'big', 'high'], ['vhigh', 'vhigh', '5more', 'more', 'big', 'high'], ['low', 'med', '3', 'more', 'big', 'med'], ['med', 'low', '5more', '4', 'big', 'low'], ['high', 'med', '3', '4', 'big', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['low', 'vhigh', '3', '2', 'big', 'high'], ['med', 'low', '4', '2', 'big', 'med'], ['high', 'vhigh', '5more', 'more', 'big', 'low'], ['low', 'high', '2', '2', 'small', 'low'], ['med', 'low', '2', 'more', 'big', 'med'], ['med', 'vhigh', '4', '2', 'big', 'med'], ['vhigh', 'low', '4', 'more', 'big', 'med'], ['med', 'vhigh', '2', '4', 'big', 'low'], ['high', 'med', '2', '4', 'big', 'low'], ['high', 'high', '5more', '4', 'big', 'med'], ['vhigh', 'low', '2', 'more', 'small', 'high'], ['med', 'med', '4', '4', 'med', 'high'], ['med', 'low', '2', '2', 'big', 'high'], ['vhigh', 'med', '2', '2', 'med', 'med'], ['med', 'med', '5more', 'more', 'med', 'low'], ['vhigh', 'vhigh', '4', '2', 'small', 'low'], ['high', 'low', '4', '2', 'big', 'med'], ['vhigh', 'med', '2', '2', 'small', 'low'], ['low', 'med', '5more', 'more', 'big', 'med'], ['low', 'high', '3', 'more', 'small', 'high'], ['vhigh', 'med', '3', '4', 'small', 'low'], ['vhigh', 'high', '2', '4', 'small', 'high'], ['high', 'high', '2', '4', 'med', 'med'], ['med', 'low', '3', 'more', 'med', 'high'], ['vhigh', 'vhigh', '2', '2', 'small', 'med'], ['high', 'vhigh', '4', '4', 'big', 'high'], ['vhigh', 'low', '4', '2', 'small', 'high'], ['vhigh', 'high', '3', '2', 'big', 'high'], ['vhigh', 'med', '3', '2', 'small', 'med'], ['med', 'vhigh', '3', '2', 'med', 'high'], ['high', 'high', '2', '2', 'small', 'med'], ['low', 'high', '5more', 'more', 'big', 'high'], ['vhigh', 'low', '2', 'more', 'med', 'low'], ['high', 'vhigh', '5more', 'more', 'med', 'med'], ['high', 'low', '4', 'more', 'med', 'high'], ['low', 'high', '2', '2', 'small', 'med'], ['low', 'vhigh', '4', 'more', 'small', 'low'], ['med', 'low', '3', 'more', 'small', 'med'], ['med', 'med', '4', 'more', 'small', 'high'], ['low', 'high', '4', '2', 'small', 'low'], ['low', 'low', '5more', 'more', 'med', 'med'], ['vhigh', 'vhigh', '4', '2', 'big', 'med'], ['high', 'high', '2', '4', 'big', 'low'], ['med', 'med', '2', '4', 'med', 'med'], ['high', 'med', '4', '4', 'small', 'med'], ['vhigh', 'high', '3', '2', 'big', 'low'], ['vhigh', 'vhigh', '4', '2', 'big', 'high'], ['vhigh', 'med', '4', '4', 'big', 'low'], ['med', 'high', '2', '4', 'med', 'high'], ['vhigh', 'high', '2', 'more', 'big', 'low'], ['high', 'vhigh', '4', '2', 'big', 'high'], ['med', 'high', '2', '2', 'small', 'low'], ['vhigh', 'vhigh', '5more', 'more', 'small', 'high'], ['med', 'vhigh', '5more', '2', 'big', 'high'], ['high', 'med', '3', '4', 'med', 'med'], ['vhigh', 'high', '4', 'more', 'big', 'high'], ['low', 'vhigh', '3', 'more', 'med', 'low'], ['low', 'vhigh', '3', '4', 'med', 'low'], ['low', 'med', '4', '2', 'med', 'low'], ['vhigh', 'low', '3', '2', 'med', 'low'], ['high', 'vhigh', '2', 'more', 'small', 'med'], ['med', 'med', '3', '2', 'big', 'low'], ['low', 'med', '4', 'more', 'big', 'med'], ['low', 'high', '5more', '4', 'med', 'high'], ['vhigh', 'vhigh', '2', 'more', 'small', 'low'], ['low', 'low', '3', '2', 'big', 'high'], ['low', 'vhigh', '5more', '4', 'big', 'med'], ['med', 'low', '2', '2', 'med', 'med'], ['med', 'med', '5more', '2', 'med', 'high'], ['vhigh', 'low', '3', '2', 'big', 'med'], ['med', 'high', '3', 'more', 'big', 'med'], ['low', 'high', '2', 'more', 'small', 'med'], ['vhigh', 'med', '3', 'more', 'small', 'med'], ['low', 'med', '2', '4', 'med', 'low'], ['vhigh', 'med', '2', '2', 'med', 'high'], ['vhigh', 'vhigh', '5more', '4', 'med', 'med'], ['med', 'vhigh', '4', '4', 'small', 'low'], ['med', 'low', '4', 'more', 'med', 'high'], ['high', 'low', '5more', '2', 'big', 'low'], ['low', 'low', '5more', 'more', 'med', 'low'], ['med', 'vhigh', '4', '4', 'med', 'low'], ['high', 'low', '4', '4', 'small', 'high'], ['med', 'high', '3', '4', 'big', 'med'], ['med', 'high', '4', 'more', 'small', 'med'], ['vhigh', 'high', '4', 'more', 'med', 'high'], ['high', 'med', '2', '2', 'small', 'low'], ['med', 'low', '4', '4', 'med', 'med'], ['med', 'vhigh', '5more', 'more', 'big', 'med'], ['high', 'low', '3', 'more', 'small', 'high'], ['med', 'low', '3', '2', 'big', 'high'], ['high', 'vhigh', '3', '2', 'big', 'med'], ['low', 'vhigh', '4', '2', 'small', 'med'], ['high', 'med', '3', '4', 'big', 'high'], ['vhigh', 'med', '2', 'more', 'med', 'med'], ['low', 'med', '3', 'more', 'small', 'high'], ['high', 'med', '2', '4', 'small', 'low'], ['vhigh', 'med', '4', '2', 'small', 'med'], ['high', 'high', '2', '4', 'big', 'med'], ['med', 'vhigh', '3', '4', 'big', 'high'], ['med', 'vhigh', '2', '4', 'big', 'med'], ['vhigh', 'high', '2', 'more', 'med', 'low'], ['med', 'low', '5more', '4', 'med', 'low'], ['low', 'low', '3', '4', 'small', 'med'], ['vhigh', 'high', '2', 'more', 'small', 'low'], ['low', 'vhigh', '5more', '4', 'big', 'high'], ['high', 'med', '3', '4', 'med', 'high'], ['vhigh', 'vhigh', '5more', '4', 'med', 'high'], ['high', 'vhigh', '4', '2', 'med', 'high'], ['high', 'vhigh', '4', 'more', 'small', 'low'], ['med', 'low', '5more', '4', 'small', 'high'], ['high', 'med', '3', 'more', 'big', 'high'], ['med', 'med', '5more', 'more', 'med', 'med'], ['high', 'vhigh', '4', 'more', 'big', 'high'], ['high', 'med', '5more', '4', 'small', 'low'], ['high', 'low', '5more', 'more', 'med', 'high'], ['low', 'vhigh', '4', '4', 'small', 'med'], ['high', 'vhigh', '5more', '2', 'small', 'med'], ['high', 'med', '3', '2', 'med', 'low'], ['low', 'vhigh', '5more', 'more', 'med', 'high'], ['vhigh', 'med', '4', 'more', 'med', 'low'], ['vhigh', 'high', '5more', '2', 'med', 'high'], ['med', 'low', '2', '4', 'big', 'low'], ['vhigh', 'low', '5more', '2', 'big', 'high'], ['low', 'med', '2', '4', 'small', 'high'], ['low', 'high', '4', '4', 'big', 'high'], ['vhigh', 'med', '5more', '4', 'small', 'high'], ['med', 'med', '5more', '4', 'big', 'high'], ['low', 'vhigh', '5more', 'more', 'small', 'med'], ['low', 'vhigh', '4', 'more', 'big', 'med'], ['high', 'vhigh', '3', 'more', 'big', 'med'], ['med', 'med', '4', '2', 'small', 'low'], ['med', 'low', '4', '4', 'small', 'med'], ['med', 'vhigh', '3', '2', 'small', 'high'], ['med', 'low', '2', '4', 'small', 'med'], ['high', 'med', '5more', '2', 'big', 'low'], ['vhigh', 'low', '2', 'more', 'small', 'low'], ['low', 'low', '2', 'more', 'med', 'med'], ['vhigh', 'high', '5more', 'more', 'big', 'low'], ['vhigh', 'vhigh', '4', '2', 'med', 'low'], ['vhigh', 'med', '3', '4', 'big', 'high'], ['med', 'med', '5more', '4', 'small', 'med'], ['high', 'high', '5more', 'more', 'med', 'high'], ['vhigh', 'low', '4', '2', 'med', 'low'], ['low', 'high', '4', '2', 'med', 'low'], ['med', 'high', '2', '2', 'med', 'med'], ['med', 'vhigh', '3', '4', 'small', 'high'], ['low', 'low', '2', 'more', 'big', 'high'], ['low', 'med', '2', '2', 'med', 'high'], ['vhigh', 'low', '3', '2', 'big', 'low'], ['low', 'vhigh', '4', '2', 'big', 'med'], ['low', 'low', '4', '2', 'small', 'high'], ['low', 'low', '3', 'more', 'small', 'med'], ['high', 'med', '5more', 'more', 'med', 'high'], ['vhigh', 'high', '3', '4', 'med', 'med'], ['med', 'med', '2', '4', 'small', 'low'], ['med', 'low', '4', '2', 'med', 'high'], ['low', 'low', '3', '4', 'med', 'low'], ['high', 'med', '2', '2', 'big', 'high'], ['med', 'low', '3', '4', 'big', 'high'], ['high', 'high', '3', '2', 'big', 'med'], ['high', 'med', '3', 'more', 'big', 'med'], ['high', 'low', '4', '4', 'small', 'low'], ['high', 'low', '2', 'more', 'small', 'low'], ['med', 'med', '3', 'more', 'small', 'high'], ['low', 'high', '2', 'more', 'big', 'low'], ['med', 'med', '5more', 'more', 'small', 'med'], ['vhigh', 'med', '4', '2', 'big', 'med'], ['low', 'high', '5more', '4', 'big', 'high'], ['med', 'med', '5more', '2', 'big', 'low'], ['vhigh', 'low', '4', '2', 'small', 'med'], ['high', 'low', '2', '4', 'small', 'low'], ['vhigh', 'low', '4', '4', 'small', 'med'], ['med', 'vhigh', '5more', '4', 'big', 'low'], ['high', 'vhigh', '4', '2', 'small', 'med'], ['vhigh', 'high', '5more', '4', 'med', 'high'], ['vhigh', 'low', '2', '2', 'small', 'med'], ['high', 'med', '2', 'more', 'small', 'high'], ['low', 'med', '5more', '2', 'small', 'high'], ['high', 'vhigh', '2', '4', 'small', 'med'], ['med', 'med', '4', 'more', 'big', 'high'], ['vhigh', 'med', '4', '2', 'med', 'med'], ['low', 'vhigh', '2', '4', 'med', 'low'], ['high', 'high', '4', '2', 'med', 'low'], ['med', 'vhigh', '3', '4', 'med', 'high'], ['low', 'vhigh', '4', '4', 'med', 'high'], ['low', 'low', '3', 'more', 'big', 'med'], ['low', 'med', '4', '4', 'med', 'low'], ['low', 'vhigh', '4', '4', 'small', 'high'], ['med', 'low', '3', 'more', 'small', 'high'], ['vhigh', 'high', '3', 'more', 'med', 'high'], ['low', 'vhigh', '3', '4', 'big', 'low'], ['low', 'low', '4', 'more', 'small', 'high'], ['high', 'vhigh', '2', '2', 'med', 'high'], ['high', 'med', '5more', '4', 'big', 'high'], ['high', 'low', '2', 'more', 'small', 'high'], ['med', 'med', '5more', '4', 'med', 'low'], ['low', 'vhigh', '2', 'more', 'big', 'low'], ['vhigh', 'high', '2', '4', 'big', 'high'], ['high', 'high', '3', '2', 'small', 'med'], ['med', 'high', '5more', '4', 'big', 'high'], ['high', 'vhigh', '3', '4', 'big', 'med'], ['med', 'med', '4', '4', 'big', 'low'], ['med', 'vhigh', '4', 'more', 'small', 'med'], ['high', 'vhigh', '3', 'more', 'small', 'low'], ['low', 'med', '4', '2', 'small', 'high'], ['high', 'high', '5more', '4', 'small', 'low'], ['vhigh', 'high', '4', 'more', 'small', 'high'], ['med', 'high', '4', '2', 'big', 'med'], ['vhigh', 'med', '5more', '4', 'small', 'low'], ['low', 'low', '3', '2', 'big', 'med'], ['high', 'low', '3', 'more', 'small', 'low'], ['low', 'med', '4', 'more', 'small', 'med'], ['med', 'high', '4', '4', 'big', 'low'], ['vhigh', 'high', '5more', '2', 'small', 'med'], ['low', 'med', '2', '2', 'big', 'low'], ['low', 'vhigh', '2', '2', 'small', 'low'], ['high', 'high', '5more', 'more', 'small', 'low'], ['high', 'med', '3', '2', 'big', 'high'], ['high', 'high', '5more', '2', 'small', 'med'], ['high', 'high', '5more', 'more', 'small', 'high'], ['high', 'vhigh', '5more', '4', 'big', 'low'], ['vhigh', 'high', '3', 'more', 'med', 'med'], ['high', 'high', '4', 'more', 'big', 'med'], ['med', 'med', '2', 'more', 'med', 'med'], ['med', 'high', '2', 'more', 'small', 'low'], ['vhigh', 'med', '4', '2', 'med', 'low'], ['low', 'low', '3', '2', 'med', 'low'], ['low', 'high', '2', '4', 'med', 'med'], ['vhigh', 'low', '5more', '4', 'med', 'med'], ['med', 'vhigh', '3', '4', 'big', 'low'], ['med', 'med', '4', 'more', 'med', 'low'], ['low', 'high', '4', 'more', 'small', 'high'], ['med', 'med', '2', 'more', 'med', 'low'], ['vhigh', 'vhigh', '4', '4', 'big', 'low'], ['low', 'high', '2', '4', 'big', 'high'], ['low', 'med', '3', 'more', 'small', 'low'], ['med', 'vhigh', '5more', '4', 'med', 'low'], ['low', 'low', '2', 'more', 'small', 'med'], ['high', 'high', '3', '4', 'small', 'high'], ['vhigh', 'vhigh', '2', 'more', 'med', 'low'], ['low', 'vhigh', '2', 'more', 'med', 'high'], ['high', 'high', '4', '4', 'small', 'low'], ['med', 'low', '5more', '2', 'big', 'low'], ['high', 'low', '2', 'more', 'big', 'med'], ['med', 'high', '3', 'more', 'med', 'high'], ['vhigh', 'low', '2', '2', 'big', 'high'], ['vhigh', 'high', '2', '2', 'small', 'high'], ['vhigh', 'med', '4', '2', 'big', 'low'], ['high', 'vhigh', '5more', '4', 'small', 'low'], ['high', 'low', '4', 'more', 'small', 'low'], ['med', 'low', '4', '2', 'med', 'med'], ['high', 'med', '4', '4', 'small', 'low'], ['vhigh', 'high', '4', '2', 'small', 'med'], ['low', 'low', '4', 'more', 'med', 'high'], ['med', 'vhigh', '4', 'more', 'big', 'low'], ['med', 'high', '4', '2', 'med', 'low'], ['high', 'high', '2', '2', 'big', 'low'], ['med', 'vhigh', '4', 'more', 'med', 'med'], ['low', 'vhigh', '4', 'more', 'big', 'low'], ['med', 'high', '2', 'more', 'med', 'low'], ['low', 'high', '3', '2', 'med', 'med'], ['vhigh', 'high', '2', 'more', 'big', 'high'], ['med', 'low', '5more', '4', 'med', 'med'], ['med', 'high', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'big', 'low'], ['high', 'med', '4', '4', 'big', 'low'], ['vhigh', 'high', '3', '4', 'med', 'high'], ['vhigh', 'low', '5more', '4', 'med', 'high'], ['med', 'vhigh', '5more', '4', 'small', 'high'], ['low', 'low', '4', '2', 'big', 'high'], ['med', 'med', '4', 'more', 'big', 'med'], ['med', 'high', '4', 'more', 'big', 'med'], ['med', 'low', '5more', 'more', 'med', 'low'], ['med', 'high', '2', '4', 'big', 'low'], ['med', 'vhigh', '2', 'more', 'big', 'med'], ['low', 'med', '5more', '4', 'small', 'med'], ['vhigh', 'med', '4', '4', 'med', 'low'], ['med', 'low', '4', '4', 'small', 'high'], ['low', 'low', '3', '2', 'med', 'high'], ['vhigh', 'low', '4', '4', 'med', 'high'], ['med', 'low', '2', '2', 'small', 'low'], ['med', 'low', '2', '4', 'med', 'low'], ['med', 'low', '5more', 'more', 'med', 'high'], ['vhigh', 'low', '5more', '2', 'med', 'low'], ['low', 'high', '5more', 'more', 'small', 'low'], ['high', 'low', '3', '4', 'big', 'low'], ['vhigh', 'high', '5more', '4', 'big', 'med'], ['high', 'vhigh', '2', '4', 'small', 'low'], ['high', 'low', '3', '2', 'big', 'med'], ['low', 'high', '5more', 'more', 'med', 'high'], ['med', 'med', '5more', '2', 'med', 'med'], ['high', 'vhigh', '2', '4', 'small', 'high'], ['high', 'low', '4', '4', 'big', 'low'], ['vhigh', 'vhigh', '3', '2', 'big', 'high'], ['vhigh', 'high', '5more', '2', 'med', 'med'], ['low', 'low', '4', '4', 'med', 'high'], ['med', 'vhigh', '5more', '4', 'small', 'low'], ['med', 'high', '5more', 'more', 'small', 'med'], ['low', 'high', '3', 'more', 'small', 'low'], ['high', 'high', '5more', 'more', 'small', 'med'], ['vhigh', 'vhigh', '2', 'more', 'med', 'high'], ['high', 'low', '3', '2', 'small', 'low'], ['high', 'low', '4', '4', 'med', 'high'], ['vhigh', 'low', '4', '2', 'big', 'low'], ['med', 'low', '5more', '2', 'med', 'low'], ['med', 'med', '2', '4', 'small', 'high'], ['high', 'vhigh', '3', '2', 'small', 'med'], ['vhigh', 'high', '4', '2', 'med', 'med'], ['high', 'med', '3', '2', 'small', 'high'], ['med', 'med', '3', '4', 'med', 'high'], ['low', 'med', '4', '4', 'small', 'high'], ['med', 'vhigh', '5more', 'more', 'med', 'med'], ['low', 'high', '2', '4', 'small', 'med'], ['high', 'vhigh', '5more', '2', 'big', 'low'], ['high', 'low', '4', 'more', 'small', 'med'], ['high', 'high', '2', 'more', 'med', 'low'], ['med', 'high', '2', '4', 'big', 'med'], ['vhigh', 'high', '2', '2', 'small', 'med'], ['low', 'low', '4', '2', 'med', 'high'], ['low', 'med', '2', '2', 'med', 'med'], ['vhigh', 'low', '3', 'more', 'small', 'low'], ['high', 'low', '3', '2', 'small', 'med'], ['med', 'low', '3', '2', 'med', 'low'], ['vhigh', 'vhigh', '4', '2', 'big', 'low'], ['med', 'vhigh', '3', '4', 'med', 'low'], ['med', 'vhigh', '5more', '4', 'small', 'med'], ['vhigh', 'med', '4', 'more', 'big', 'high'], ['med', 'med', '5more', '4', 'med', 'med'], ['high', 'med', '5more', '2', 'med', 'high'], ['high', 'low', '4', '4', 'big', 'high'], ['high', 'vhigh', '3', 'more', 'med', 'low'], ['low', 'low', '2', 'more', 'med', 'high'], ['vhigh', 'med', '5more', 'more', 'big', 'med'], ['vhigh', 'low', '2', 'more', 'med', 'high'], ['high', 'vhigh', '5more', '4', 'med', 'med'], ['med', 'high', '5more', '2', 'small', 'low'], ['high', 'vhigh', '4', 'more', 'med', 'high'], ['med', 'high', '5more', '2', 'med', 'med'], ['high', 'med', '2', 'more', 'big', 'med'], ['low', 'vhigh', '3', '4', 'big', 'high'], ['low', 'low', '4', 'more', 'small', 'low'], ['med', 'med', '3', '2', 'small', 'low'], ['vhigh', 'med', '5more', 'more', 'med', 'med'], ['vhigh', 'med', '2', '4', 'med', 'med'], ['high', 'high', '3', 'more', 'small', 'high'], ['med', 'high', '5more', 'more', 'small', 'high'], ['vhigh', 'low', '3', '2', 'small', 'high'], ['med', 'vhigh', '3', '2', 'big', 'med'], ['high', 'med', '2', 'more', 'big', 'high'], ['low', 'vhigh', '4', '2', 'small', 'low'], ['high', 'low', '3', '4', 'big', 'med'], ['high', 'high', '4', '4', 'med', 'low'], ['vhigh', 'med', '3', 'more', 'small', 'high'], ['vhigh', 'vhigh', '4', 'more', 'med', 'med'], ['low', 'vhigh', '3', '2', 'med', 'med'], ['vhigh', 'high', '4', '4', 'med', 'high'], ['vhigh', 'high', '3', '2', 'small', 'med'], ['high', 'low', '5more', '4', 'big', 'med'], ['vhigh', 'med', '5more', '2', 'med', 'low'], ['med', 'med', '3', '4', 'small', 'low'], ['vhigh', 'low', '3', '4', 'small', 'low'], ['high', 'med', '5more', '2', 'small', 'low'], ['vhigh', 'high', '3', '4', 'small', 'med'], ['low', 'high', '5more', '4', 'small', 'low'], ['med', 'med', '5more', 'more', 'small', 'low'], ['med', 'low', '2', '4', 'small', 'high'], ['vhigh', 'high', '4', '4', 'small', 'med'], ['med', 'vhigh', '2', '4', 'small', 'high'], ['med', 'med', '3', '4', 'big', 'med'], ['high', 'vhigh', '5more', 'more', 'small', 'low'], ['med', 'low', '3', '2', 'big', 'low'], ['low', 'med', '5more', 'more', 'small', 'med'], ['vhigh', 'med', '5more', 'more', 'med', 'high'], ['low', 'high', '3', 'more', 'big', 'low'], ['high', 'low', '3', 'more', 'big', 'med'], ['high', 'high', '4', '4', 'med', 'med'], ['med', 'high', '3', '2', 'big', 'high'], ['high', 'low', '5more', 'more', 'med', 'low'], ['med', 'high', '2', '2', 'big', 'low'], ['low', 'high', '4', '4', 'med', 'low'], ['med', 'high', '2', 'more', 'small', 'med'], ['high', 'high', '2', '4', 'small', 'med'], ['high', 'low', '2', '4', 'small', 'high'], ['vhigh', 'med', '4', 'more', 'small', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'low'], ['high', 'med', '5more', '2', 'med', 'low'], ['high', 'high', '4', '2', 'small', 'med'], ['med', 'high', '3', 'more', 'small', 'high'], ['low', 'med', '3', '4', 'big', 'low'], ['med', 'vhigh', '5more', 'more', 'small', 'low'], ['low', 'low', '2', '4', 'big', 'low'], ['med', 'low', '4', 'more', 'big', 'low'], ['low', 'high', '3', 'more', 'med', 'low'], ['low', 'high', '2', '2', 'big', 'high'], ['low', 'low', '5more', '4', 'med', 'high'], ['high', 'low', '3', '4', 'big', 'high'], ['low', 'vhigh', '2', '4', 'big', 'med'], ['high', 'med', '3', '4', 'small', 'med'], ['low', 'low', '3', '2', 'small', 'low'], ['low', 'high', '4', 'more', 'med', 'high'], ['vhigh', 'med', '3', '2', 'big', 'high'], ['low', 'vhigh', '5more', '2', 'med', 'low'], ['high', 'vhigh', '5more', '2', 'big', 'med'], ['vhigh', 'med', '3', '2', 'big', 'low'], ['high', 'low', '3', 'more', 'big', 'high'], ['high', 'med', '4', '2', 'med', 'med'], ['vhigh', 'med', '4', '4', 'big', 'med'], ['low', 'high', '3', '4', 'small', 'high'], ['high', 'low', '3', 'more', 'small', 'med'], ['low', 'med', '3', 'more', 'big', 'high'], ['med', 'med', '5more', '4', 'small', 'low'], ['vhigh', 'med', '2', 'more', 'med', 'low'], ['vhigh', 'low', '4', '4', 'big', 'low'], ['med', 'high', '3', '2', 'med', 'high'], ['high', 'vhigh', '2', '4', 'big', 'med'], ['vhigh', 'high', '3', 'more', 'med', 'low'], ['low', 'high', '2', '2', 'big', 'med'], ['high', 'low', '2', '2', 'big', 'med'], ['low', 'med', '2', '4', 'small', 'low'], ['high', 'high', '3', '4', 'big', 'low'], ['vhigh', 'vhigh', '2', '4', 'small', 'med'], ['vhigh', 'high', '3', '4', 'big', 'med'], ['med', 'low', '4', '4', 'med', 'high'], ['med', 'low', '2', '4', 'big', 'high'], ['vhigh', 'low', '5more', 'more', 'med', 'low'], ['med', 'vhigh', '5more', '2', 'med', 'low'], ['med', 'high', '2', 'more', 'med', 'high'], ['low', 'vhigh', '2', '4', 'small', 'low'], ['high', 'low', '5more', '2', 'med', 'med'], ['vhigh', 'high', '2', '4', 'med', 'high'], ['med', 'vhigh', '4', '2', 'big', 'low'], ['vhigh', 'vhigh', '3', '2', 'med', 'high'], ['low', 'med', '4', '2', 'big', 'high'], ['vhigh', 'low', '5more', '2', 'small', 'med'], ['high', 'high', '5more', '2', 'med', 'high'], ['low', 'high', '3', 'more', 'med', 'high'], ['vhigh', 'high', '5more', 'more', 'small', 'low'], ['high', 'low', '5more', '2', 'big', 'med'], ['low', 'vhigh', '5more', 'more', 'big', 'high'], ['low', 'high', '2', 'more', 'med', 'low'], ['low', 'med', '3', '2', 'small', 'low'], ['high', 'low', '2', '2', 'big', 'low'], ['low', 'high', '2', '4', 'small', 'low'], ['high', 'high', '3', 'more', 'big', 'high'], ['low', 'med', '5more', '4', 'big', 'high'], ['med', 'low', '4', '2', 'big', 'high'], ['vhigh', 'med', '2', 'more', 'big', 'high'], ['med', 'high', '5more', '4', 'small', 'med'], ['vhigh', 'vhigh', '3', '4', 'med', 'low'], ['high', 'med', '5more', 'more', 'small', 'high'], ['low', 'low', '2', '2', 'big', 'med'], ['low', 'vhigh', '4', '2', 'big', 'high'], ['vhigh', 'high', '5more', '2', 'med', 'low'], ['vhigh', 'low', '3', 'more', 'small', 'high'], ['low', 'high', '3', '2', 'med', 'high'], ['low', 'low', '3', 'more', 'med', 'med'], ['high', 'low', '5more', 'more', 'small', 'med'], ['vhigh', 'vhigh', '5more', 'more', 'med', 'high'], ['low', 'med', '5more', '2', 'small', 'med'], ['low', 'low', '5more', '4', 'small', 'high'], ['low', 'med', '4', '2', 'small', 'low'], ['med', 'high', '4', 'more', 'med', 'low'], ['med', 'vhigh', '2', '2', 'big', 'med'], ['med', 'vhigh', '2', 'more', 'big', 'low'], ['vhigh', 'low', '5more', 'more', 'big', 'med'], ['low', 'vhigh', '5more', 'more', 'big', 'med'], ['high', 'vhigh', '2', '2', 'med', 'low'], ['low', 'low', '4', 'more', 'small', 'med'], ['low', 'vhigh', '2', 'more', 'small', 'low'], ['med', 'med', '2', '4', 'big', 'med'], ['high', 'high', '5more', '4', 'med', 'med'], ['med', 'vhigh', '4', '4', 'small', 'high'], ['med', 'vhigh', '5more', '2', 'big', 'low'], ['low', 'low', '5more', 'more', 'small', 'low'], ['med', 'high', '5more', '4', 'med', 'med'], ['vhigh', 'vhigh', '3', 'more', 'med', 'high'], ['high', 'vhigh', '2', '4', 'med', 'med'], ['vhigh', 'high', '5more', 'more', 'big', 'high'], ['low', 'high', '5more', '2', 'big', 'low'], ['low', 'low', '4', '2', 'small', 'low'], ['vhigh', 'vhigh', '3', '4', 'med', 'med'], ['vhigh', 'high', '5more', '4', 'big', 'low'], ['low', 'med', '2', '4', 'small', 'med'], ['high', 'vhigh', '4', '2', 'big', 'med'], ['vhigh', 'high', '4', '4', 'big', 'high'], ['low', 'med', '5more', 'more', 'small', 'low'], ['vhigh', 'med', '2', '4', 'small', 'high'], ['vhigh', 'low', '2', '2', 'med', 'med'], ['high', 'low', '5more', '2', 'small', 'low'], ['high', 'low', '3', '2', 'med', 'med'], ['vhigh', 'vhigh', '4', 'more', 'small', 'low'], ['vhigh', 'med', '5more', '4', 'med', 'low'], ['vhigh', 'vhigh', '2', '4', 'small', 'low'], ['med', 'high', '5more', 'more', 'big', 'low'], ['high', 'high', '3', '2', 'med', 'low'], ['low', 'low', '2', '2', 'big', 'high'], ['low', 'med', '5more', '4', 'big', 'med'], ['high', 'vhigh', '5more', '4', 'med', 'low'], ['low', 'low', '4', '2', 'med', 'med'], ['low', 'high', '3', 'more', 'med', 'med'], ['med', 'low', '5more', '2', 'big', 'high'], ['med', 'low', '5more', '4', 'small', 'low'], ['high', 'vhigh', '3', '2', 'big', 'high'], ['med', 'low', '2', 'more', 'med', 'med'], ['vhigh', 'high', '4', '2', 'small', 'high'], ['high', 'vhigh', '5more', '2', 'small', 'low'], ['vhigh', 'high', '3', '4', 'big', 'low'], ['med', 'vhigh', '3', 'more', 'small', 'med'], ['low', 'high', '4', '2', 'med', 'med'], ['high', 'vhigh', '3', '4', 'big', 'high'], ['high', 'vhigh', '4', 'more', 'small', 'med'], ['vhigh', 'vhigh', '4', '4', 'small', 'med'], ['low', 'high', '3', 'more', 'small', 'med'], ['med', 'high', '4', '4', 'med', 'high'], ['high', 'vhigh', '3', '4', 'big', 'low'], ['low', 'vhigh', '2', '2', 'med', 'med'], ['med', 'vhigh', '4', '4', 'small', 'med'], ['vhigh', 'high', '5more', '2', 'big', 'med'], ['vhigh', 'low', '2', 'more', 'big', 'med'], ['low', 'low', '3', '4', 'med', 'high'], ['vhigh', 'med', '3', 'more', 'big', 'high'], ['vhigh', 'vhigh', '4', '4', 'med', 'high'], ['high', 'vhigh', '5more', '2', 'med', 'high'], ['low', 'med', '4', '4', 'med', 'med'], ['low', 'vhigh', '4', 'more', 'med', 'high'], ['vhigh', 'med', '4', '4', 'big', 'high'], ['med', 'med', '4', '2', 'big', 'high'], ['low', 'low', '2', '4', 'med', 'low'], ['vhigh', 'vhigh', '3', '4', 'small', 'low'], ['low', 'med', '5more', '2', 'med', 'med'], ['high', 'low', '5more', 'more', 'med', 'med'], ['med', 'low', '3', 'more', 'big', 'high'], ['high', 'vhigh', '5more', 'more', 'med', 'high'], ['high', 'vhigh', '5more', '4', 'big', 'high'], ['high', 'med', '3', 'more', 'med', 'high'], ['vhigh', 'med', '4', 'more', 'big', 'low'], ['high', 'med', '2', 'more', 'med', 'high'], ['med', 'high', '5more', '2', 'small', 'med'], ['med', 'high', '2', 'more', 'big', 'med'], ['high', 'low', '3', 'more', 'med', 'low'], ['med', 'low', '2', '4', 'small', 'low'], ['med', 'high', '5more', '2', 'big', 'high'], ['low', 'med', '3', 'more', 'big', 'low'], ['med', 'vhigh', '3', 'more', 'big', 'high'], ['vhigh', 'vhigh', '3', 'more', 'med', 'med'], ['vhigh', 'high', '5more', 'more', 'small', 'med'], ['high', 'high', '5more', 'more', 'big', 'high'], ['low', 'high', '2', '2', 'med', 'low'], ['low', 'vhigh', '4', '2', 'med', 'low'], ['vhigh', 'high', '5more', '2', 'small', 'high'], ['high', 'vhigh', '3', '2', 'small', 'high'], ['med', 'vhigh', '5more', 'more', 'med', 'high'], ['vhigh', 'med', '5more', '4', 'big', 'high'], ['low', 'high', '3', '4', 'med', 'med'], ['med', 'low', '5more', '2', 'med', 'med'], ['high', 'low', '4', '4', 'med', 'low'], ['high', 'high', '2', '2', 'small', 'low'], ['vhigh', 'vhigh', '5more', '4', 'med', 'low'], ['low', 'med', '2', '2', 'med', 'low'], ['med', 'low', '3', 'more', 'big', 'low'], ['med', 'med', '3', 'more', 'med', 'med'], ['med', 'vhigh', '4', '2', 'small', 'low'], ['low', 'low', '5more', '4', 'big', 'high'], ['vhigh', 'vhigh', '4', '4', 'small', 'low'], ['med', 'high', '3', '2', 'small', 'med'], ['high', 'med', '3', '4', 'big', 'low'], ['high', 'high', '3', 'more', 'small', 'low'], ['vhigh', 'med', '2', '4', 'big', 'low'], ['low', 'med', '2', 'more', 'small', 'med'], ['low', 'med', '2', '2', 'big', 'high'], ['med', 'low', '5more', '4', 'big', 'med'], ['med', 'low', '4', 'more', 'small', 'low'], ['high', 'med', '4', '2', 'small', 'high'], ['vhigh', 'vhigh', '2', '2', 'big', 'high'], ['vhigh', 'vhigh', '4', 'more', 'small', 'high'], ['vhigh', 'med', '5more', '4', 'med', 'med'], ['high', 'vhigh', '4', 'more', 'med', 'low'], ['vhigh', 'vhigh', '5more', '2', 'big', 'med']]

car_labels = ['acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'acc', 'unacc', 'acc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'vgood', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'good', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'vgood', 'unacc', 'acc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'vgood', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'good', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'good', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'good', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'good', 'acc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'vgood', 'vgood', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'good', 'unacc', 'vgood', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'vgood', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'vgood', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'good', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'good', 'vgood', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'good', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'good', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'good', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'good', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'vgood', 'acc', 'vgood', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'good', 'unacc', 'acc', 'good', 'unacc', 'unacc', 'good', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'good', 'unacc', 'acc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'acc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'vgood', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'good', 'acc', 'unacc', 'unacc', 'good', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'vgood', 'unacc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'acc', 'acc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'vgood', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'unacc', 'good', 'unacc', 'unacc', 'unacc', 'unacc', 'acc', 'unacc', 'unacc']

tree = build_tree(cars, car_labels)

#--------------------------- END of tree.py-------------------------------
'''
DECISION TREES
Decision Trees in scikit-learn
Nice work! You’ve written a decision tree from scratch that is able to classify new points. Let’s take a look at how the Python library scikit-learn implements decision trees.

The sklearn.tree module contains the DecisionTreeClassifier class. To create a DecisionTreeClassifier object, call the constructor:

classifier = DecisionTreeClassifier()
Next, we want to create the tree based on our training data. To do this, we’ll use the .fit() method.

.fit() takes a list of data points followed by a list of the labels associated with that data. Note that when we built our tree from scratch, our data points contained strings like "vhigh" or "5more". When creating the tree using scikit-learn, it’s a good idea to map those strings to numbers. For example, for the first feature representing the price of the car, "low" would map to 1, "med" would map to 2, and so on.

classifier.fit(training_data, training_labels)
Finally, once we’ve made our tree, we can use it to classify new data points. The .predict() method takes an array of data points and will return an array of classifications for those data points.

predictions = classifier.predict(test_data)
If you’ve split your data into a test set, you can find the accuracy of the model by calling the .score() method using the test data and the test labels as parameters.

print(classifier.score(test_data, test_labels))
.score() returns the percentage of data points from the test set that it classified correctly.
'''
#-----------------script.py-------------------------------
from cars import training_points, training_labels, testing_points, testing_labels
from sklearn.tree import DecisionTreeClassifier

print(training_points[0],training_labels[0]) 

classifier = DecisionTreeClassifier()

classifier.fit(training_points, training_labels)

predictions = classifier.predict(testing_points)

print(classifier.score(testing_points, testing_labels))
#--------------------- 
import random
random.seed(1)

def make_cars():
    f = open("car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars
  
def change_data(data):
    dicts = [{'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'2' : 1.0, '3' : 2.0, '4' : 3.0, '5more' : 4.0},
    {'2' : 1.0, '4' : 2.0, 'more' : 3.0},
    {'small' : 1.0, 'med' : 2.0, 'big' : 3.0},
    {'low' : 1.0, 'med' : 2.0, 'high' : 3.0}]

    for row in data:
        for i in range(len(dicts)):
            row[i] = dicts[i][row[i]]

    return data
  
cars = change_data(make_cars())
random.shuffle(cars)
car_data = [x[:-1] for x in cars]
car_labels = [x[-1] for x in cars]

training_points = car_data[:int(len(car_data)*0.9)]
training_labels = car_labels[:int(len(car_labels)*0.9)]

testing_points = car_data[int(len(car_data)*0.9):]
testing_labels = car_labels[int(len(car_labels)*0.9):]

#---------------------------car.data-----------------------
vhigh,vhigh,2,2,small,low,unacc
vhigh,vhigh,2,2,small,med,unacc
vhigh,vhigh,2,2,small,high,unacc
vhigh,vhigh,2,2,med,low,unacc
vhigh,vhigh,2,2,med,med,unacc
vhigh,vhigh,2,2,med,high,unacc
vhigh,vhigh,2,2,big,low,unacc
vhigh,vhigh,2,2,big,med,unacc
vhigh,vhigh,2,2,big,high,unacc
vhigh,vhigh,2,4,small,low,unacc
vhigh,vhigh,2,4,small,med,unacc
vhigh,vhigh,2,4,small,high,unacc
vhigh,vhigh,2,4,med,low,unacc
vhigh,vhigh,2,4,med,med,unacc
vhigh,vhigh,2,4,med,high,unacc
vhigh,vhigh,2,4,big,low,unacc
vhigh,vhigh,2,4,big,med,unacc
vhigh,vhigh,2,4,big,high,unacc
vhigh,vhigh,2,more,small,low,unacc
vhigh,vhigh,2,more,small,med,unacc
vhigh,vhigh,2,more,small,high,unacc
vhigh,vhigh,2,more,med,low,unacc
'
'
'
'
'
'
'
'
'
#---------------------------end of car.data------------------
'''
DECISION TREES
Decision Tree Limitations
Now that we have an understanding of how decision trees are created and used, let’s talk about some of their limitations.

One problem with the way we’re currently making our decision trees is that our trees aren’t always globablly optimal. This means that there might be a better tree out there somewhere that produces better results. But wait, why did we go through all that work of finding information gain if it’s not producing the best possible tree?

Our current strategy of creating trees is greedy. We assume that the best way to create a tree is to find the feature that will result in the largest information gain right now and split on that feature. We never consider the ramifications of that split further down the tree. It’s possible that if we split on a suboptimal feature right now, we would find even better splits later on. Unfortunately, finding a globally optimal tree is an extremely difficult task, and finding a tree using our greedy approach is a reasonable substitute.

Another problem with our trees is that they potentially overfit the data. This means that the structure of the tree is too dependent on the training data and doesn’t accurately represent the way the data in the real world looks like. In general, larger trees tend to overfit the data more. As the tree gets bigger, it becomes more tuned to the training data and it loses a more generalized understanding of the real world data.

One way to solve this problem is to prune the tree. The goal of pruning is to shrink the size of the tree. There are a few different pruning strategies, and we won’t go into the details of them here. scikit-learn currently doesn’t prune the tree by default, however we can dig into the code a bit to prune it ourselves.
'''
'''
Instructions
1.
We’ve created a decision tree classifier for you and printed its accuracy. Let’s see how big this tree is.

If your classifier is named classifier, you can find the depth of the tree by printing classifier.tree_.max_depth.

Print the depth of classifier‘s decision tree.

Take note of the accuracy as well.

Print classifier.tree_.max_depth.

Don’t forget the underscore at the end of tree_!

2.
classifier should have a depth of 12. Let’s prune it! When you create classifier, set the parameter max_depth equal to 11.

What is the accuracy of the classifier after pruning the tree from size 12 to size 11?

The constructor should now look like this:

classifier = DecisionTreeClassifier(random_state = 0, max_depth = ____)
Fill in the value for the new max_depth.
'''
from cars import training_points, training_labels, testing_points, testing_labels
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state = 0, max_depth = 10)
classifier.fit(training_points, training_labels)
print(classifier.score(testing_points, testing_labels))

print(classifier.tree_.max_depth)

'''

DECISION TREES
Review
Great work! In this lesson, you learned how to create decision trees and use them to make classifications. Here are some of the major takeaways:

Good decision trees have pure leaves. A leaf is pure if all of the data points in that class have the same label.
Decision trees are created using a greedy algorithm that prioritizes finding the feature that results in the largest information gain when splitting the data using that feature.
Creating an optimal decision tree is difficult. The greedy algorithm doesn’t always find the globally optimal tree.
Decision trees often suffer from overfitting. Making the tree small by pruning helps to generalize the tree so it is more accurate on data in the real world.
'''
'''
RANDOM FORESTS
Random Forest
We’ve seen that decision trees can be powerful supervised machine learning models. However, they’re not without their weaknesses — decision trees are often prone to overfitting.

We’ve discussed some strategies to minimize this problem, like pruning, but sometimes that isn’t enough. We need to find another way to generalize our trees. This is where the concept of a random forest comes in handy.

A random forest is an ensemble machine learning technique — a random forest contains many decision trees that all work together to classify new points. When a random forest is asked to classify a new point, the random forest gives that point to each of the decision trees. Each of those trees reports their classification and the random forest returns the most popular classification. It’s like every tree gets a vote, and the most popular classification wins.

Some of the trees in the random forest may be overfit, but by making the prediction based on a large number of trees, overfitting will have less of an impact.

In this lesson, we’ll learn how the trees in a random forest get created.
'''
'''
RANDOM FORESTS
Bagging
You might be wondering how the trees in the random forest get created. After all, right now, our algorithm for creating a decision tree is deterministic — given a training set, the same tree will be made every time.

Random forests create different trees using a process known as bagging. Every time a decision tree is made, it is created using a different subset of the points in the training set. For example, if our training set had 1000 rows in it, we could make a decision tree by picking 100 of those rows at random to build the tree. This way, every tree is different, but all trees will still be created from a portion of the training data.

One thing to note is that when we’re randomly selecting these 100 rows, we’re doing so with replacement. Picture putting all 100 rows in a bag and reaching in and grabbing one row at random. After writing down what row we picked, we put that row back in our bag.

This means that when we’re picking our 100 random rows, we could pick the same row more than once. In fact, it’s very unlikely, but all 100 randomly picked rows could all be the same row!

Because we’re picking these rows with replacement, there’s no need to shrink our bagged training set from 1000 rows to 100. We can pick 1000 rows at random, and because we can get the same row more than once, we’ll still end up with a unique data set.

Let’s implement bagging! We’ll be using the data set of cars that we used in our decision tree lesson.'''
'''
Instructions
1.
Start by creating a tree using all of the data we’ve given you. Create a variable named tree and set it equal to the build_tree() function using car_data and car_labels as parameters.

Then call print_tree() using tree as a parameter. Scroll up to the top to see the root of the tree. Which feature is used to split the data at the root?

2.
For now, comment out printing the tree.

Let’s now implement bagging. The original dataset has 1000 items in it. We want to randomly select a subset of those with replacement.

Create a list named indices that contains 1000 random numbers between 0 and 1000. We’ll use this list to remember the 1000 cars and the 1000 labels that we’re going to build a tree with.

You can use either a for loop or list comprehension to make this list. To get a random number between 0 and 1000, use random.randint(0, 999).

If you choose to use a for loop, your code might look something like this:

indices = []
for i in range(1000):
  indices.append(_____)
If you choose to use list comprehension, your code might look like this:

indices = [_____ for i in range(1000)]
3.
Create two new lists named data_subset and labels_subset. These two lists should contain the cars and labels found at each index in indices.

Once again, you can use either a for loop or list comprehension to make these lists.

If you choose to use a for loop, your code might look something like this:

data_subset = []
labels_subset = []
for index in indices:
  data_subset.append(car_data[index])
  labels_subset.append(_____)
If you choose to use list comprehension, your code might look like this:

data_subset = [car_data[index] for index in indices]
labels_subset = [_____]
4.
Create a tree named subset_tree using the build_tree() function with data_subset and labels_subset as parameters.

Print subset_tree using the print_tree() function.

Which feature is used to split the data at the root? Is it a different feature than the feature that split the tree that was created using all of the data?

You’ve just created a new tree from the training set! If you used 1000 different indices, you’d get another different tree. You could now create a random forest by creating multiple different trees!

Fill in the correct parameters:

subset_tree = build_tree(____, ____)
Then make sure to print the tree.
'''

#--------------tree.py-------------------------------
from collections import Counter
import random
random.seed(1)

def make_cars():
    f = open("car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars
  
cars = make_cars()
random.shuffle(cars)
cars = cars[:1000]
car_data = [x[:-1] for x in cars]
car_labels = [x[-1] for x in cars]

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset)/len(starting_labels)
  return info_gain  
  
class Leaf:

    def __init__(self, labels, value):
        self.predictions = Counter(labels)
        self.value = value

class Decision_Node:


    def __init__(self,
                 question,
                 branches, value):
        self.question = question
        self.branches = branches
        self.value = value
  
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1:"Price of maintenance", 2:"Number of doors", 3:"Person Capacity", 4:"Size of luggage boot", 5:"Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + question_dict[node.question])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print (spacing + '--> Branch ' + node.branches[i].value+':')
        print_tree(node.branches[i], spacing + "  ")
        
def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_gain, best_feature
  
def build_tree(rows, labels, value = ""):
    gain, question = find_best_split(rows, labels)
    if gain == 0:
        return Leaf(labels, value)
    data_subsets, label_subsets = split(rows, labels, question)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][question])
        branches.append(branch)
    return Decision_Node(question, branches, value)

#--------------------script.py-------------------------------
from tree import build_tree, print_tree, car_data, car_labels
import random
random.seed(4)

tree = build_tree(car_data, car_labels)
#print_tree(tree)

#indices = [random.randint(0, 999) x for x in range(1000)]
indices = []
for i in range(1000):
  indices.append(random.randint(0,999))

#print(indices)

data_subset = [car_data[x] for x in indices]
labels_subset = [car_labels[x] for x in indices]

subset_tree =  build_tree(data_subset, labels_subset)
print_tree(subset_tree)


#----------------------car.data------------------
vhigh,vhigh,2,2,small,low,unacc
vhigh,vhigh,2,2,small,med,unacc
vhigh,vhigh,2,2,small,high,unacc
vhigh,vhigh,2,2,med,low,unacc
vhigh,vhigh,2,2,med,med,unacc
vhigh,vhigh,2,2,med,high,unacc
vhigh,vhigh,2,2,big,low,unacc
vhigh,vhigh,2,2,big,med,unacc
vhigh,vhigh,2,2,big,high,unacc
vhigh,vhigh,2,4,small,low,unacc
vhigh,vhigh,2,4,small,med,unacc
vhigh,vhigh,2,4,small,high,unacc
vhigh,vhigh,2,4,med,low,unacc
vhigh,vhigh,2,4,med,med,unacc
vhigh,vhigh,2,4,med,high,unacc
vhigh,vhigh,2,4,big,low,unacc
vhigh,vhigh,2,4,big,med,unacc
vhigh,vhigh,2,4,big,high,unacc
vhigh,vhigh,2,more,small,low,unacc
vhigh,vhigh,2,more,small,med,unacc
vhigh,vhigh,2,more,small,high,unacc
vhigh,vhigh,2,more,med,low,unacc
vhigh,vhigh,2,more,med,med,unacc
vhigh,vhigh,2,more,med,high,unacc
vhigh,vhigh,2,more,big,low,unacc
vhigh,vhigh,2,more,big,med,unacc
vhigh,vhigh,2,more,big,high,unacc
vhigh,vhigh,3,2,small,low,unacc
vhigh,vhigh,3,2,small,med,unacc
vhigh,vhigh,3,2,small,high,unacc
vhigh,vhigh,3,2,med,low,unacc
....

'''
RANDOM FORESTS
Bagging Features
We’re now making trees based on different random subsets of our initial dataset. But we can continue to add variety to the ways our trees are created by changing the features that we use.

Recall that for our car data set, the original features were the following:

The price of the car
The cost of maintenance
The number of doors
The number of people the car can hold
The size of the trunk
The safety rating
Right now when we create a decision tree, we look at every one of those features and choose to split the data based on the feature that produces the most information gain. We could change how the tree is created by only allowing a subset of those features to be considered at each split.

For example, when finding which feature to split the data on the first time, we might randomly choose to only consider the price of the car, the number of doors, and the safety rating.

After splitting the data on the best feature from that subset, we’ll likely want to split again. For this next split, we’ll randomly select three features again to consider. This time those features might be the cost of maintenance, the number of doors, and the size of the trunk. We’ll continue this process until the tree is complete.

One question to consider is how to choose the number of features to randomly select. Why did we choose 3 in this example? A good rule of thumb is to randomly select the square root of the total number of features. Our car dataset doesn’t have a lot of features, so in this example, it’s difficult to follow this rule. But if we had a dataset with 25 features, we’d want to randomly select 5 features to consider at every split point.

'''
'''
Instructions
1.
We’ve given you access to the code that finds the best feature to split on. Right now, it considers all possible features. We’re going to want to change that!

For now, let’s see what the best feature to split the dataset is. At the bottom of your code, call find_best_split() using data_subset and labels_subset as parameters and print the results.

This function returns the information gain and the index of the best feature. What was the index?

That index corresponds to the features of our car. For example, if the best feature index to split on was 0, that means we’re splitting on the price of the car.

2.
We now want to modify our find_best_split() function to only consider a subset of the features. We want to pick 3 features without replacement.

The random.choice() function found in Python’s numpy module can help us do this. random.choice() returns a list of values between 0 and the first parameter. The size of the list is determined by the second parameter. And we can choose without replacement by setting replace = False.

For example, the following code would choose ten unique numbers between 0 and 100 (exclusive) and put them in a list.

lst = np.random.choice(100, 10, replace = False)
Inside find_best_split(), create a list named features that contains 3 numbers between 0 and len(dataset[0]).

Instead of looping through feature in range(len(dataset[0])), loop through feature in features.

Now that we’ve implemented feature bagging, what is the best index to use as the split index?

Fill in the correct first parameter of np.random.choice. Then make sure to loop through features.

 features = np.random.choice(____, 3, replace=False)
    for feature in features:
      #Code in the loop shouldn't change

'''

#------------ script.py-------------------------------
from tree import car_data, car_labels, split, information_gain
import random
import numpy as np
np.random.seed(10)
random.seed(4)

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    #Create features here
    features = np.random.choice(len(dataset[0]), 3, replace = False)
    print(features)
    
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_gain, best_feature
  
indices = [random.randint(0, 999) for i in range(1000)]

data_subset = [car_data[index] for index in indices]
labels_subset = [car_labels[index] for index in indices]

print(find_best_split(data_subset, labels_subset))

#---------- tree.py -------------------------------------
from collections import Counter
import random
random.seed(1)



def make_cars():
    f = open("car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars
  
cars = make_cars()
random.shuffle(cars)
cars = cars[:1000]
car_data = [x[:-1] for x in cars]
car_labels = [x[-1] for x in cars]

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset)/len(starting_labels)
  return info_gain  
  
class Leaf:

    def __init__(self, labels, value):
        self.predictions = Counter(labels)
        self.value = value

class Decision_Node:


    def __init__(self,
                 question,
                 branches, value):
        self.question = question
        self.branches = branches
        self.value = value
  
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1:"Price of maintenance", 2:"Number of doors", 3:"Person Capacity", 4:"Size of luggage boot", 5:"Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + question_dict[node.question])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print (spacing + '--> Branch ' + node.branches[i].value+':')
        print_tree(node.branches[i], spacing + "  ")
        
'''
RANDOM FORESTS
Classify
Now that we can make different decision trees, it’s time to plant a whole forest! Let’s say we make different 8 trees using bagging and feature bagging. We can now take a new unlabeled point, give that point to each tree in the forest, and count the number of times different labels are predicted.

The trees give us their votes and the label that is predicted most often will be our final classification! For example, if we gave our random forest of 8 trees a new data point, we might get the following results:

["vgood", "vgood", "good", "vgood", "acc", "vgood", "good", "vgood"]
Since the most commonly predicted classification was "vgood", this would be the random forest’s final classification.

Let’s write some code that can classify an unlabeled point!'''

'''
Instructions
1.
At the top of your code, we’ve included a new unlabeled car named unlabeled_point that we want to classify. We’ve also created a tree named subset_tree that was created using bagging and feature bagging.

Let’s see how that tree classifies this point. Print the results of classify() using unlabeled_point and subset_tree as parameters.

Print classify(unlabeled_point, subset_tree).

2.
That’s the prediction using one tree. Let’s make 20 trees and record the prediction of each one!

Take all of your code between creating indices and the print statement you just wrote and put it in a for loop that happens 20 times.

Above your for loop, create a variable named predictions and set it equal to an empty list. Inside your for loop, instead of printing the prediction, use .append() to add it to predictions.

Finally after your for loop, print predictions.

Your loop should look like this:

for i in range(20):
  # Code that creates the tree and makes the classification.
Inside your for loop, you should now have this line instead of your print statement:

predictions.append(classify(unlabeled_point, subset_tree))
3.
We now have a list of 20 predictions — let’s find the most common one! You can find the most common element in a list by using this line of code:

max(predictions, key=predictions.count)
Outside of your for loop, store the most common element in a variable named final_prediction and print that variable.'''

#-------- script.py-------------------------------

from tree import build_tree, print_tree, car_data, car_labels, classify
import random
random.seed(4)

# The features are the price of the car, the cost of maintenance, the number of doors, the number of people the car can hold, the size of the trunk, and the safety rating
unlabeled_point = ['high', 'vhigh', '3', 'more', 'med', 'med']



predictions = []

for x in list(range(20)):

  indices = [random.randint(0, 999) for i in range(1000)]
  data_subset = [car_data[index] for index in indices]
  labels_subset = [car_labels[index] for index in indices]
  subset_tree = build_tree(data_subset, labels_subset)

  pred = classify(unlabeled_point, subset_tree)
  predictions.append(pred)

print(predictions)


final_prediction = max(predictions, key=predictions.count)
print(final_prediction)


#-------- tree.py--------------------------------------

import operator
from collections import Counter
import random
import numpy as np
np.random.seed(1)
random.seed(1)

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset)/len(starting_labels)
  return info_gain

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value

class Internal_Node:
    def __init__(self,
                 feature,
                 branches,
                 value):
        self.feature = feature
        self.branches = branches
        self.value = value

def find_best_split_subset(dataset, labels, num_features):
    features = np.random.choice(6, 3, replace=False)
    best_gain = 0
    best_feature = 0
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def build_tree(data, labels, value = ""):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain < 0.00000001:
    return Leaf(Counter(labels), value)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
    branches.append(branch)
  return Internal_Node(best_feature, branches, value)

def build_tree_forest(data,labels, n_features, value=""):
    best_feature, best_gain = find_best_split_subset(data, labels, n_features)
    if best_gain < 0.00000001:
      return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
      branch = build_tree_forest(data_subsets[i], label_subsets[i], n_features, data_subsets[i][0][best_feature])
      branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1:"Price of maintenance", 2:"Number of doors", 3:"Person Capacity", 4:"Size of luggage boot", 5:"Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + str(node.labels))
        return

    # Print the question at this node
    print (spacing + "Splitting on " + question_dict[node.feature])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print (spacing + '--> Branch ' + node.branches[i].value+':')
        print_tree(node.branches[i], spacing + "  ")


def make_cars():
    f = open("car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars



def change_data(data):
    dicts = [{'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'2' : 1.0, '3' : 2.0, '4' : 3.0, '5more' : 4.0},
    {'2' : 1.0, '4' : 2.0, 'more' : 3.0},
    {'small' : 1.0, 'med' : 2.0, 'big' : 3.0},
    {'low' : 1.0, 'med' : 2.0, 'high' : 3.0}]

    for row in data:
        for i in range(len(dicts)):
            row[i] = dicts[i][row[i]]

    return data


def classify(datapoint, tree):
  if isinstance(tree, Leaf):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]

  value = datapoint[tree.feature]
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)
  #return classify(datapoint, tree.branches[random.randint(0, len(tree.branches)-1)])



cars = make_cars()
random.shuffle(cars)
car_data = [x[:-1] for x in cars]
car_labels = [x[-1] for x in cars]

'''RANDOM FORESTS
Test Set
We’re now able to create a random forest, but how accurate is it compared to a single decision tree? To answer this question we’ve split our data into a training set and test set. By building our models using the training set and testing on every data point in the test set, we can calculate the accuracy of both a single decision tree and a random forest.

We’ve given you code that calculates the accuracy of a single tree. This tree was made without using any of the bagging techniques we just learned. We created the tree by using every row from the training set once and considered every feature when splitting the data rather than a random subset.

Let’s also calculate the accuracy of a random forest and see how it compares!'''

'''
Instructions
1.
Begin by taking a look at the code we’ve given you. We’ve created a single tree using the training data, looped through every point in the test set, counted the number of points the tree classified correctly and reported the percentage of correctly classified points — this percentage is known as the accuracy of the model.

Run the code to see the accuracy of the single decision tree.

2.
Right below where tree is created, create a random forest named forest using our make_random_forest() function.

This function takes three parameters — the number of trees in the forest, the training data, and the training labels. It returns a list of trees.

Create a random forest with 40 trees using training_data and training_labels.

You should also create a variable named forest_correct and start it at 0. This is the variable that will keep track of how many points in the test set the random forest correctly classifies.

Fill in the last two parameters:

forest = make_random_forest(40, ____, ____)
Don’t forget to create forest_correct as well!

3.
For every data point in the test set, we want every tree to classify the data point, find the most common classification, and compare that prediction to the true label of the data point. This is very similar to what you did in the previous exercise.

To begin, at the end of the for loop outside the if statement, create an empty list named predictions. Next, loop through every forest_tree in forest. Call classify() using testing_data[i] and forest_tree as parameters and append the result to predictions.

Inside the for loop, you should add code that looks like this. Fill in the correct parameters to the classify() function:

predictions = []
for forest_tree in forest:
  predictions.append(classify(____, ____))
4.
After we loop through every tree in the forest, we now want to find the most common prediction and compare it to the true label. The true label can be found using testing_labels[i]. If they’re equal, we’ve correctly classified a point and should add 1 to forest_correct.

An easy way of finding the most common prediction is by using this line of code:

forest_prediction = max(predictions,key=predictions.count)
Your conditional should look like this:

if forest_prediction == testing_labels[i]:
  forest_correct += 1
5.
Finally, after looping through all of the points in the test set, we want to print out the accuracy of our random forest. Divide forest_correct by the number of items in the test set and print the result.

How did the random forest do compared to the single decision tree?

Finish the line of code:

print(____/len(testing_data))'''

#----------script.py-------------------------------
from tree import training_data, training_labels, testing_data, testing_labels, make_random_forest, make_single_tree, classify
import numpy as np
import random
np.random.seed(1)
random.seed(1)

tree = make_single_tree(training_data, training_labels)
forest = make_random_forest (40, training_data, training_labels)
single_tree_correct = 0
forest_correct = 0

for i in range(len(testing_data)):
  
  prediction = classify(testing_data[i], tree)
  if prediction == testing_labels[i]:
    single_tree_correct += 1
  
  predictions = []

  for forest_tree in forest:    
    predictions.append(classify(testing_data[i], forest_tree))
  forest_prediction = max(predictions,key=predictions.count)
  if forest_prediction == testing_labels[i]:
    forest_correct += 1 





print(single_tree_correct/len(testing_data))
print(forest_correct/len(testing_data))

#-----------------------tree.py-------------------------------

import operator
from collections import Counter
import random
import numpy as np
np.random.seed(1)
random.seed(1)

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset)/len(starting_labels)
  return info_gain

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value

class Internal_Node:
    def __init__(self,
                 feature,
                 branches,
                 value):
        self.feature = feature
        self.branches = branches
        self.value = value

def find_best_split_subset(dataset, labels, num_features):
    features = np.random.choice(6, 3, replace=False)
    best_gain = 0
    best_feature = 0
    for feature in features:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def make_single_tree(data, labels, value = ""):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain < 0.00000001:
    return Leaf(Counter(labels), value)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branch = make_single_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
    branches.append(branch)
  return Internal_Node(best_feature, branches, value)

def build_tree_forest(data,labels, n_features, value=""):
    best_feature, best_gain = find_best_split_subset(data, labels, n_features)
    if best_gain < 0.00000001:
      return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
      branch = build_tree_forest(data_subsets[i], label_subsets[i], n_features, data_subsets[i][0][best_feature])
      branches.append(branch)
    return Internal_Node(best_feature, branches, value)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1:"Price of maintenance", 2:"Number of doors", 3:"Person Capacity", 4:"Size of luggage boot", 5:"Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + str(node.labels))
        return

    # Print the question at this node
    print (spacing + "Splitting on " + question_dict[node.feature])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print (spacing + '--> Branch ' + node.branches[i].value+':')
        print_tree(node.branches[i], spacing + "  ")

def make_cars():
    f = open("car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars

def change_data(data):
    dicts = [{'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'2' : 1.0, '3' : 2.0, '4' : 3.0, '5more' : 4.0},
    {'2' : 1.0, '4' : 2.0, 'more' : 3.0},
    {'small' : 1.0, 'med' : 2.0, 'big' : 3.0},
    {'low' : 1.0, 'med' : 2.0, 'high' : 3.0}]

    for row in data:
        for i in range(len(dicts)):
            row[i] = dicts[i][row[i]]

    return data


def classify(datapoint, tree):
  if isinstance(tree, Leaf):
    items = list(tree.labels.items()) 
    items.sort()
    return max(items, key=operator.itemgetter(1))[0]

  value = datapoint[tree.feature]
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)
  #return classify(datapoint, tree.branches[random.randint(0, len(tree.branches)-1)])


cars = make_cars()
random.shuffle(cars)
car_data = [x[:-1] for x in cars]
car_labels = [x[-1] for x in cars]
# car_data = car_data[:500]
# car_labels = car_labels[:500]


training_data = car_data[:int(len(car_data)*0.8)]
training_labels = car_labels[:int(len(car_data)*0.8)]

testing_data = car_data[int(len(car_data)*0.8):]
testing_labels = car_labels[int(len(car_data)*0.8):]

def make_random_forest(n, training_data, training_labels):
    trees = []
    for i in range(n):
        indices = [random.randint(0, len(training_data)-1) for x in range(len(training_data))]

        training_data_subset = [training_data[index] for index in indices]
        training_labels_subset = [training_labels[index] for index in indices]

        tree = build_tree_forest(training_data_subset, training_labels_subset, 2)
        trees.append(tree)
    return trees
    
'''
RANDOM FORESTS
Random Forest in Scikit-learn
You now have the ability to make a random forest using your own decision trees. However, scikit-learn has a RandomForestClassifier class that will do all of this work for you! RandomForestClassifier is in the sklearn.ensemble module.

RandomForestClassifier works almost identically to DecisionTreeClassifier — the .fit(), .predict(), and .score() methods work in the exact same way.

When creating a RandomForestClassifier, you can choose how many trees to include in the random forest by using the n_estimators parameter like this:

classifier = RandomForestClassifier(n_estimators = 100)
We now have a very powerful machine learning model that is fairly resistant to overfitting!'''
'''
Instructions
1.
Create a RandomForestClassifier named classifier. When you create it, pass two parameters to the constructor:

n_estimators should be 2000. Our forest will be pretty big!
random_state should be 0. There’s an element of randomness when creating random forests thanks to bagging. Setting the random_state to 0 will help us test your code.
classifier = RandomForestClassifier(random_state = ___, n_estimators = ___)
2.
Train the forest using the training data by calling the .fit() method. .fit() takes two parameters — training_points and training_labels.

Fill in the correct parameters:

classifier.fit(____, ____)
3.
Test the random forest on the testing set and print the results. How accurate was the model?

Call .score() using testing_points and testing_labels as parameters.'''

#------------------script.py-------------------------------

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from cars import training_points, training_labels, testing_points, testing_labels
import warnings
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier (random_state = 0, n_estimators = 2000)

classifier.fit(training_points, training_labels)

score = classifier.score(testing_points, testing_labels )
print(score)

#----------------cars.py-------------------------------

import random
random.seed(1)

def make_cars():
    f = open("car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars
  
def change_data(data):
    dicts = [{'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'vhigh' : 1.0, 'high' : 2.0, 'med' : 3.0, 'low' : 4.0},
    {'2' : 1.0, '3' : 2.0, '4' : 3.0, '5more' : 4.0},
    {'2' : 1.0, '4' : 2.0, 'more' : 3.0},
    {'small' : 1.0, 'med' : 2.0, 'big' : 3.0},
    {'low' : 1.0, 'med' : 2.0, 'high' : 3.0}]

    for row in data:
        for i in range(len(dicts)):
            row[i] = dicts[i][row[i]]

    return data
  
cars = change_data(make_cars())
random.shuffle(cars)
car_data = [x[:-1] for x in cars]
car_labels = [x[-1] for x in cars]

training_points = car_data[:int(len(car_data)*0.9)]
training_labels = car_labels[:int(len(car_labels)*0.9)]

testing_points = car_data[int(len(car_data)*0.9):]
testing_labels = car_labels[int(len(car_labels)*0.9):]

'''
RANDOM FORESTS
Review
Nice work! Here are some of the major takeaways about random forests:

A random forest is an ensemble machine learning model. It makes a classification by aggregating the classifications of many decision trees.
Random forests are used to avoid overfitting. By aggregating the classification of multiple trees, having overfitted trees in a random forest is less impactful.
Every decision tree in a random forest is created by using a different subset of data points from the training set. Those data points are chosen at random with replacement, which means a single data point can be chosen more than once. This process is known as bagging.
When creating a tree in a random forest, a randomly selected subset of features are considered as candidates for the best splitting feature. If your dataset has n features, it is common practice to randomly select the square root of n features.'''
'''
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












  
  
































  
  
  
















