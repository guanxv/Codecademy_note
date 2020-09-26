
# CATEGORICAL VARIABLES IN REGRESSION ANALYSIS 

# reference https://stats.idre.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis-2/#:~:text=Categorical%20variables%20require%20special%20attention,entered%20into%20the%20regression%20model.

# dummy coding

#sample data 
# RACE          Mean	  N
# hispanic      46.4583	24
# asian	        58.0000	11
# african-amer	48.2000	20
# white	        54.0552	145
# Total	        52.7750	200

# dummy coding new variable 

# RACE          is_hispanic   is_asian    is_african-amer
# hispanic      1             0           0 
# asian	        0             1           0
# african-amer	0             0           1
# white	        0             0           0












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