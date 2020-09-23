



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-PYTHON NUMPY STATISTIC PYTHON NUMPY STATISTIC PYTHON NUMPY STATISTIC PYTHON NUMPY STATISTIC #-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#INTRODUCTION TO STATISTICS WITH NUMPY 3/13

import numpy as np

class_year = np.array([1967, 1949, 2004, 1997, 1953, 1950, 1958, 1974, 1987, 2006, 2013, 1978, 1951, 1998, 1996, 1952, 2005, 2007, 2003, 1955, 1963, 1978, 2001, 2012, 2014, 1948, 1970, 2011, 1962, 1966, 1978, 1988, 2006, 1971, 1994, 1978, 1977, 1960, 2008, 1965, 1990, 2011, 1962, 1995, 2004, 1991, 1952, 2013, 1983, 1955, 1957, 1947, 1994, 1978, 1957, 2016, 1969, 1996, 1958, 1994, 1958, 2008, 1988, 1977, 1991, 1997, 2009, 1976, 1999, 1975, 1949, 1985, 2001, 1952, 1953, 1949, 2015, 2006, 1996, 2015, 2009, 1949, 2004, 2010, 2011, 2001, 1998, 1967, 1994, 1966, 1994, 1986, 1963, 1954, 1963, 1987, 1992, 2008, 1979, 1987])

millennials = np.mean(class_year >= 2005 ) 

print(millennials )

#INTRODUCTION TO STATISTICS WITH NUMPY 4/13

import numpy as np

allergy_trials = np.array([[6, 1, 3, 8, 2], 
                           [2, 6, 3, 9, 8], 
                           [5, 2, 6, 9, 9]])

total_mean = np.mean(allergy_trials)


trial_mean = np.mean(allergy_trials, axis = 1)

patient_mean = np.mean(allergy_trials, axis = 0)

print(total_mean)
print(trial_mean)
print(patient_mean)

#INTRODUCTION TO STATISTICS WITH NUMPY 7/13

import numpy as np

large_set = np.genfromtxt('household_income.csv', delimiter=',')

small_set = [10100, 35500, 105000, 85000, 25500, 40500, 65000]

small_set.sort()
small_set_median = small_set[(len(small_set)-1)/2]

large_set_median = np.median(large_set)

print(small_set_median)
print(large_set_median)

#INTRODUCTION TO STATISTICS WITH NUMPY 8/13

import numpy as np

time_spent = np.genfromtxt('file.csv', delimiter=',')

print(time_spent)

minutes_mean = np.mean(time_spent)

minutes_more_than_2 = np.mean(time_spent > 8)

minutes_median = np.median(time_spent)

print(minutes_mean)
print(minutes_median)
print(minutes_more_than_2)

best_measure = minutes_median

#INTRODUCTION TO STATISTICS WITH NUMPY
#Percentiles, Part II
Some percentiles have specific names:

The 25th percentile is called the first quartile
The 50th percentile is called the median
The 75th percentile is called the third quartile
The minimum, first quartile, median, third quartile, and maximum of a dataset are called a five-number summary. This set of numbers is a great thing to compute when we get a new dataset.

The difference between the first and third quartile is a value called the interquartile range. For example, say we have the following array:

d = [1, 2, 3, 4, 4, 4, 6, 6, 7, 8, 8]
We can calculate the 25th and 75th percentiles using np.percentile:

np.percentile(d, 25)
>>> 3.5
np.percentile(d, 75)
>>> 6.5
Then to find the interquartile range, we subtract the value of the 25th percentile from the value of the 75th:

6.5 - 3.5 = 3
50% of the dataset will lie within the interquartile range. The interquartile range gives us an idea of how spread out our data is. The smaller the interquartile range value, the less variance in our dataset. The greater the value, the larger the variance.

#INTRODUCTION TO STATISTICS WITH NUMPY 13/13

import numpy as np

rainfall = np.array([5.21, 3.76, 3.27, 2.35, 1.89, 1.55, 0.65, 1.06, 1.72, 3.35, 4.82, 5.11])


rain_mean = np.mean(rainfall)

rain_median = np.median(rainfall)

first_quarter = np.percentile(rainfall, 25)

third_quarter = np.percentile(rainfall, 75)

interquartile_range = third_quarter - first_quarter

rain_std = np.std(rainfall)

print(rainfall)
print(rain_mean)
print(rain_median)
print(first_quarter)
print(third_quarter)
print(interquartile_range)
print(rain_std)


#------------------------------------------------------------------------------

import codecademylib

import numpy as np

calorie_stats = np.genfromtxt('cereal.csv',delimiter = ",")

average_calories = np.mean(calorie_stats)

print(average_calories)

calorie_stats_sorted = sorted(calorie_stats)

print (calorie_stats)

print (calorie_stats_sorted)

median_calories = np.median(calorie_stats)

print("\n")
print (median_calories)

nth_percentile = np.percentile(calorie_stats, 3.8961)

print("\n")
print nth_percentile

more_calories = np.mean (calorie_stats > 60)

print (more_calories)

more_calories = np.std(calorie_stats)
print (more_calories)

#------------------------------------------------------------------------------


import codecademylib
import numpy as np
from matplotlib import pyplot as plt

# Brachiosaurus
b_data = np.random.normal(6.7,0.7,size = 1000)

# Fictionosaurus
f_data = np.random.normal(7.7,0.3,size = 1000)

plt.hist(b_data,
         bins=30, range=(5, 8.5), histtype='step',
         label='Brachiosaurus')
plt.hist(f_data,
         bins=30, range=(5, 8.5), histtype='step',
         label='Fictionosaurus')
plt.xlabel('Femur Length (ft)')
plt.legend(loc=2)
plt.show()

#--------------------------------------------
STATISTICAL DISTRIBUTIONS WITH NUMPY
Review
Let’s review! In this lesson, you learned how to use NumPy to analyze different distributions and generate random numbers to produce datasets. Here’s what we covered:

What is a histogram and how to map one using Matplotlib
How to identify different dataset shapes, depending on peaks or distribution of data
The definition of a normal distribution and how to use NumPy to generate one using NumPy’s random number functions
The relationships between normal distributions and standard deviations
The definition of a binomial distribution
Now you can use NumPy to analyze and graph your own datasets! You should practice building your intuition about not only what the data says, but what conclusions can be drawn from your observations.

Instructions
1.
Practice what you’ve just learned with a dataset on sunflower heights! Imagine that you work for a botanical garden and you want to see how the sunflowers you planted last year did to see if you want to plant more of them.

Calculate the mean and standard deviation of this dataset. Save the mean to sunflowers_mean and the standard deviation to sunflowers_std.

2.
We can see from the histogram that our data isn’t normally distributed. Let’s create a normally distributed sample to compare against what we observed.

Generate 5,000 random samples with the same mean and standard deviation as sunflowers. Save these to sunflowers_normal.

3.
Now that you generated sunflowers_normal, uncomment (remove all of the #) the second plt.hist statement. Press run to see your normal distribution and your observed distribution.

4.
Generally, 10% of sunflowers that are planted fail to bloom. We planted 200, and want to know the probability that fewer than 20 will fail to bloom.

First, generate 5,000 binomial random numbers that represent our situation. Save them to experiments.

5.
What percent of experiments had fewer than 20 sunflowers fail to bloom?

Save your answer to the variable prob. This is the approximate probability that fewer than 20 of our sunflowers will fail to bloom.

6.
Print prob. Is it likely that fewer than 20 of our sunflowers will fail to bloom?
#--------------------------------------------

import codecademylib
import numpy as np
from matplotlib import pyplot as plt

sunflowers = np.genfromtxt('sunflower_heights.csv',
                           delimiter=',')

# Calculate mean and std of sunflowers here:
sunflowers_mean = np.mean(sunflowers)
sunflowers_std = np.std(sunflowers)

print (sunflowers_mean)
print (sunflowers_std)

# Calculate sunflowers_normal here:
sunflowers_normal = np.random.normal(sunflowers_mean,sunflowers_std,5000)

plt.hist(sunflowers,
         range=(11, 15), histtype='step', linewidth=2,
        label='observed', normed=True)
plt.hist(sunflowers_normal,
         range=(11, 15), histtype='step', linewidth=2,
        label='normal', normed=True)
plt.legend()
plt.show()

# Calculate probabilities here:
experiments = np.random.binomial(200,0.1,5000)

prob = np.mean(experiments<20)

print("\n")
print (prob)

#-----------------------------------------------------------------
NUMPY: A PYTHON LIBRARY FOR STATISTICS
Election Results
You’re part of an impartial research group that conducts phone surveys prior to local elections. During this election season, the group conducted a survey to determine how many people would vote for Cynthia Ceballos vs. Justin Kerrigan in the mayoral election.

Now that the election has occurred, your group wants to compare the survey responses to the actual results.

Was your survey a good indicator? Let’s find out!

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
7/9Complete
Mark the tasks as complete by checking them off
PROJECT STEPS
1.
First, import numpy and matplotlib.

2.
At the top of script.py is a list of the different survey responses.

Calculate the number of people who answered ‘Ceballos’ and save the answer to the variable total_ceballos.

Print the variable to the terminal to see its value.

total = sum([1 for n in list if n == foo])
3.
Calculate the percentage of people in the survey who voted for Ceballos and save it to the variable percentage_ceballos.

Print the variable to the terminal to see its value.

percentage = 100 * total/len(list)
4.
In the real election, 54% of the 10,000 town population voted for Cynthia Ceballos. Your supervisors are concerned because this is a very different outcome than what the poll predicted. They want you to determine if there is something wrong with the poll or if given the sample size, it was an entirely reasonable result.

Generate a binomial distribution that takes the number of total survey responses, the actual success rate, and the size of the town’s population as its parameters. Then divide the distribution by the number of survey responses. Save your calculation to the variable possible_surveys.

a = np.random.binomial(N, P, size=10000) / N.
Where N is the number of trials, P is the probability of success, and size is the total population. Also notice the period at the end of the line of code above. The period ensures that you are dividing each element in a by a float.

If you do not include the period, Python will assume you want integer division (an integer divided by an integer). Naturally, integer division returns an integer. For quotients less than 1 (for example, 7 divided by 18), this is problematic. Python will return a 0 (as opposed to a decimal, like 0.388), which can result in erroneous calculations.

5.
Plot a histogram of possible_surveys with a range of 0-1 and 20 bins.

plt.hist(array, range=(a, b), bins=n)
6.
As we saw, 47% of people we surveyed said they would vote for Ceballos, but 54% of people voted for Ceballos in the actual election.

Calculate the percentage of surveys that could have an outcome of Ceballos receiving less than 50% of the vote and save it to the variable ceballos_loss_surveys.

Print the variable to the terminal.

np.mean(array < 0.5)
7.
With this current poll, about 20% of the time a survey output would predict Kerrigan winning, even if Ceballos won the actual election.

Your co-worker points out that your poll would be more accurate if it had more responders.

Generate another binomial distribution, but this time, see what would happen if you had instead surveyed 7,000 people. Divide the distribution by the size of the survey and save your findings to large_survey.

8.
Now, recalculate the percentage of surveys that would have an outcome of Ceballos losing and save it to the variable ceballos_loss_new, and print the value to the terminal.

What do we notice about this new value?

What advice would you give to your supervisors about predicting results from surveys?

9.
Click here for a video walkthrough from our experts to help you check your work!
#--------------------------------------------------------------------


import codecademylib
import numpy as np
from matplotlib import pyplot as plt


survey_responses = ['Ceballos', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos','Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 
'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 'Ceballos', 'Ceballos', 'Ceballos',
'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Ceballos',
'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Ceballos']

#total_ceballos = survey_responses.count("Ceballos")
total_ceballos = sum([1 for response in survey_responses if response == "Ceballos"])
servey_length = float(len(survey_responses))
#percentage_ceballos = total_ceballos / 70.
percentage_ceballos = total_ceballos / servey_length
print total_ceballos
print percentage_ceballos

possible_surveys = np.random.binomial(servey_length,.54,size = 10000) / servey_length

plt.hist(possible_surveys, bins = 20, range = (0,1))
plt.show()

#Codecademy answer
possible_serveys_length = float(len(possible_surveys))
incorrect_predicitons = len(possible_surveys[possible_surveys < 0.5 ])
ceballos_loss_surveys = incorrect_predicitons / possible_serveys_length
print ceballos_loss_surveys

#Xu's answer
ceballos_loss_surveys = np.mean(possible_surveys < 0.5)
print (ceballos_loss_surveys)

#Codecademy answer
large_survey_length = float (7000)
large_survey = np.random.binomial(large_survey_length, 0.54, size = 10000) / large_survey_length

incorrect_predictions = len(large_survey[large_survey < 0.5])
ceballos_loss_new = incorrect_predictions / large_survey_length
#Xu's answer
large_survey = np.random.binomial(7000, 0.54, size = 10000) / 7000.

ceballos_loss_new = np.mean(large_survey < 0.5)

print (ceballos_loss_new)


#Learn Statistics With Python

#get average / mean of a list / series
np.average(list)

'''Median
The formal definition for the median of a dataset is:

The value that, assuming the dataset is ordered from smallest to largest, falls in the middle. If there are an even number of values in a dataset, you either report both of the middle two values or their average.'''

#np.sort()----------------

import numpy as np

# Array of the first five author ages
five_author_ages = np.array([29, 49, 42, 43, 32])

# Fill in the empty array with the values sorted
sorted_author_ages = np.sort(five_author_ages)

# Save the median value to median_value
median_age = np.median(sorted_author_ages)

# Print the sorted array and median value
print("The sorted array is: " + str(sorted_author_ages))
print("The median of the array is: " + str(median_age))


'''
Mode
The formal definition for the mode of a dataset is:

The most frequently occurring observation in the dataset. A dataset can have multiple modes if there is more than one value with the same maximum frequency.'''

from scipy import stats

example_array = np.array([24, 16, 12, 10, 12, 28, 38, 12, 28, 24])

example_mode = stats.mode(example_array)

'''
The code above calculates the mode of the values in example_array and saves it to example_mode.

The result of stats.mode() is an object with the mode value, and its count.

>>> example_mode
ModeResult(mode=array([12]), count=array([3]))


If there are multiple modes, the stats.mode() function will always return the smallest mode in the dataset.

Let’s look at an array with two modes, 12 and 24:
'''
from scipy import stats

example_array = np.array([24, 16, 12, 10, 12, 24, 38, 12, 28, 24])

example_mode = stats.mode(example_array)
'''
The result of stats.mode() is an object with the smallest mode value, and its count.

>>> example_mode
ModeResult(mode=array([12]), count=array([3]))'''


#review

# Import packages
import codecademylib
import numpy as np
import pandas as pd
from scipy import stats

# Import matplotlib pyplot
from matplotlib import pyplot as plt

# Read in transactions data
greatest_books = pd.read_csv("top-hundred-books.csv")

# Save transaction times to a separate numpy array
author_ages = greatest_books['Ages']

# Calculate the average and median value of the author_ages array
average_age = np.average(author_ages)
median_age = np.median(author_ages)
mode_age = 38

# Plot the figure
plt.hist(author_ages, range=(10, 80), bins=14,  edgecolor='black')
plt.title("Author Ages at Publication")
plt.xlabel("Publication Age")
plt.ylabel("Count")
plt.axvline(average_age, color='r', linestyle='solid', linewidth=3, label="Mean")
plt.axvline(median_age, color='y', linestyle='dotted', linewidth=3, label="Median")
plt.axvline(mode_age, color='orange', linestyle='dashed', linewidth=3, label="Mode")
plt.legend()

plt.show()

#--------------------------------------


# Import packages
import numpy as np
import pandas as pd
from scipy import stats

# Read in housing data
brooklyn_one_bed = pd.read_csv('brooklyn-one-bed.csv')
brooklyn_price = brooklyn_one_bed['rent']

manhattan_one_bed = pd.read_csv('manhattan-one-bed.csv')
manhattan_price = manhattan_one_bed['rent']

queens_one_bed = pd.read_csv('queens-one-bed.csv')
queens_price = queens_one_bed['rent']

# Add mean calculations below 
brooklyn_mean = np.mean(brooklyn_price)

manhattan_mean = np.mean(manhattan_price)

queens_mean = np.mean(queens_price)


# Add median calculations below
brooklyn_median = np.median(brooklyn_price)

manhattan_median = np.median(manhattan_price)

queens_median = np.median(queens_price)


# Add mode calculations below
brooklyn_mode = stats.mode(brooklyn_price)

manhattan_mode = stats.mode(manhattan_price)

queens_mode = stats.mode(queens_price)


# Don't look below here
# Mean
try:
    print("The mean price in Brooklyn is " + str(round(brooklyn_mean, 2)))
except NameError:
    print("The mean price in Brooklyn is not yet defined.")
try:
    print("The mean price in Manhattan is " + str(round(manhattan_mean, 2)))
except NameError:
    print("The mean in Manhattan is not yet defined.")
try:
    print("The mean price in Queens is " + str(round(queens_mean, 2)))
except NameError:
    print("The mean price in Queens is not yet defined.")
    
    
# Median
try:
    print("The median price in Brooklyn is " + str(brooklyn_median))
except NameError:
    print("The median price in Brooklyn is not yet defined.")
try:
    print("The median price in Manhattan is " + str(manhattan_median))
except NameError:
    print("The median price in Manhattan is not yet defined.")
try:
    print("The median price in Queens is " + str(queens_median))
except NameError:
    print("The median price in Queens is not yet defined.")
    
    
#Mode
try:
    print("The mode price in Brooklyn is " + str(brooklyn_mode[0][0]) + " and it appears " + str(brooklyn_mode[1][0]) + " times out of " + str(len(brooklyn_price)))
except NameError:
    print("The mode price in Brooklyn is not yet defined.")
try:
    print("The mode price in Manhattan is " + str(manhattan_mode[0][0]) + " and it appears " + str(manhattan_mode[1][0]) + " times out of " + str(len(manhattan_price)))
except NameError:
    print("The mode price in Manhattan is not yet defined.")
try:
    print("The mode price in Queens is " + str(queens_mode[0][0]) + " and it appears " + str(queens_mode[1][0]) + " times out of " + str(len(queens_price)))
except NameError:
    print("The mode price in Queens is not yet defined.")
    
    

'''VARIANCE
Variance
Finding the mean, median, and mode of a dataset is a good way to start getting an understanding of the general shape of your data

However, those three descriptive statistics only tell part of the story. Consider the two datasets below:

dataset_one = [-4, -2, 0, 2, 4]
dataset_two = [-400, -200, 0, 200, 400]
These two datasets have the same mean and median — both of those values happen to be 0. If we only reported these two statistics, we would not be communicating any meaninful difference between these two datasets.

This is where variance comes into play. Variance is a descriptive statistic that describes how spread out the points in a data set are.'''

'''
VARIANCE
Distance From Mean
Now that you have learned the importance of describing the spread of a dataset, let’s figure out how to mathematically compute this number.

How would you attempt to capture the spread of the data in a single number?

Let’s start with our intuition — we want the variance of a dataset to be a large number if the data is spread out, and a small number if the data is close together.

Two histograms. One with a large spread and one with a smaller spread.
A lot of people may initially consider using the range of the data. But that only considers two points in your entire dataset. Instead, we can include every point in our calculation by finding the difference between every data point and the mean.

The difference between the mean and four different points.
If the data is close together, then each data point will tend to be close to the mean, and the difference will be small. If the data is spread out, the difference between every data point and the mean will be larger.

Mathematically, we can write this comparison as

\text{difference} = X - \mudifference=X−μ
Where X is a single data point and the Greek letter mu is the mean.

VARIANCE
Average Distances
We now have five different values that describe how far away each point is from the mean. That seems to be a good start in describing the spread of the data. But the whole point of calculating variance was to get one number that describes the dataset. We don’t want to report five values — we want to combine those into one descriptive statistic.

To do this, we’ll take the average of those five numbers. By adding those numbers together and dividing by 5, we’ll end up with a single number that describes the average distance between our data points and the mean.

Note that we’re not quite done yet — our final answer is going to look a bit strange here. There’s a small problem that we’ll fix in the next exercise.'''

'''

VARIANCE
Square The Differences
We’re almost there! We have one small problem with our equation. Consider this very small dataset:

[-5, 5]
The mean of this dataset is 0, so when we find the difference between each point and the mean we get -5 - 0 = -5 and 5 - 0 = 5.

When we take the average of -5 and 5 to get the variance, we get 0!

Now think about what would happen if the dataset were [-200, 200]. We’d get the same result! That can’t possibly be right — the dataset with 200 is much more spread out than the dataset with 5, so the variance should be much larger!

The problem here is with negative numbers. Because one of our data points was 5 units below the mean and the other was 5 units above the mean, they canceled each other out!

When calculating variance, we don’t care if a data point was above or below the mean — all we care about is how far away it was. To get rid of those pesky negative numbers, we’ll square the difference between each data point and the mean.

Our equation for finding the difference between a data point and the mean now looks like this:

difference = (X - mu)^2

'''

import numpy as np

grades = [88, 82, 85, 84, 90]
mean = np.mean(grades)

#When calculating these variables, square the difference.
difference_one = (88 - mean)**2
difference_two = (82 - mean)**2
difference_three = (85 - mean)**2
difference_four = (84 - mean)**2
difference_five = (90 - mean)**2

difference_sum = difference_one + difference_two + difference_three + difference_four + difference_five

variance = difference_sum / 5

print("The sum of the squared differences is " + str(difference_sum))
print("The variance is " + str(variance))

'''

VARIANCE
Variance In NumPy
Well done! You’ve calculated the variance of a data set. The full equation for the variance is as follows:

 
​σ^2 = ∑ (N i=1) (Xi - mu )^2 / N

​	 
Let’s dissect this equation a bit.

Variance is usually represented by the symbol sigma squared.
We start by taking every point in the dataset — from point number 1 to point number N — and finding the difference between that point and the mean.
Next, we square each difference to make all differences positive.
Finally, we average those squared differences by adding them together and dividing by N, the total number of points in the dataset.
All of this work can be done quickly using Python’s NumPy library. The var() function takes a list of numbers as a parameter and returns the variance of that dataset.
'''
import numpy as np

dataset = [3, 5, -2, 49, 10]
variance = np.var(dataset)

'''
STANDARD DEVIATION
Variance Recap
When beginning to work with a dataset, one of the first pieces of information you might want to investigate is the spread — is the data close together or far apart? One of the tools in our statistics toolbelt to do this is the descriptive statistic variance:

        N
      ∑ i=1 (Xi - mu )^2
​σ^2 = --------------------
               N

​	 
By finding the variance of a dataset, we can get a numeric representation of the spread of the data. If you want to take a deeper dive into how to calculate variance, check out our variance course.

But what does that number really mean? How can we use this number to interpret the spread?

It turns out, using variance isn’t necessarily the best statistic to use to describe spread. Luckily, there is another statistic — standard deviation — that can be used instead.

In this lesson, we’ll be working with two datasets. The first dataset contains the heights (in inches) of a random selection of players from the NBA. The second dataset contains the heights (in inches) of a random selection of users on the dating platform OkCupid — let’s hope these users were telling the truth about their height!

Instructions
1.
Run the code to see a histogram of the datasets. Look at the console to see the mean and variance of each dataset. Try to answer the following questions:

What does it mean for the OkCupid dataset to have a larger variance than the NBA dataset?
What are the units of the mean? Is someone who is 80 inches tall taller than the average of either group? Which group(s)?
In this example, the units of variance are inches squared. Can you interpret what it means for the variance of the NBA dataset to be 13.32 inches squared?



STANDARD DEVIATION
Standard Deviation
Variance is a tricky statistic to use because its units are different from both the mean and the data itself. For example, the mean of our NBA dataset is 77.98 inches. Because of this, we can say someone who is 80 inches tall is about two inches taller than the average NBA player.

However, because the formula for variance includes squaring the difference between the data and the mean, the variance is measured in units squared. This means that the variance for our NBA dataset is 13.32 inches squared.

This result is hard to interpret in context with the mean or the data because their units are different. This is where the statistic standard deviation is useful.

Standard deviation is computed by taking the square root of the variance. sigma is the symbol commonly used for standard deviation. Conveniently, sigma squared is the symbol commonly used for variance:


σ  = sqrt(sigma ^2)  = sqrt(∑ (N i=1) (Xi - mu )^2 / N)

STANDARD DEVIATION
Standard Deviation in NumPy
There is a NumPy function dedicated to finding the standard deviation of a dataset — we can cut out the step of first finding the variance. The NumPy function std() takes a dataset as a parameter and returns the standard deviation of that dataset:
'''
import numpy as np

dataset = [4, 8, 15, 16, 23, 42]
standard_deviation = np.std()

'''

STANDARD DEVIATION
Using Standard Deviation
Now that we’re able to compute the standard deviation of a dataset, what can we do with it?

Now that our units match, our measure of spread is easier to interpret. By finding the number of standard deviations a data point is away from the mean, we can begin to investigate how unusual that datapoint truly is. In fact, you can usually expect around 68% of your data to fall within one standard deviation of the mean, 95% of your data to fall within two standard deviations of the mean, and 99.7% of your data to fall within three standard deviations of the mean.

A histogram showing where the standard deviations fallIf you have a data point that is over three standard deviations away from the mean, that's an incredibly unusual piece of data!


STANDARD DEVIATION
Review
In the last exercise you saw that Lebron James was 0.55 standard deviations above the mean of NBA player heights. He’s taller than average, but compared to the other NBA players, he’s not absurdly tall.

However, compared to the OkCupid dating pool, he is extremely rare! He’s almost three full standard deviations above the mean. You’d expect only about 0.15% of people on OkCupid to be more than 3 standard deviations away from the mean.

This is the power of standard deviation. By taking the square root of the variance, the standard deviation gives you a statistic about spread that can be easily interpreted and compared to the mean.

'''
import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from data import nba_data, okcupid_data

nba_mean = np.mean(nba_data)
okcupid_mean = np.mean(okcupid_data)

#Change this variable to your height (in inches)!
your_height = 0

nba_standard_deviation = np.std(nba_data)
okcupid_standard_deviation = np.std(okcupid_data)

plt.subplot(211)
plt.title("NBA Player Heights")
plt.xlabel("Height (inches)")

plt.hist(nba_data)

plt.axvline(nba_mean, color='#FD4E40', linestyle='solid', linewidth=2, label = "Mean")

plt.axvline(nba_mean + nba_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2, label = "Standard Deviations")
plt.axvline(nba_mean - nba_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(nba_mean + nba_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(nba_mean - nba_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(nba_mean + nba_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(nba_mean - nba_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(your_height, color='#62EDBF', linestyle='solid', linewidth=2, label = "You")

plt.xlim(55, 90)
plt.legend()


plt.subplot(212)
plt.title("OkCupid Profile Heights")
plt.xlabel("Height (inches)")

plt.hist(okcupid_data)

plt.axvline(okcupid_mean, color='#FD4E40', linestyle='solid', linewidth=2, label = "Mean")

plt.axvline(okcupid_mean + okcupid_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2, label = "Standard Deviations")
plt.axvline(okcupid_mean - okcupid_standard_deviation, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(okcupid_mean + okcupid_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(okcupid_mean - okcupid_standard_deviation * 2, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(okcupid_mean + okcupid_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)
plt.axvline(okcupid_mean - okcupid_standard_deviation * 3, color='#FFB908', linestyle='solid', linewidth=2)

plt.axvline(your_height, color='#62EDBF', linestyle='solid', linewidth=2, label = "You")

plt.xlim(55, 90)
plt.legend()




plt.tight_layout()
plt.show()


'''
LEARN STATISTICS WITH PYTHON
Variance in Weather
You’re planning a trip to London and want to get a sense of the best time of the year to visit. Luckily, you got your hands on a dataset from 2015 that contains over 39,000 data points about weather conditions in London. Surely, with this much information, you can discover something useful about when to make your trip!

In this project, the data is stored in a Pandas DataFrame. If you’ve never used a DataFrame before, we’ll walk you through how to filter and manipulate this data. If you want to learn more about Pandas, check out our Data Science Path.

'''

import codecademylib3_seaborn
import pandas as pd
import numpy as np
from weather_data import london_data

#print(type(london_data))
#print(london_data.head())
#print(london_data.columns)
#print(london_data.loc[100:200]['station'])
#print(london_data.iloc[100:200]['station'])
#print(len(london_data))

temp = london_data["TemperatureC"]
average_temp = np.mean(temp)
temperature_var = np.var(temp)
temperature_standard_deviation = np.std(temp)

june = london_data.loc[london_data['month'] == 6]['TemperatureC']
july = london_data.loc[london_data['month'] == 7]['TemperatureC']

june_temp_mean = np.mean(june)
july_temp_mean = np.mean(july)

#print(june_temp_mean)
#print(july_temp_mean)

#print(np.std(june))
#print(np.std(july))

for i in range(1, 13):
  month = london_data.loc[london_data["month"] == i]["TemperatureC"]
  print("The mean temperature in month "+str(i) +" is "+ str(np.mean(month)))
  print("The standard deviation of temperature in month "+str(i) +" is "+ str(np.std(month)) +"\n")


'''

HISTOGRAMS
Histograms
While counting the number of values in a bin is straightforward, it is also time-consuming. How long do you think it would take you to count the number of values in each bin for:

an exercise class of 50 people?
a grocery store with 300 loaves of bread?
Most of the data you will analyze with histograms includes far more than ten values.

For these situations, we can use the numpy.histogram() function. In the example below, we use this function to find the counts for a twenty-person exercise class.
'''

exercise_ages = np.array([22, 27, 45, 62, 34, 52, 42, 22, 34, 26, 24, 65, 34, 25, 45, 23, 45, 33, 52, 55])

np.histogram(exercise_ages, range = (20, 70), bins = 5)

'''
Below, we explain each of the function’s inputs:

exercise_ages is the input array
range = (20, 70) — is the range of values we expect in our array. Range includes everything from 20, up until but not including 70.
bins = 5 is the number of bins. Python will automatically calculate equally-sized bins based on the range and number of bins.
Below, you can see the output of the numpy.histogram() function:

(array([7, 4, 4, 3, 2]), array([20., 30., 40., 50., 60., 70.]))
The first array, array([7, 4, 4, 3, 2]), is the counts for each bin. The second array, array([20., 30., 40., 50., 60., 70.]), includes the minimum and maximum values for each bin:

Bin 1: 20 to <30
Bin 2: 30 to <40
Bin 3: 40 to <50
Bin 4: 50 to <60
Bin 5: 60 to <70'''

'''
HISTOGRAMS
Plotting a Histogram
At this point, you’ve learned how to find the numerical inputs to a histogram. Thus far the size of our datasets and bins have produced results that we can interpret. This becomes increasingly difficult as the number of bins in a histogram increases.

Because of this, histograms are typically viewed graphically, with bin ranges on the x-axis and counts on the y-axis. The figure below shows the graphical representation of the histogram for our exercise class example from last exercise. Notice, there are five equally-spaced bars, with each displaying a count for an age range. Compare the graph to the table, just below it.

Histogram
20-29	30-39	40-49	50-59	60-69
7	4	4	3	2
Histograms are an easy way to visualize trends in your data. When I look at the above graph, I think, “More people in the exercise class are in their twenties than any other decade. Additionally, the histogram is skewed, indicating the class is made of more younger people than older people.”

We created the plot above using the matplotlib.pyplot package. We imported the package using the following code:
'''
from matplotlib import pyplot as plt

'''
We plotted the histogram with the following code. Notice, the range and bins arguments are the same as we used in the last exercise:
'''
plt.hist(exercise_ages, range = (20, 70), bins = 5, edgecolor='black')

plt.title("Decade Frequency")
plt.xlabel("Ages")
plt.ylabel("Count")

plt.show()

'''
In the code above, we used the plt.hist() function to create the plot, then added a title, x-label, and y-label before showing the graph with plt.show().'''

#-----------------------------------------

# Import packages
import numpy as np
import pandas as pd

# Read in transactions data
transactions = pd.read_csv("transactions.csv")

# Save transaction data to numpy arrays
times = transactions["Transaction Time"]
cost = transactions["Cost"]

# Find the minimum time, maximum time, and range
min_time = np.amin(times)
max_time = np.amax(times)
range_time = max_time - min_time

# Printing the values
print("Earliest Time: " + str(min_time))
print("Latest Time: " + str(max_time))
print("Time Range: " + str(range_time))

#----------------------------

# Import packages
import codecademylib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Read in transactions data
transactions = pd.read_csv("transactions.csv")

# Save transaction times to a separate numpy array
times = transactions["Transaction Time"].values

# Use plt.hist() below
plt.hist(times, range=(0, 24), bins=24,  edgecolor="black")
plt.title("Weekday Frequency of Customers")
plt.xlabel("Hours (1 hour increments)")
plt.ylabel("Count")

plt.show()


#----------------------------------------------------


'''
In this lesson, you will learn how to interpret a distribution using the following five features of a dataset:

Center
Spread
Skew
Modality
Outliers
If you’re one for mnemonics, maybe this will help:

Cream Shoes are Stylish, Modern, and Outstanding.
'''

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import codecademylib3_seaborn

cp_data = pd.read_csv("cp.csv") 

cp_average = cp_data[' Average Covered Charges '].mean()

cp_median = cp_data[' Average Covered Charges '].median()

plt.hist(cp_data[' Average Covered Charges '], bins=20, edgecolor='black')

plt.title("Distribution of Chest Pain Treatment Cost by Hospital", fontsize = 16)
plt.xlabel("Cost ($)", fontsize = 16)
plt.ylabel("Count", fontsize = 16)
plt.axvline(cp_average, color='r', linestyle='solid', linewidth=2, label="Mean")
plt.axvline(cp_median, color='y', linestyle='solid', linewidth=2, label="Median")
plt.legend()

plt.show()

'''
Skew
Once you have the center and range of your data, you can begin to describe its shape. The skew of a dataset is a description of the data’s symmetry.

A dataset with one prominent peak, and similar tails to the left and right is called symmetric. The median and mean of a symmetric dataset are similar.

A histogram with a tail that extends to the right is called a right-skewed dataset. The median of this dataset is less than the mean.

histogram

A histogram with one prominent peak to the right, and a tail that extends to the left is called a left-skewed dataset. The median of this dataset is greater than the mean.


Modality
The modality describes the number of peaks in a dataset. Thus far, we have only looked at datasets with one distinct peak, known as unimodal. This is the most common.

A bimodal dataset has two distinct peaks.

histogram

A multimodal dataset has more than two peaks. The histogram below displays three peaks.

histogram

You may also see datasets with no obvious clustering. Datasets such as these are called uniform distributions.

Outliers
An outlier is a data point that is far away from the rest of the dataset. Ouliers do not have a formal definition, but are easy to determine by looking at histogram. The histogram below shows an example of an oulier. There is one datapoint that is much larger than the rest.

title

If you see an outlier in your dataset, it’s worth reporting and investigating. This data can often indicate an error in your data or an interesting insight.'''



#-------------summary.txt ---------------

'''
This histogram displays the distribution of chest pain cost for over 2,000 hospitals across the United States. The average and median costs are $16,948 and $14,659.6, respectively. Given that the data is unimodal, with one local maximum and a right skew, the fact that the average is greater than the median, matches our expectation.

The range of costs is very large, $78,623, with the smallest cost equal to $2,459 and the largest cost equal to $81,083. There is one hospital, Bayonne Hospital Center that charges far more than the rest at $81,083.'''

#---------------script.py---------------

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import codecademylib3_seaborn


np.random.seed(1)

fig, axes = plt.subplots(nrows=4, ncols=1)

plt.subplot(4,1,1)
mu, sigma = 80,5
exam_1 = np.random.normal(mu, sigma, 120)
exam_1[50] = 55
exam_1[51] = 55
count, bins, ignored = plt.hist(exam_1, 25, range=[50, 100])
plt.ylabel("Count", fontsize=12)
plt.title("Exam 1", fontsize=14)




plt.subplot(4,1,2)
mu, sigma = 85,5
exam_2_norm = np.random.normal(mu, sigma, 85)
exam_2_u = np.random.uniform(60, 80, 35)
exam_2 = np.concatenate((exam_2_norm, exam_2_u))

count, bins, ignored = plt.hist(exam_2, 25, range=[50, 100])
plt.ylabel("Count", fontsize=12)
plt.title("Exam 2", fontsize=14)



plt.subplot(4,1,3)
mu, sigma = 85,5
exam_2_norm = np.random.normal(mu, sigma, 70)
exam_2_u = np.random.normal(65, 3.5, 50)
exam_2 = np.concatenate((exam_2_norm, exam_2_u))

count, bins, ignored = plt.hist(exam_2, 25, range=[50, 100])
plt.ylabel("Count", fontsize=12)
plt.title("Exam 3", fontsize=14)



plt.subplot(4,1,4)
mu, sigma = 80,6
exam_2_norm = np.random.normal(mu, sigma, 120)
exam_2 = np.concatenate((exam_2_norm, np.array([96,96])))
print(np.average(exam_2))
print(np.median(exam_2))


count, bins, ignored = plt.hist(exam_2, 25, range=[50, 100])
plt.xlabel("Score (%)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Final Exam", fontsize=14)

fig.tight_layout()

plt.show()

#---------------summary.txt ---------------------
'''
On exam 1, the average and median scores were 80 and 80, respectively. The distribution is symmetric, with a similar distribution of scores to the left and right of the center.

The range is close to 55, with the lowest grade close to 35 and the largest grade close to 90. There is one student, who scored close to 55, who is considered an outlier.

#####################
#####################

On exam 2, the average and median scores were 82 and 84 respectively. The distribution has a left skew, which agrees with our finding that the average of our dataset is smaller than the median.

The range is close to 38 with the lowest grade close to 60 and the largest grade close to 98.

#####################
#####################

On exam 3, the average and median scores were 77 and 80, respectively. The distribution is bimodal, and both modes have a similar tail on both sides of their peak, indicating that each is symmetric.

The range is close to 42, with the lowest grade close to 56 and the largest grade close to 98.


#####################
#####################


On the final exam, the average and median scores were 80 and 80, respectively. The distribution is symmetric, with a similar distribution of scores to the left and right of the center.

The range is close to 30, with the lowest grade close to 68 and the largest grade close to 98. There is one student, who scored close to 98, that is considered an outlier.'''

'''

QUARTILES
Quartiles
A common way to communicate a high-level overview of a dataset is to find the values that split the data into four groups of equal size.

By doing this, we can then say whether a new datapoint falls in the first, second, third, or fourth quarter of the data.

20 data points, with three lines splitting the data into 4 groups of 5.
The values that split the data into fourths are the quartiles.

Those values are called the first quartile (Q1), the second quartile (Q2), and the third quartile (Q3)

In the image above, Q1 is 10, Q2 is 13, and Q3 is 22. Those three values split the data into four groups that each contain five datapoints.'''

'''
QUARTILES
The Second Quartile

Let’s begin by finding the second quartile (Q2). Q2 happens to be exactly the median. Half of the data falls below Q2 and half of the data falls above Q2.

The first step in finding the quartiles of a dataset is to sort the data from smallest to largest. For example, below is an unsorted dataset:

[8, 15, 4, -108, 16, 23, 42][8,15,4,−108,16,23,42]
After sorting the dataset, it looks like this:

[-108, 4, 8, 15, 16, 23, 42][−108,4,8,15,16,23,42]
Now that the list is sorted, we can find Q2. In the example dataset above, Q2 (and the median) is 15 — there are three points below 15 and three points above 15.

Even Number of Datapoints
You might be wondering what happens if there is an even number of points in the dataset. For example, if we remove the -108 from our dataset, it will now look like this:

[4, 8, 15, 16, 23, 42][4,8,15,16,23,42]
Q2 now falls somewhere between 15 and 16. There are a couple of different strategies that you can use to calculate Q2 in this situation. One of the more common ways is to take the average of those two numbers. In this case, that would be 15.5.

Recall that you can find the average of two numbers by adding them together and dividing by two.'''


'''
QUARTILES
Q1 and Q3
Now that we’ve found Q2, we can use that value to help us find Q1 and Q3. Recall our demo dataset:

[-108, 4, 8, 15, 16, 23, 42][−108,4,8,15,16,23,42]
In this example, Q2 is 15. To find Q1, we take all of the data points smaller than Q2 and find the median of those points. In this case, the points smaller than Q2 are:

[-108, 4, 8][−108,4,8]
The median of that smaller dataset is 4. That’s Q1!

To find Q3, do the same process using the points that are larger than Q2. We have the following points:

[16, 23, 42][16,23,42]
The median of those points is 23. That’s Q3! We now have three points that split the original dataset into groups of four equal sizes.
'''

'''
QUARTILES
Method Two: Including Q2
You just learned a commonly used method to calculate the quartiles of a dataset. However, there is another method that is equally accepted that results in different values!

Note that there is no universally agreed upon method of calculating quartiles, and as a result, two different tools might report different results.

The second method includes Q2 when trying to calculate Q1 and Q3. Let’s take a look at an example:

[-108, 4, 8, 15, 16, 23, 42][−108,4,8,15,16,23,42]
Using the first method, we found Q1 to be 4. When looking at all of the points below Q2, we excluded Q2. Using this second method, we include Q2 in each half.

For example, when calculating Q1 using this new method, we would now find the median of this dataset:

[-108, 4, 8, 15][−108,4,8,15]
Using this method, Q1 is 6.'''

'''
QUARTILES
Quartiles in NumPy
We were able to find quartiles manually by looking at the dataset and finding the correct division points. But that gets much harder when the dataset starts to get bigger. Luckily, there is a function in the NumPy library that will find the quartiles for you.

The NumPy function that we’ll be using is named quantile(). You can learn more about quantiles in our lesson, but for right now all you need to know is that a quartile is a specific kind of quantile.

The code below calculates the third quartile of the given dataset:

import numpy as np

dataset = [50, 10, 4, -3, 4, -20, 2]
third_quartile = np.quantile(dataset, 0.75)
The quantile() function takes two parameters. The first is the dataset you’re interested in. The second is a number between 0 and 1. Since we calculated the third quartile, we used 0.75 — we want the point that splits the first 75% of the data from the rest.

For the second quartile, we’d use 0.5. This will give you the point that 50% of the data is below and 50% is above.

Notice that the dataset doesn’t need to be sorted for NumPy’s function to work!'''


from song_data import songs
import numpy as np

#Create the variables songs_q1, songs_q2, and songs_q3 here:
songs_q1 = np.quantile(songs, 0.25)
songs_q2 = np.quantile(songs, 0.5)
songs_q3 = np.quantile(songs, 0.75)

favorite_song = 300
quarter = 4
#Ignore the code below here:
try:
  print("The first quartile of dataset one is " + str(songs_q1) + " seconds")
except NameError:
  print("You haven't defined songs_q1")
try:
  print("The second quartile of dataset one is " + str(songs_q2)+ " seconds")
except NameError:
  print("You haven't defined songs_q2")
try:
  print("The third quartile of dataset one is " + str(songs_q3)+ " seconds")
except NameError:
  print("You haven't defined songs_q3\n")
  
'''
QUANTILES
Quantiles
Quantiles are points that split a dataset into groups of equal size. For example, let’s say you just took a test and wanted to know whether you’re in the top 10% of the class. One way to determine this would be to split the data into ten groups with an equal number of datapoints in each group and see which group you fall into.

Thirty students grades split into ten groups of three.
There are nine values that split the dataset into ten groups of equal size — each group has 3 different test scores in it.

Those nine values that split the data are quantiles! Specifically, they are the 10-quantiles, or deciles.

You can find any number of quantiles. For example, if you split the dataset into 100 groups of equal size, the 99 values that split the data are the 100-quantiles, or percentiles.

The quartiles are some of the most commonly used quantiles. The quartiles split the data into four groups of equal size.

In this lesson, we’ll show you how to calculate quantiles using NumPy and discuss some of the most commonly used quantiles.'''



import codecademylib3_seaborn
from song_data import songs
import matplotlib.pyplot as plt
import numpy as np

q1 = np.quantile(songs, 0.25)
q2 = np.quantile(songs, 0.5)
q3 = np.quantile(songs, 0.75)

plt.subplot(3,1,1)
plt.hist(songs, bins = 200)
plt.axvline(x=q1, c = 'r')
plt.axvline(x=q2, c = 'r')
plt.axvline(x=q3, c = 'r')
plt.xlabel("Song Length (Seconds)")
plt.ylabel("Count")
plt.title("4-Quantiles")

plt.subplot(3,1,2)
plt.hist(songs, bins = 200)
plt.axvline(x=np.quantile(songs, 0.2), c = 'r')
plt.axvline(x=np.quantile(songs, 0.4), c = 'r')
plt.axvline(x=np.quantile(songs, 0.6), c = 'r')
plt.axvline(x=np.quantile(songs, 0.8), c = 'r')
plt.xlabel("Song Length (Seconds)")
plt.ylabel("Count")
plt.title("5-Quantiles")

plt.subplot(3,1,3)
plt.hist(songs, bins = 200)
for i in range(1, 10):
  plt.axvline(x=np.quantile(songs, i/10), c = 'r')
plt.xlabel("Song Length (Seconds)")
plt.ylabel("Count")
plt.title("10-Quantiles")

plt.tight_layout()
plt.show()


'''


QUANTILES
Many Quantiles
In the last exercise, we found a single “quantile” — we split the first 23% of the data away from the remaining 77%.

However, quantiles are usually a set of values that split the data into groups of equal size. For example, you wanted to get the 5-quantiles, or the four values that split the data into five groups of equal size, you could use this code:

import numpy as np

dataset = [5, 10, -20, 42, -9, 10]
ten_percent = np.quantile(dataset, [0.2, 0.4, 0.6, 0.8])
Note that we had to do a little math in our head to make sure that the values [0.2, 0.4, 0.6, 0.8] split the data into groups of equal size. Each group has 20% of the data.

The data is split into 5 groups where each group has 4 datapoints.
If we used the values [0.2, 0.4, 0.7, 0.8], the function would return the four values at those split points. However, those values wouldn’t split the data into five equally sized groups. One group would only have 10% of the data and another group would have 30% of the data!
'''

from song_data import songs
import numpy as np

# Define quartiles, deciles, and tenth here:
quartiles = np.quantile(songs, [0.25, 0.5, 0.75])
deciles = np.quantile(songs, [x* 0.1 for x in range(1,10)])
tenth = 3


#Ignore the code below here:
try:
  print("The quariles are " + str(quartiles) + "\n")
except NameError:
  print("You haven't defined quartiles.\n")
try:
  print("The deciles are " + str(deciles) + "\n")
except NameError:
  print("You haven't defined deciles.\n")

'''
QUANTILES
Common Quantiles
One of the most common quantiles is the 2-quantile. This value splits the data into two groups of equal size. Half the data will be above this value, and half the data will be below it. This is also known as the median!

Ten points are below the median and ten points are above the median.
The 4-quantiles, or the quartiles, split the data into four groups of equal size. We found the quartiles in the previous exercise. Options

Quartiles split a dataset of 20 points into 4 groups with 5 points each
Finally, the percentiles, or the values that split the data into 100 groups, are commonly used to compare new data points to the dataset. You might hear statements like “You are above the 80th percentile in height”. This means that your height is above whatever value splits the first 80% of the data from the remaining 20%.'''

import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from data import school_one, school_two, school_three

deciles_one = np.quantile(school_one, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
deciles_two = np.quantile(school_two, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
deciles_three = np.quantile(school_three, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

plt.subplot(311)
plt.hist(school_one)
for decile in deciles_one:
  plt.axvline(x=decile, c = 'r')
plt.title("School One")
plt.xlabel("SAT Score")
  
plt.subplot(312)
plt.hist(school_two)
for decile in deciles_two:
  plt.axvline(x=decile, c = 'r')
plt.title("School Two")
plt.xlabel("SAT Score")
  
plt.subplot(313)
plt.hist(school_three)
for decile in deciles_three:
  plt.axvline(x=decile, c = 'r')
plt.title("School Three")
plt.xlabel("SAT Score")
plt.tight_layout()
plt.show()

'''
INTERQUARTILE RANGE
Range Review
One of the most common statistics to describe a dataset is the range. The range of a dataset is the difference between the maximum and minimum values. While this descriptive statistic is a good start, it is important to consider the impact outliers have on the results:

A dataset with some outliers.
In this image, most of the data is between 0 and 15. However, there is one large negative outlier (-20) and one large positive outlier (40). This makes the range of the dataset 60 (The difference between 40 and -20). That’s not very representative of the spread of the majority of the data!

The interquartile range (IQR) is a descriptive statistic that tries to solve this problem. The IQR ignores the tails of the dataset, so you know the range around-which your data is centered.

In this lesson, we’ll teach you how to calculate the interquartile range and interpret it.'''

'''
INTERQUARTILE RANGE
Quartiles
The interquartile range is the difference between the third quartile (Q3) and the first quartile (Q1). If you need a refresher on quartiles, you can take a look at our lesson.

For now, all you need to know is that the first quartile is the value that separates the first 25% of the data from the remaining 75%.

The third quartile is the opposite — it separates the first 75% of the data from the remaining 25%.

The interquartile range of the dataset is shown to be between Q3 and Q1.
The interquartile range is the difference between these two values.'''

from song_data import songs
import numpy as np

q1, q3 = np.quantile(songs, [0.25, 0.75])
#Create the variables q3 and interquartile_range here:

interquartile_range = q3 - q1


#-----------------------------

import codecademylib3_seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("country_data.csv")
#print(data.head())

life_expectancy = data['Life Expectancy']
life_expectancy_quartiles = np.quantile(life_expectancy, [0.25, 0.5, 0.75])
print(life_expectancy_quartiles)

gdp = data['GDP']
#print(gdp.head())
median_gdp = gdp.median()
low_gdp = data[data['GDP'] <= median_gdp]
high_gdp = data[data['GDP'] > median_gdp]
#print(low_gdp.head())

low_gdp_quartiles = np.quantile(low_gdp['Life Expectancy'], [0.25, 0.5, 0.75])
high_gdp_quartiles = np.quantile(high_gdp['Life Expectancy'], [0.25, 0.5, 0.75])
print(low_gdp_quartiles)
print(high_gdp_quartiles)

plt.hist(high_gdp["Life Expectancy"], alpha = 0.5, label = "High GDP")
plt.hist(low_gdp["Life Expectancy"], alpha = 0.5, label = "Low GDP")
plt.legend()
plt.show()


'''

BOXPLOTS
Boxplots in Matplotlib
We’ve spent this lesson building a boxplot by hand. Let’s now look at how Python’s Matplotlib library does it!

The matplotlib.pyplot module has a function named boxplot(). boxplot() takes a dataset as a parameter. This dataset could be something like a list of numbers, or a Pandas DataFrame.'''

import matplotlib.pyplot as plt

data = [1, 2, 3, 4, 5]
plt.boxplot(data)
plt.show()
'''
One of the strengths of Matplotlib is the ease of plotting two boxplots side by side. If you pass boxplot() a list of datasets, Matplotlib will make a boxplot for each, allowing you to compare their spread and central tendencies,'''

import matplotlib.pyplot as plt

dataset_one = [1, 2, 3, 4, 5]
dataset_two = [3, 4, 5, 6, 7]
plt.boxplot([dataset_one, dataset_two], labels = ['a' , 'b'])
plt.show()


#----------------------------

import codecademylib3_seaborn
import pandas as pd
from matplotlib import pyplot as plt

healthcare = pd.read_csv("healthcare.csv")
#print(healthcare.head())
#print(healthcare['DRG Definition'].unique())

chest_pain = healthcare[healthcare['DRG Definition'] == '313 - CHEST PAIN']
#print(chest_pain.head())

alabama_chest_pain  = chest_pain[chest_pain['Provider State'] == 'AL']

average_cost = alabama_chest_pain[' Average Covered Charges ']
costs = alabama_chest_pain[' Average Covered Charges '].values
#print(type(average_cost))
#print(type(cost))

states = chest_pain['Provider State'].unique()

dataset = []

for state in states:
  dataset.append(chest_pain[chest_pain['Provider State'] == state][' Average Covered Charges '].values)
  
plt.figure(figsize=(20,6))

plt.boxplot(dataset, labels= states)
plt.show()

#-------------------------------------------------------------------------

'''
STATISTICAL CONCEPTS
Sample Mean and Population Mean
Suppose you want to know the average height of an oak tree in your local park. On Monday, you measure 10 trees and get an average height of 32 ft. On Tuesday, you measure 12 different trees and reach an average height of 35 ft. On Wednesday, you measure the remaining 11 trees in the park, whose average height is 31 ft. Overall, the average height for all trees in your local park is 32.8 ft.

The individual measurements on Monday, Tuesday, and Wednesday are called samples. A sample is a subset of the entire population. The mean of each sample is the sample mean and it is an estimate of the population mean.

Note that the sample means (32 ft., 35 ft., and 31 ft.) were all close to the population mean (32.8 ft.), but were all slightly different from the population mean and from each other.

For a population, the mean is a constant value no matter how many times it’s recalculated. But with a set of samples, the mean will depend on exactly what samples we happened to choose. From a sample mean, we can then extrapolate the mean of the population as a whole. There are many reasons we might use sampling, such as:

We don’t have data for the whole population.
We have the whole population data, but it is so large that it is infeasible to analyze.
We can provide meaningful answers to questions faster with sampling.
When we have a numerical dataset and want to know the average value, we calculate the mean. For a population, the mean is a constant value no matter how many times it’s recalculated. But with a set of samples, the mean will depend on exactly what samples we happened to choose. From a sample mean, we can then extrapolate the mean of the population as a whole.'''

import numpy as np

population = np.random.normal(loc=65, scale=3.5, size=300)
population_mean = np.mean(population)

print "Population Mean: {}".format(population_mean)

sample_1 = np.random.choice(population, size=30, replace=False)
sample_2 = np.random.choice(population, size=30, replace=False)
sample_3 = np.random.choice(population, size=30, replace=False)
sample_4 = np.random.choice(population, size=30, replace=False)
sample_5 = np.random.choice(population, size=30, replace=False)

sample_1_mean = np.mean(sample_1)
print "Sample 1 Mean: {}".format(sample_1_mean)

sample_2_mean = np.mean(sample_1)
sample_3_mean = np.mean(sample_2)
sample_4_mean = np.mean(sample_3)
sample_5_mean = np.mean(sample_4)

print "Sample 2 Mean: {}".format(sample_2_mean)
print "Sample 3 Mean: {}".format(sample_3_mean)
print "Sample 4 Mean: {}".format(sample_4_mean)
print "Sample 5 Mean: {}".format(sample_5_mean)


'''
STATISTICAL CONCEPTS
Central Limit Theorem
Perhaps, this time, you’re a tailor of school uniforms at a middle school. You need to know the average height of people from 10-13 years old in order to know which sizes to make the uniforms. Knowing the best decisions are based on data, you set out to do some research at your local middle school.

Organizing with the school, you measure the heights of some students. Their average height is 57.5 inches. You know a little about sampling and decide that measuring 30 out of the 300 students gives enough data to assume 57.5 inches is roughly the average height of everyone at the middle school. You set to work with this dimension and make uniforms that fit people of this height, some smaller and some larger.

Unfortunately, when you go about making your uniforms many reports come back saying that they are too small. Something must have gone wrong with your decision-making process! You go back to collect the rest of the data: you measure the sixth graders one day (56.7, not so bad), the seventh graders after that (59 inches tall on average), and the eighth graders the next day (61.7 inches!). Your sample mean was so far off from your population mean. How did this happen?

Well, your sample selection was skewed to one direction of the total population. It looks like you must have measured more sixth graders than is representative of the whole middle school. How do you get an average sample height that looks more like the average population height?

In the previous exercise, we looked at different sets of samples taken from a population and how the mean of each set could be different from the population mean. This is a natural consequence of the fact that a set of samples has less data than the population to which it belongs. If our sample selection is poor then we will have a sample mean seriously skewed from our population mean.

There is one surefire way to mitigate the risk of having a skewed sample mean — take a larger set of samples. The sample mean of a larger sample set will more closely approximate the population mean. This phenomenon, known as the Central Limit Theorem, states that if we have a large enough sample size, all of our sample means will be sufficiently close to the population mean.

Later, we’ll learn how to put numeric values on “large enough” and “sufficiently close”.'''

import numpy as np

# Create population and find population mean
population = np.random.normal(loc=65, scale=100, size=3000)
population_mean = np.mean(population)

# Select increasingly larger samples
extra_small_sample = population[:10]
small_sample = population[:50]
medium_sample = population[:100]
large_sample = population[:500]
extra_large_sample = population[:1000]

# Calculate the mean of those samples
extra_small_sample_mean = np.mean(extra_small_sample)
small_sample_mean = np.mean(small_sample)
medium_sample_mean = np.mean(medium_sample)
large_sample_mean = np.mean(large_sample)
extra_large_sample_mean = np.mean(extra_large_sample)

# Print them all out!
print "Extra Small Sample Mean: {}".format(extra_small_sample_mean)
print "Small Sample Mean: {}".format(small_sample_mean)
print "Medium Sample Mean: {}".format(medium_sample_mean)
print "Large Sample Mean: {}".format(large_sample_mean)
print "Extra Large Sample Mean: {}".format(extra_large_sample_mean)

print "\nPopulation Mean: {}".format(population_mean)

'''Question
How does taking a larger number of samples solve the issue of a skewed sample mean? How does the Central Limit Theorem help here?

Answer
The Central Limit Theorem (CLT) is, roughly, the following statement

Regardless of the distribution of our data, if we take a large number of samples of a fixed size and plot the sample statistic which we care about (e.g. mean or standard deviation) the distribution of the resulting plot will be roughly normal, i.e. a bell curve.

Note: The distribution that we get from plotting our sample statistics is called a sampling distribution.

We can see the truth of this claim, experimentally, by playing with the applet in this lesson 77.

Okay. So how does this help us solve the issue of a skewed sample? The CLT helps because it turns out that the mean of our sampling distribution will become arbitrarily close to the mean of our original distribution as we take more and more samples. This is great because we often never know the mean of the original distribution. The CLT gives us a mathematical assurance that we can calculate it from taking samples (which we can directly calculate the mean for).

In conclusion, if we have a skewed sample mean, by

taking a larger number of samples,
plotting the mean of each sample, and
taking the mean, call it M, of the resulting distribution
M is likely to be close to the population mean by the Central Limit Theorem.'''


'''
STATISTICAL CONCEPTS
Hypothesis Tests
When observing differences in data, a data analyst understands the possibility that these differences could be the result of random chance.

Suppose we want to know if men are more likely to sign up for a given programming class than women. We invite 100 men and 100 women to this class. After one week, 34 women sign up, and 39 men sign up. More men than women signed up, but is this a “real” difference?

We have taken sample means from two different populations, men and women. We want to know if the difference that we observe in these sample means reflects a difference in the population means. To formally answer this question, we need to re-frame it in terms of probability:

“What is the probability that men and women have the same level of interest in this class and that the difference we observed is just chance?”

In other words, “If we gave the same invitation to every person in the world, would more men still sign up?”

A more formal version is: “What is the probability that the two population means are the same and that the difference we observed in the sample means is just chance?”

These statements are all ways of expressing a null hypothesis. A null hypothesis is a statement that the observed difference is the result of chance.

Hypothesis testing is a mathematical way of determining whether we can be confident that the null hypothesis is false. Different situations will require different types of hypothesis testing, which we will learn about in the next lesson.'''


import numpy as np

def intersect(list1, list2):
  return [sample for sample in list1 if sample in list2]

# the true positives and negatives:
actual_positive = [2, 5, 6, 7, 8, 10, 18, 21, 24, 25, 29, 30, 32, 33, 38, 39, 42, 44, 45, 47]
actual_negative = [1, 3, 4, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 26, 27, 28, 31, 34, 35, 36, 37, 40, 41, 43, 46, 48, 49]

# the positives and negatives we determine by running the experiment:
experimental_positive = [2, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 32, 35, 36, 38, 39, 40, 45, 46, 49]
experimental_negative = [1, 3, 6, 12, 14, 23, 25, 29, 30, 31, 33, 34, 37, 41, 42, 43, 44, 47, 48]

#define type_i_errors and type_ii_errors here
type_i_errors = intersect(actual_negative, experimental_positive)

type_ii_errors = intersect(actual_positive, experimental_negative)

'''
STATISTICAL CONCEPTS
P-Values
We have discussed how a hypothesis test is used to determine the validity of a null hypothesis. A hypothesis test provides a numerical answer, called a p-value, that helps us decide how confident we can be in the result. In this context, a p-value is the probability that we yield the observed statistics under the assumption that the null hypothesis is true.

A p-value of 0.05 would mean that if we assume the null hypothesis is true, there is a 5% chance that the data results in what was observed due only to random sampling error. This generally means there is a 5% chance that there is no difference between the two population means.

Before conducting a hypothesis test, we determine the necessary threshold we would need before concluding that the results are significant. A higher p-value is more likely to give a false positive so if we want to be very sure that the result is not due to just chance, we will select a very small p-value.

It is important that we choose the significance level before we perform our statistical hypothesis tests to yield a p-value. If we wait until after we see the results, we might pick our threshold such that we get the result we want to see. For instance, if we’re trying to publish our results, we might set a significance level that makes our results seem statistically significant. Choosing our significance level in advance helps keep us honest.

Generally, we want a p-value of less than 0.05, meaning that there is less than a 5% chance that our results are due to random chance.'''

'''
Question
What is the difference between the p-value and the significance level of an experiment?

Answer
The p-value is a statistic that we can compute as a conditional probability. It is the probability that we would observe the same sample statistics given that the null hypothesis, H_0, is true. As an equation:

p-value: P( observe the same sample statistics | H_0)
By way of an example, let’s say we’re performing a test and we want to determine if there is a significant increase in time spent on our website if we change the background color from white to yellow. Suppose that the current average time spent on our website is 15 minutes. After changing the background color to yellow, we take a sample of 100 users and we find that the average time spent on the site is 20 minutes. Does this result show a significant change or not? The significance level, produced before we perform the sample, is the threshold value from which we will determine to reject, or fail to reject, the null hypothesis; often chosen as 0.05. The p-value, however, is the following conditional probability,

P( sample mean >= 20 minutes | background of site is unchanged )
The sample mean is the average time spent on the site from a given sample and the background of the site being unchanged is the null hypothesis. To conclude, the significance level is a threshold value chosen beforehand and the p-value is a conditional probability to calculate the significance of an observation.'''


'''
HYPOTHESIS TESTING
Types of Hypothesis Test
When we are trying to compare datasets, we often need a way to be confident knowing if datasets are significantly different from each other.
Some situations involve correlating numerical data, such as:

a professor expects an exam average to be roughly 75%, and wants to know if the actual scores line up with this expectation. Was the test actually too easy or too hard?
a manager of a chain of stores wants to know if certain locations have different revenues on different days of the week. Are the revenue differences a result of natural fluctuations or a significant difference between the stores’ sales patterns?
a PM for a website wants to compare the time spent on different versions of a homepage. Does one version make users stay on the page significantly longer?
Others involve categorical data, such as:

a pollster wants to know if men and women have significantly different yogurt flavor preferences. Does a result where men more often answer “chocolate” as their favorite reflect a significant difference in the population?
do different age groups have significantly different emotional reactions to different ads?
In this lesson, you will learn how about how we can use hypothesis testing to answer these questions. There are several different types of hypothesis tests for the various scenarios you may encounter. Luckily, SciPy has built-in functions that perform all of these tests for us, normally using just one line of code.

For numerical data, we will cover:

One Sample T-Tests
Two Sample T-Tests
ANOVA
Tukey Tests
For categorical data, we will cover:

Binomial Tests
Chi Square
After this lesson, you will have a wide range of tools in your arsenal to find meaningful correlations in data.
'''

'''
Introduction Video about  Test
https://www.youtube.com/watch?v=0Pd3dc1GcHc

t test is a statistic that check if two means (averages) are reliably different from each other. 
if just look at the means, may show a difference, but we can't be sure if that is a reliable difference.
t test is a inferential statistics, it allow us to make inferences about the population beyond our data. 

Bigger t-value = Different Group
Small t-value = similar Group

t= variance between groups / variance within groups

the p-value tells us the likelihood that there is a real difference.
the p-value is the probablity that the pattern of data in the sample could be produced by random data.

type of t test

independent-samples

	test the means of two different groups.
	
	testing the average quality of two different batches of beer.
	
	other names : Between-samples, unpaired sample t test, 2 sample t test

paired-samples

	test the means of one group twice (e.g before and after)

	also called : within - subjects, repeated - measures , dependent sampels

one-sample

	tests the means of one group against a set mean
	
'''
'''

HYPOTHESIS TESTING
1 Sample T-Testing
Let’s imagine the fictional business BuyPie, which sends ingredients for pies to your household, so that you can make them from scratch. Suppose that a product manager wants the average age of visitors to BuyPie.com to be 30. In the past hour, the website had 100 visitors and the average age was 31. Are the visitors too old? Or is this just the result of chance and a small sample size?

We can test this using a univariate T-test. A univariate T-test compares a sample mean to a hypothetical population mean. It answers the question “What is the probability that the sample came from a distribution with the desired mean?”

When we conduct a hypothesis test, we want to first create a null hypothesis, which is a prediction that there is no significant difference. The null hypothesis that this test examines can be phrased as such: “The set of samples belongs to a population with the target mean”.

The result of the 1 Sample T Test is a p-value, which will tell us whether or not we can reject this null hypothesis. Generally, if we receive a p-value of less than 0.05, we can reject the null hypothesis and state that there is a significant difference.

SciPy has a function called ttest_1samp, which performs a 1 Sample T-Test for you.

ttest_1samp requires two inputs, a distribution of values and an expected mean:

tstat, pval = ttest_1samp(example_distribution, expected_mean)
print pval
It also returns two outputs: the t-statistic (which we won’t cover in this course), and the p-value — telling us how confident we can be that the sample of values came from a distribution with the mean specified.'''

from scipy.stats import ttest_1samp
import numpy as np

ages = np.genfromtxt("ages.csv")

print(ages)

ages_mean = np.mean(ages)

tstat, pval = ttest_1samp(ages, 25)

print(pval)

print(tstat, pval)

'''

HYPOTHESIS TESTING
One Sample T-Test II
In the last exercise, we got a p-value that was much higher than 0.05, so we cannot reject the null hypothesis. Does this mean that if we wait for more visitors to BuyPie, the average age would definitely be 30 and not 31? Not necessarily. In fact, in this case, we know that the mean of our sample was 31.

P-values give us an idea of how confident we can be in a result. Just because we don’t have enough data to detect a difference doesn’t mean that there isn’t one. Generally, the more samples we have, the smaller a difference we’ll be able to detect. You can learn more about the exact relationship between the number of samples and detectable differences in the Sample Size Determination course.

To gain some intuition on how our confidence levels can change, let’s explore some distributions with different means and how our p-values from the 1 Sample T-Tests change.'''
'''
1.
We have loaded a dataset daily_visitors into the editor that represents the ages of visitors to BuyPie.com in the last 1000 days. Each entry daily_visitors[i] is an array of entries representing the age per visitor to the website on day i.

We predicted that the average age would be 30, and we want to know if the actual data differs from that.

We have made a for loop that goes through the 1000 inner lists. Inside this loop, perform a 1 Sample T-Test with each day of data (daily_visitors[i]). For now, just print out the p-value from each test.


Hint
To perform the T-Test in each iteration, you would use:

tstatistic, pval = ttest_1samp(daily_visitors[i], 30)
2.
If we get a pval < 0.05, we can conclude that it is unlikely that our sample has a true mean of 30. Thus, the hypothesis test has correctly rejected the null hypothesis, and we call that a correct result.

Every time we get a correct result within the 1000 experiments, add 1 to correct_results.'''

from scipy.stats import ttest_1samp
import numpy as np

correct_results = 0 # Start the counter at 0

daily_visitors = np.genfromtxt("daily_visitors.csv", delimiter=",")

for i in range(1000): # 1000 experiments
   #your ttest here:
    tstat, pval = ttest_1samp(daily_visitors[i], 30)
    #print(daily_visitors[i])
    if pval < 0.05:
      correct_results += 1
   
   #print the pvalue here:
    print(pval)
  
print "We correctly recognized that the distribution was different in " + str(correct_results) + " out of 1000 experiments."
print "We correctly recognized that the distribution was different in " + str(correct_results) + " out of 1000 experiments."

'''
First, let’s note that the null hypothesis is usually the status quo. If we expect that the population mean is 30, this is the status quo and this is why our null hypothesis is

The set of samples belongs to a population with the target mean of 30

By performing test_1samp(ages, 30), we are testing the likelihood that the samples that we have in ages were taken/drawn from a distribution with mean 30. We could of course have just gotten somewhat unlucky with our sampling in this case, especially since the number of samples for ages is small. If the resulting p-value is less than 0.05, we will reject the null hypothesis, meaning that we’re saying it is unlikely that the sample was drawn from a distribution with mean 30. A p-value greater than or equal to 0.05 means that we fail to reject the null hypothesis, meaning that we cannot be confident that the samples were not drawn from a distribution with mean 30.'''


'''
HYPOTHESIS TESTING
2 Sample T-Test
Suppose that last week, the average amount of time spent per visitor to a website was 25 minutes. This week, the average amount of time spent per visitor to a website was 28 minutes. Did the average time spent per visitor change? Or is this part of natural fluctuations?

One way of testing whether this difference is significant is by using a 2 Sample T-Test. A 2 Sample T-Test compares two sets of data, which are both approximately normally distributed.

The null hypothesis, in this case, is that the two distributions have the same mean.

We can use SciPy’s ttest_ind function to perform a 2 Sample T-Test. It takes the two distributions as inputs and returns the t-statistic (which we don’t use), and a p-value. If you can’t remember what a p-value is, refer to the earlier exercise on univariate t-tests.'''


from scipy.stats import ttest_ind
import numpy as np

week1 = np.genfromtxt("week1.csv",  delimiter=",")
week2 = np.genfromtxt("week2.csv",  delimiter=",")

week1_mean = np.mean(week1)
week2_mean = np.mean(week2)

week1_std = np.std(week1)
week2_std = np.std(week2)

tstatstic, pval = ttest_ind(week1, week2)

print (tstatstic, pval)


'''HYPOTHESIS TESTING
Dangers of Multiple T-Tests
Suppose that we own a chain of stores that sell ants, called VeryAnts. There are three different locations: A, B, and C. We want to know if the average ant sales over the past year are significantly different between the three locations.

At first, it seems that we could perform T-tests between each pair of stores.

We know that the p-value is the probability that we incorrectly reject the null hypothesis on each t-test. The more t-tests we perform, the more likely that we are to get a false positive, a Type I error.

For a p-value of 0.05, if the null hypothesis is true then the probability of obtaining a significant result is 1 – 0.05 = 0.95. When we run another t-test, the probability of still getting a correct result is 0.95 * 0.95, or 0.9025. That means our probability of making an error is now close to 10%! This error probability only gets bigger with the more t-tests we do.'''


from scipy.stats import ttest_ind
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

a_mean = np.mean(a)
b_mean = np.mean(b)
c_mean = np.mean(c)

a_std = np.std(a)
b_std = np.std(b)
c_std = np.std(c)

a_b_pval = ttest_ind(a, b).pvalue
a_c_pval = ttest_ind(a, c).pvalue
b_c_pval = ttest_ind(b, c).pvalue

#error_prob = (1-(0.95**3))
error_prob = 1- (1-a_b_pval)*(1-a_c_pval)*(1-b_c_pval)

'''I think the main confusion here is due to these lessons using the term “p-value” interchangeably for both the significance value (the threshold at which we will determine the results are significant) and the actual p-value that is returned by running a T-test.

Here are the concepts to remember with T-tests:

We are comparing samples of different populations to see if the populations are significantly different

We determine a significance value (or p-value threshold) prior to conducting the T-tests that will act as a cut-off point for whether we will find significance

A T-test returns two values: a test statistic (tstat) and a p-value. The test statistic is basically a number that represents the difference between population means based on the variations in your sample. The larger it is, the less likely a null hypothesis is true. If it is closer to 0, it is more likely there isn’t a significant difference. The p-value is the likelihood of getting a test statistic of equal or higher value to the one returned, if the null-hypothesis is true.

The p-value itself is not the probability of a Type I error, but rather the probability of getting a test statistic (tstat) of equal or higher value if the null hypothesis is true (i.e., if the populations have the same mean and the observed differences were merely by chance). The smaller the p-value, the more likely there is significance.

Prior to running the T-tests, however, we decide that a p-value at .05 or less will indicate significance – thus we are accepting a risk of being wrong 5% of the time when we reject the null hypothesis. We would reject a null hypothesis equally if the p-value was .04 or .00004. Thus, we have a fixed risk of Type I error per T-test that is determined prior to running the experiment. This is the error the lesson is referring to.

This 5% accepted risk is compounded for each T-test we need to run during the experiment to compare each sample with each other sample, and that is why running multiple T-tests can be problematic.

Hope this helps!'''

'''
HYPOTHESIS TESTING
ANOVA
In the last exercise, we saw that the probability of making a Type I error got dangerously high as we performed more t-tests.

When comparing more than two numerical datasets, the best way to preserve a Type I error probability of 0.05 is to use ANOVA. ANOVA (Analysis of Variance) tests the null hypothesis that all of the datasets have the same mean. If we reject the null hypothesis with ANOVA, we’re saying that at least one of the sets has a different mean; however, it does not tell us which datasets are different.

We can use the SciPy function f_oneway to perform ANOVA on multiple datasets. It takes in each dataset as a different input and returns the t-statistic and the p-value. For example, if we were comparing scores on a videogame between math majors, writing majors, and psychology majors, we could run an ANOVA test with this line:

fstat, pval = f_oneway(scores_mathematicians, scores_writers, scores_psychologists)
The null hypothesis, in this case, is that all three populations have the same mean score on this videogame. If we reject this null hypothesis (if we get a p-value less than 0.05), we can say that we are reasonably confident that a pair of datasets is significantly different. After using only ANOVA, we can’t make any conclusions on which two populations have a significant difference.

Let’s look at an example of ANOVA in action.'''

from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

fstat, pval = f_oneway(a, b, c)

print(pval)

# 0.000153411660078

# after change store_b_new.csv

#8.49989098083e-215
 
'''

I don’t get it:
In the explanation: “The null hypothesis, in this case, is that all three populations have the same mean … If we reject this null hypothesis (if we get a p-value less than 0.05), we can say that we are reasonably confident that a pair of datasets is significantly different.”
But in the exercise:
With store_b the means are : 58.349636084 65.6262871356 62.3611731859
and p-value is 0.000153411660078 ie we can reject the null hypothesis (see above) and the samples are different.
With store_b_new the means are: 58.349636084 148.354940186 62.3611731859
and p-value is 8.49989098083e-215 ie we cannot reject the null hypothesis (see above) and the samples are basically the same.
Surely that is the wrong way round?

'''

'''No, it’s correct.

The null hypothesis in this case is “There is no significant difference in sales between the stores.”

Rejecting the null hypothesis (p-value < 0.05) would mean there IS a significant difference between the at least one store.

The new sales numbers for Store B easily pass the eye test and you’d expect to reject the null hypothesis. And that’s exactly what happened in the ANOVA test (p-value = 8.49989098083e-215). You would say that there is a 99.999999…% chance that a store is significant.'''

'''


HYPOTHESIS TESTING
Assumptions of Numerical Hypothesis Tests
Before we use numerical hypothesis tests, we need to be sure that the following things are true:

1. The samples should each be normally distributed…ish
Data analysts in the real world often still perform hypothesis on sets that aren’t exactly normally distributed. What is more important is to recognize if there is some reason to believe that a normal distribution is especially unlikely. If your dataset is definitively not normal, the numerical hypothesis tests won’t work as intended.

For example, imagine we have three datasets, each representing a day of traffic data in three different cities. Each dataset is independent, as traffic in one city should not impact traffic in another city. However, it is unlikely that each dataset is normally distributed. In fact, each dataset probably has two distinct peaks, one at the morning rush hour and one during the evening rush hour. The histogram of a day of traffic data might look something like this:

histogram

In this scenario, using a numerical hypothesis test would be inappropriate.

2. The population standard deviations of the groups should be equal
For ANOVA and 2-Sample T-Tests, using datasets with standard deviations that are significantly different from each other will often obscure the differences in group means.

To check for similarity between the standard deviations, it is normally sufficient to divide the two standard deviations and see if the ratio is “close enough” to 1. “Close enough” may differ in different contexts but generally staying within 10% should suffice.

3. The samples must be independent
When comparing two or more datasets, the values in one distribution should not affect the values in another distribution. In other words, knowing more about one distribution should not give you any information about any other distribution.

Here are some examples where it would seem the samples are not independent:

the number of goals scored per soccer player before, during, and after undergoing a rigorous training regimen
a group of patients’ blood pressure levels before, during, and after the administration of a drug
It is important to understand your datasets before you begin conducting hypothesis tests on it so that you know you are choosing the right test.

'''
'''HYPOTHESIS TESTING
Tukey's Range Test
Let’s say that we have performed ANOVA to compare three sets of data from the three VeryAnts stores. We received the result that there is some significant difference between datasets.

Now, we have to find out which datasets are different.

We can perform a Tukey’s Range Test to determine the difference between datasets.

If we feed in three datasets, such as the sales at the VeryAnts store locations A, B, and C, Tukey’s Test can tell us which pairs of locations are distinguishable from each other.

The function to perform Tukey’s Range Test is pairwise_tukeyhsd, which is found in statsmodel, not scipy. We have to provide the function with one list of all of the data and a list of labels that tell the function which elements of the list are from which set. We also provide the significance level we want, which is usually 0.05.

For example, if we were looking to compare mean scores of movies that are dramas, comedies, or documentaries, we would make a call to pairwise_tukeyhsd like this:

movie_scores = np.concatenate([drama_scores, comedy_scores, documentary_scores])
labels = ['drama'] * len(drama_scores) + ['comedy'] * len(comedy_scores) + ['documentary'] * len(documentary_scores)

tukey_results = pairwise_tukeyhsd(movie_scores, labels, 0.05)
It will return a table of information, telling you whether or not to reject the null hypothesis for each pair of datasets.'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

stat, pval = f_oneway(a, b, c)
print pval

# Using our data from ANOVA, we create v and l
v = np.concatenate([a, b, c])
labels = ['a'] * len(a) + ['b'] * len(b) + ['c'] * len(c)

tukey_results = pairwise_tukeyhsd(v, labels, 0.05)

print(tukey_results )

#0.000153411660078
#Multiple Comparison of Means - Tukey HSD,FWER=0.05
#=============================================
#group1 group2 meandiff  lower   upper  reject
#---------------------------------------------
#  a      b     7.2767   3.2266 11.3267  True 
#  a      c     4.0115  -0.0385  8.0616 False 
#  b      c    -3.2651  -7.3152  0.7849 False 
#---------------------------------------------

'''
HYPOTHESIS TESTING
Binomial Test
Let’s imagine that we are analyzing the percentage of customers who make a purchase after visiting a website. We have a set of 1000 customers from this month, 58 of whom made a purchase. Over the past year, the number of visitors per every 1000 who make a purchase hovers consistently at around 72. Thus, our marketing department has set our target number of purchases per 1000 visits to be 72. We would like to know if this month’s number, 58, is a significant difference from that target or a result of natural fluctuations.

How do we begin comparing this, if there’s no mean or standard deviation that we can use? The data is divided into two discrete categories, “made a purchase” and “did not make a purchase”.

So far, we have been working with numerical datasets. The tests we have looked at, the 1- and 2-Sample T-Tests, ANOVA, and Tukey’s Range test, will not work if we can’t find the means of our distributions and compare them.

If we have a dataset where the entries are not numbers, but categories instead, we have to use different methods.

To analyze a dataset like this, with two different possibilities for entries, we can use a Binomial Test. A Binomial Test compares a categorical dataset to some expectation.

Examples include:

Comparing the actual percent of emails that were opened to the quarterly goals
Comparing the actual percentage of respondents who gave a certain survey response to the expected survey response
Comparing the actual number of heads from 1000 coin flips of a weighted coin to the expected number of heads
The null hypothesis, in this case, would be that there is no difference between the observed behavior and the expected behavior. If we get a p-value of less than 0.05, we can reject that hypothesis and determine that there is a difference between the observation and expectation.

SciPy has a function called binom_test, which performs a Binomial Test for you.

binom_test requires three inputs, the number of observed successes, the number of total trials, and an expected probability of success. For example, with 1000 coin flips of a fair coin, we would expect a “success rate” (the rate of getting heads), to be 0.5, and the number of trials to be 1000. Let’s imagine we get 525 heads. Is the coin weighted? This function call would look like:
'''
pval = binom_test(525, n=1000, p=0.5)'''
It returns a p-value, telling us how confident we can be that the sample of values was likely to occur with the specified probability. If we get a p-value less than 0.05, we can reject the null hypothesis and say that it is likely the coin is actually weighted, and that the probability of getting heads is statistically different than 0.5.'''

'''
1.
Suppose the goal of VeryAnts’s marketing team this quarter was to have 6% of customers click a link that was emailed to them. They sent out a link to 10,000 customers and 510 clicked the link, which comes out to 5.1% instead of 6%. Did they do significantly worse than the target? Let’s use a binomial test to answer this question.

Use SciPy’s binom_test function to calculate the p-value the experiment returns for this distribution, where we wanted the mean to be 6% of emails opened, or p=0.06, but only saw 5.1% of emails opened.

Store the p-value in a variable called pval and print it out.


Stuck? Get a hint
2.
For the next quarter, marketing has tried out a new email tactic, including puns in every line of every email. As a result, 590 people out of 10000 opened the link in the newest email.

If we still wanted the mean to be 6% of emails opened, but now have 5.9% of emails opened, what is the new p-value. Save your results to the variable pval2

Does this new p-value make sense?'''

from scipy.stats import binom_test

pval = binom_test(510, n=10000, p=0.06)

pval2 = binom_test(590, n=10000, p=0.06)

print(pval) #0.000115920327245

print(pval2) #
'''

HYPOTHESIS TESTING
Chi Square Test
In the last exercise, we looked at data where customers visited a website and either made a purchase or did not make a purchase. What if we also wanted to track if visitors added any items to their shopping cart? With three discrete categories of data per dataset, we can no longer use a Binomial Test. If we have two or more categorical datasets that we want to compare, we should use a Chi Square test. It is useful in situations like:

An A/B test where half of users were shown a green submit button and the other half were shown a purple submit button. Was one group more likely to click the submit button?
Men and women were both given a survey asking “Which of the following three products is your favorite?” Did the men and women have significantly different preferences?
In SciPy, you can use the function chi2_contingency to perform a Chi Square test.

The input to chi2_contingency is a contingency table where:

The columns are each a different condition, such as men vs. women or Interface A vs. Interface B
The rows represent different outcomes, like “Survey Response A” vs. “Survey Response B” or “Clicked a Link” vs. “Didn’t Click”
This table can have as many rows and columns as you need.

In this case, the null hypothesis is that there’s no significant difference between the datasets. We reject that hypothesis, and state that there is a significant difference between two of the datasets if we get a p-value less than 0.05.

Instructions
1.
The management at the VeryAnts ant store wants to know if their two most popular species of ants, the Leaf Cutter and the Harvester, vary in popularity between 1st, 2nd, and 3rd graders.

We have created a table representing the different ants bought by the children in grades 1, 2, and 3 after the last big field trip to VeryAnts. Run the code to see what happens when we enter this table into SciPy’s chi-square test.

Does the resulting p-value mean that we should reject or accept the null hypothesis?

2.
A class of 40 4th graders comes into VeryAnts in the next week and buys 20 sets of Leaf Cutter ants and 20 sets of Harvester ants.

Add this data to the contingency table, rerun the chi-square test, and see if there is now a low enough value to reject the null hypothesis.

'''

from scipy.stats import chi2_contingency

# Contingency table
#         harvester |  leaf cutter
# ----+------------------+------------
# 1st gr | 30       |  10
# 2nd gr | 35       |  5
# 3rd gr | 28       |  12

X = [[30, 10],
     [35, 5],
     [28, 12],
     [20,20]]

chi2, pval, dof, expected = chi2_contingency(X)
print pval


#0.155082308077
#0.00281283455955

'''
I don’t understand. How can p be so big (= 0.155082308077) for this table:

# Contingency table
#         harvester |  leaf cutter
# ----+------------------+------------
# 1st gr | 30       |  10
# 2nd gr | 35       |  5
# 3rd gr | 28       |  12
when there are visibly extreme differences between harvester and leaf cutter, while at the same time it’s so small (p = 0.00281283455955) for this table:

# Contingency table
#         harvester |  leaf cutter
# ----+------------------+------------
# 1st gr | 30       |  10
# 2nd gr | 35       |  5
# 3rd gr | 28       |  12
# 4th gr | 20       |  20
after we added an equal amount of ants to both columns, which should even things out and therefore increase the probability of there not being a major difference between these sets.'''
'''
answer:
This is a very good point. However, I think that the answer to your questions can be found in the way you read and interprete the tables; you look at the tables only in an “horizontal” way (harvester vs. leaf cutter) but you should look at them in a “vertical” way as well (harvester & leaf cutter in relation to each grade). Below is what I mean.

In the first place, I will tell you that it was easier for me to read the table and interprete the statistical results if I gave some discrete attributes to the grades of students. For example consider that the 1st grade come from USA, the 2nd from Canada and the 3rd from France. Now, recall what you are looking for: is species-ants preference related to the country of origin of the students? The Null Hypothesis is that there is no association between country and ants-preferences.

Before you make the statistical calculations, you can make a guess by looking at the first table: For each county individually, the majority of students show an obvious preference to harvester ; this makes you guess that the country of origin does not matter; whichever of the three countries students come from, most of them prefer harvester. After the calculation of the chi-square test and the result of the p-value, it seems that you were right. P-value is quite high , you can not reject the Ho, meaning that there is no association.
Now, a fourth group of students visit the VeryAnts ant store, they come from Spain. They are again 40 in total, but something different happens now. You notice a different attitude, the majority of them don’t show a preference towards harvester as the previous 3 groups did; their preferences are equally shared (20-20). You start to be suspicious that maybe the country does matter to the preferences of ants. The calculation of chi-square test and p-value proves that. Low p-value -> rejection of Ho -> there seems to be actually an association.

To feel more confident with the above results, I did the following: I added another group of “extreme” values to the contigency table. For instance, a 5th group of students, coming let’say from Japan, visit the store. They are again 40 in total. 39 buy harvester and only 1 leaf cutter! One may be very surprised by that overwhelming preference to harvester and may wonder why. Well, this might be a part of another survey. In the current survey, now we feel much more confident that country does matter. We may guess what the p-value will be , extemely low, the Ho is rejected .'''

'''
answer 2: 
Begin with the totals:

Grade	Harvester	Leaf	Total
1	       30	     10	     40
2	       35	      5	     40
3	       28	     12	     40
Total	   93	     27	    120

So, 120 students made 1 selection each, or 120 selections.

Of all 120 ant selections, harvester comprised 93, or 77.5%

Of all 120 students, 40 (33%) are in 1st grade

Therefore, we would expect from all of the 120 ant selections, 0.333 * 0.775 * 120 = 31 would be in the first grade:

This can be done in each block to create an “Expected” table:

Grade	Harvester	Leaf	Total
1	       31	     9	     40
2	       31	     9	     40
3	       31	     9	     40
Total	   93	     27	    120
Note that the totals remain the same, and that in this particular case, numbers are the same for each grade, since there are the same number of students in each grade, and our default (null) assumption is that there is no difference in the number of ants selected between grades 1, 2 and 3.

Now, chi-square is simply sum((Observed - Expected)**2/ Expected)

For instance for grade 1, Observed - Expected is -1; that squared is 1, and dividing by Expected gives 1/31, or 0.03226. The similarly calculated values for the other five blocks are (0.51613, 0.29032, 0.11111, 1.77778 and 1.0). The sum of those 6 is 3.7276, which, as you will note, compares nicely with the “chi2” value yielded in the exercise.

I do not know the details of getting there to the p-value, mainly because nearly every reference says either “look it up in a table” or “use an online p-value calculator.” You need to know the “degrees of freedom” which is one less than the number of categories, in our case 2. So you take chi-square of 3.7276 and 2 degrees of freedom to a p-value calculator 1 to get: 0.1551 Check!'''

'''

Quiz:

1.Let’s say that last month 7% of free users of a site converted to paid users, but this month only 5% of free users converted. What kind of test should we use to see if this difference is significant?

2.Let’s say we run a 1 Sample T-Test on means for an exam. We expect the mean to be 75%, but we want to see if the actual scores are significantly better or worse than what we expected. After running the T-Test, we get a p-value of 0.25. What does this result mean?

3.What kind of test would you use to see if men and women identify differently as “Republican”, “Democrat”, or “Independent”?

4.You regularly order delivery from two different Pho restaurants, “What the Pho” and “Pho Tonic”. You want to know if there’s a significant difference between these two restaurants’ average time to deliver to your house. What test could you use to determine this?

5.You own a juice bar and you theorize that 75% of your customers live in the surrounding 5 blocks. You survey a random sample of 12 customers and find that 7 of them live within those 5 blocks. What test do you run to determine if your results significantly differ from your expectation?

6.You’ve surveyed 10 people who work in finance, 10 people who work in education, and 10 people who work in the service industry on how many cups of coffee they drink per day. What test can you use to determine if there is a significant difference between the average coffee consumption of these three groups?

7.Let’s say we are comparing the time that users spend on three different versions of a landing page for a website. What test do we use to determine if there is a significant difference between any two of the sets?

8. You just bought a new tea kettle that is supposed to heat water to boiling in 2 minutes. What kind of test can you run to determine if the time-to-boil is averaging significantly more than 2 minutes?


9.If we perform an ANOVA test on 3 datasets and reject the null hypothesis, what test should we perform to determine which pairs of datasets are different?

10.You’ve collected data on 1000 different sites that end with .com, .edu, and .org and have recorded the number of each that have Times New Roman, Helvetica, or another font as their main font. What test can you use to determine if there’s a relationship between top-level domain and font type?


Quiz Answer

1.Chi Square
2.we cannot confidently reject the null-hypothesis, so we do not have enough data to say that the mean on this exam is different from 75%
3.Chi Square
4.2 sample test
5.Binomial test
6.Anova
7.Anova
8.1 sample
9.tukey's range test
10.Chi saqare

'''
'''1.
We’re going to start by including a data interface that a previous software engineer wrote for you, it’s aptly titled familiar, so just import that.


Stuck? Get a hint
2.
Perfect, now the first thing we want to show is that our most basic package, the Vein Pack, actually has a significant impact on the subscribers. It would be a marketing goldmine if we can show that subscribers to the Vein Pack live longer than other people.

Lifespans of Vein Pack users are returned by the function lifespans(package='vein'), which is part of the familiar module. Call that function and save the data into a variable called vein_pack_lifespans.


Stuck? Get a hint
3.
We’d like to find out if the average lifespan of a Vein Pack subscriber is significantly different from the average life expectancy of 71 years.

Import the statistical test we would use to determine if a sample comes from a population that has a given mean from scipy.stats.


Stuck? Get a hint
4.
Now use the 1-Sample T-Test to compare vein_pack_lifespans to the average life expectancy 71. Save the result into a variable called vein_pack_test.


Stuck? Get a hint
5.
Let’s check if the results are significant! Check the pvalue of vein_pack_test. If it’s less than 0.05, we’ve got significance!

6.
We want to present this information to the CEO, Vlad, of this incredible finding. Let’s print some information out! If the test’s p-value is less than 0.05, print “The Vein Pack Is Proven To Make You Live Longer!”. Otherwise print “The Vein Pack Is Probably Good For You Somehow!”


Stuck? Get a hint
Upselling Familiar: Pumping Life Into The Company
7.
In order to differentiate Familiar’s different product lines, we’d like to compare this lifespan data between our different packages. Our next step up from the Vein Pack is the Artery Pack. Let’s get the lifespans of Artery Pack subscribers using the same method, called with package='artery' instead. Save the value into a variable called artery_pack_lifespans.


Stuck? Get a hint
8.
Now we want to show that the subscribers to the Artery Pack experience a significant improvement even beyond what a Vein Pack subscriber’s benefits. Import the 2-Sample T-Test and we’ll use that to see if there is a significant difference between the two subscriptions.


Stuck? Get a hint
9.
Okay let’s run the 2-Sample test! Save the results into a variable named package_comparison_results.


Stuck? Get a hint
10.
Let’s see the results! If the p-value from our experiment is less than 0.05, the results are significant and we should print out “the Artery Package guarantees even stronger results!”. Otherwise we should print out “the Artery Package is also a great product!”


Stuck? Get a hint
11.
Well, shame that it’s not significantly better, but maybe there’s a way to demonstrate the benefits of the Artery Package yet.


Stuck? Get a hint
Benefitting Everyone: A Familiar Problem
12.
If your lifespan isn’t significantly increased by signing up for the Artery Package, maybe we can make some other claim about the benefits of the package. To that end, we’ve sent out a survey collecting the iron counts for our subscribers, and filtered that data into “low”, “normal”, and “high”.

We received 200 responses from our Vein Package subscribers. 70% of them had low iron counts, 20% had normal, and 10% of them have high iron counts.

We were only able to get 145 responses from our Artery Package subscribers, but only 20% of them had low iron counts. 60% had normal, and 20% have high iron counts.

13.
The data from the survey has been collected and formatted into a contingency table. You can access that data from the function familiar.iron_counts_for_package(). Save the survey results into a variable called iron_contingency_table.


Stuck? Get a hint
14.
We want to be able to tell if what seems like a higher number of our Artery Package subscribers is a significant difference from what was reported by Vein Package subscribers. Import the Chi-Squared test so that we can find out.


Stuck? Get a hint
15.
Run the Chi-Squared test on the iron_contingency_table and save the p-value in a variable called iron_pvalue. Remember that this test returns four things: the test statistic, the p-value, the number of degrees of freedom, and the expected frequencies.


Hint
Run the Chi-Squared test on the contingency table and save the p-value like so:

_, iron_pvalue, _, _ = chi2_contingency(iron_contingency_table)
16.
Here’s the big moment: if the iron_pvalue is less than 0.05, print out “The Artery Package Is Proven To Make You Healthier!” otherwise we’ll have to use our other marketing copy: “While We Can’t Say The Artery Package Will Help You, I Bet It’s Nice!”

17.
Fantastic! With proven benefits to both of our product lines, we can definitely ramp up our marketing and sales. Look out for a Familiar face in drug stores everywhere.'''



#------------------------script.py--------------------------------------
from familiar import *
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

vein_pack_lifespans = lifespans('vein')
artery_pack_lifespans = lifespans('artery')

vein_pack_lifespans_mean = np.mean(vein_pack_lifespans)
artery_pack_lifespans_mean = np.mean(artery_pack_lifespans)
#print(vein_pack_lifespans_mean)

vein_pack_test = ttest_1samp(vein_pack_lifespans , 71)

print(vein_pack_test)

if vein_pack_test.pvalue > 0.05:
  print('The Vein Pack Is Probably Good For You Somehow!')
else:
  print('The Vein Pack Is Proven To Make You Live Longer!')

package_comparison_results = ttest_ind(vein_pack_lifespans, artery_pack_lifespans)

print(package_comparison_results)

if package_comparison_results.pvalue > 0.05:
  print('The Artery Package is also a great product!')
else:
  print('The Artery Package guarantees even stronger results!')

iron_contingency_table = iron_counts_for_package()

_, iron_pvalue, _, _ = chi2_contingency(iron_contingency_table)
print(iron_pvalue)

if iron_pvalue < 0.05:
  print('The Artery Package Is Proven To Make You Healthier!')
else:
  print('while we can\'t say the artery package will help you, i bet it''s nice')

#------------------familiar.py------------------------------------------

def lifespans(package):
  if package == 'vein':
    return [76.937674313716172, 75.993359130146814, 74.798150123540481, 74.502021471585508, 77.48888897587436, 72.142565731540429, 75.993031671911822, 76.341550480952279, 77.484755629998816, 76.532101480086695, 76.255089552764176, 77.58398316566651, 77.047370349622938, 72.874751745947108, 77.435045470028442, 77.492341410789194, 78.326720468799522, 73.343702468870674, 79.969157652363464, 74.838005833003251]
  elif package == 'artery':
    return [76.335370084268348, 76.923082315590619, 75.952441644877794, 74.544983480720305, 76.404504275447195, 73.079248886365761, 77.023544610529925, 74.117420420068797, 77.38650656208344, 73.044765837189928, 74.963118508661665, 73.319543019334859, 75.857401376968625, 76.152653513512547, 73.355102863226705, 73.902212564587884, 73.771211950924751, 68.314898302855781, 74.639757177753282, 78.385477308439789]
  else:
    print "Package not found. Possible values 'vein' or 'artery'"
    return None

def iron_counts_for_package():
  """
            vein     |  artery
    ----+------------+------------
     low|200 * 0.7   |145 * 0.2
  normal|200 * 0.2   |145 * 0.2
    high|200 * 0.1   |145 * 0.6
  """
  return [[140, 29],
          [40, 87],
          [20, 29]]


#-----------------------Hypothesis testing project #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

#----dog_data.csv -----


is_rescue,weight,tail_length,age,color,likes_children,is_hypoallergenic,name,breed
0,4,3.59,3,grey,0,0,Nonah,chihuahua
0,2,6.89,1,grey,1,0,Kai,chihuahua
0,5,1.54,6,grey,1,1,Janith,chihuahua
0,3,3.39,5,grey,0,0,Nicky,chihuahua
0,5,4.28,6,grey,0,1,Tobe,chihuahua
0,8,4.63,2,grey,0,1,Carroll,chihuahua
0,1,3.48,3,grey,0,0,Bjorn,chihuahua
0,8,0.97,3,grey,0,0,Janina,chihuahua
0,3,1.93,3,grey,1,1,Vick,chihuahua
0,3,4.18,2,grey,0,0,Gunilla,chihuahua
0,4,2.63,4,grey,0,1,Kahaleel,chihuahua
1,6,2.26,1,grey,1,0,Loleta,chihuahua
0,4,2.79,3,grey,1,1,Brook,chihuahua
0,6,6.76,3,grey,0,0,Janine,chihuahua
0,6,2.27,2,grey,0,0,Tilly,chihuahua
0,8,4,2,grey,1,0,Oren,chihuahua
1,7,1.47,4,grey,0,0,Hadleigh,chihuahua
0,5,5.95,2,grey,0,1,Zorina,chihuahua
0,7,0.62,2,grey,0,0,Gasparo,chihuahua
0,6,5.42,3,grey,1,0,Lowrance,chihuahua
0,8,2.63,3,grey,0,0,Keelia,chihuahua
0,1,1.45,1,grey,0,1,Jannel,chihuahua
0,3,2.88,4,grey,0,0,Perkin,chihuahua
0,7,3.23,3,grey,1,0,Chrissy,chihuahua
0,4,5.64,3,grey,0,0,Robena,chihuahua
0,5,1.84,2,grey,0,0,Glenden,chihuahua
0,3,4.74,1,grey,1,0,Inez,chihuahua
0,7,0.89,2,white,1,0,Zed,chihuahua
0,5,1.16,1,white,0,1,Silvio,chihuahua
0,3,10.08,1,white,1,0,Weber,chihuahua
0,6,4.88,2,white,0,0,Domenic,chihuahua
0,72,13.13,6,black,1,0,Talbert,greyhound
0,88,20.98,4,black,0,0,Brandtr,greyhound
0,90,17.6,7,black,1,0,Dorthea,greyhound
0,66,13.71,7,black,0,1,Kip,greyhound
0,57,20.3,6,black,1,0,Stephanus,greyhound
0,76,9.53,3,black,0,1,Stephanie,greyhound
0,70,10.18,7,black,0,1,Lorenzo,greyhound
0,100,17.92,1,black,0,1,Eadmund,greyhound
0,65,14.18,4,black,0,1,Tyne,greyhound
0,84,18.91,4,black,0,0,Evania,greyhound
0,62,21.14,5,grey,1,0,Derrik,greyhound
0,81,16.96,4,grey,0,0,Hervey,greyhound
0,84,17.45,8,grey,1,1,Tobie,greyhound
0,85,16.92,1,grey,0,0,Say,greyhound
0,87,27.34,2,grey,1,1,Winne,greyhound
0,72,16.6,5,grey,1,1,Lyn,greyhound
0,72,18.48,1,grey,0,0,Shelia,greyhound
0,70,14.91,4,grey,1,1,Shep,greyhound
0,88,18.95,6,grey,0,1,Chrystel,greyhound
0,61,15.23,1,grey,0,1,Sheff,greyhound
0,82,12.83,3,grey,1,0,Verla,greyhound
0,65,22.53,3,grey,0,1,Jarrett,greyhound
0,77,16.25,12,grey,1,0,Julius,greyhound
0,71,20.58,2,grey,0,0,Mack,greyhound
0,90,12.36,1,grey,0,1,Anallise,greyhound
0,54,17.98,7,grey,0,1,Jenifer,greyhound
0,68,26.59,1,grey,0,1,Ransell,greyhound
0,66,14.11,8,grey,1,0,Sher,greyhound
0,93,24.3,9,grey,1,0,Tiffi,greyhound
0,74,15.12,5,grey,1,1,Lianna,greyhound
1,65,23.64,6,grey,1,1,Steward,greyhound
0,86,14.62,3,grey,1,0,Farly,greyhound
0,35,14.67,1,white,1,0,Hamish,pitbull
0,52,12.8,7,white,0,0,Walsh,pitbull
0,43,7.88,4,white,0,0,Robert,pitbull
0,40,3.78,4,white,1,1,Raynor,pitbull
0,41,10.23,4,white,0,0,Rene,pitbull
0,39,9.13,4,white,0,0,Jolene,pitbull
0,44,9.53,4,white,0,1,Alicea,pitbull
0,58,8.05,1,black,1,0,Moise,poodle
0,56,9.44,4,black,1,0,Boote,poodle
1,59,4.04,4,black,1,0,Beatrix,poodle
0,70,12.37,1,black,1,0,Rabbi,poodle
0,52,11.42,2,black,0,0,Tallou,poodle
0,56,8.7,5,black,1,0,Evvie,poodle
0,57,9.47,5,black,1,1,Sayers,poodle
0,66,8.32,5,black,1,0,Hillie,poodle
1,58,11.89,1,black,0,1,Kath,poodle
0,61,7.57,5,black,0,0,Joelly,poodle
0,56,9.66,2,black,1,0,Kellen,poodle
0,65,8.65,4,black,0,0,Arch,poodle
0,60,11.95,8,grey,1,0,Sibella,poodle
1,63,10.78,6,grey,0,0,Turner,poodle
0,67,10,2,grey,1,0,Bibby,poodle
0,64,12.03,10,grey,1,0,Gregg,poodle
0,64,6.34,8,grey,1,0,Nevin,poodle
1,66,9.68,6,grey,1,0,Sonny,poodle
0,54,8,7,white,1,0,Ford,poodle
0,67,3.9,11,white,1,0,Haroun,poodle
1,60,5.66,3,white,0,0,Almeta,poodle
0,57,3.55,10,grey,0,0,Corbin,rottweiler
0,37,2.53,8,grey,1,0,Gaby,rottweiler
0,58,3.57,12,grey,0,0,Bea,rottweiler
0,58,5.36,5,grey,0,1,Fransisco,rottweiler
0,54,3.94,3,grey,1,0,Heindrick,rottweiler
0,58,1.8,7,grey,1,0,Lorrin,rottweiler
0,47,5.14,9,grey,0,0,Katinka,rottweiler
1,58,4.42,3,grey,0,0,Onida,rottweiler
0,58,2.02,1,grey,0,0,Jsandye,rottweiler
1,46,6.08,5,grey,0,0,Stirling,rottweiler
0,50,0.82,8,grey,0,0,Kata,rottweiler
0,75,5.67,4,grey,0,0,Bettine,rottweiler
0,51,6.45,3,grey,0,0,Conant,rottweiler
0,65,4.22,7,grey,1,0,Lucien,rottweiler
1,46,1.36,7,grey,1,0,Gladys,rottweiler
0,54,3.25,11,grey,1,0,Court,rottweiler
0,67,4.4,5,grey,0,1,Niccolo,rottweiler
0,64,5.16,9,grey,1,1,Gage,rottweiler
0,54,5.33,7,grey,1,0,Bryn,rottweiler
0,50,9.32,10,grey,0,0,Paulina,rottweiler
0,47,3.06,12,grey,0,0,Northrup,rottweiler
0,11,2.77,5,brown,0,0,Ursulina,shihtzu
0,13,2.02,2,brown,1,0,Lind,shihtzu
0,11,2.99,3,brown,0,1,Gabriela,shihtzu
0,9,2.38,4,brown,1,0,Loralie,shihtzu
0,16,3.38,4,brown,1,0,Townie,shihtzu
0,12,2.99,4,brown,1,0,Mariquilla,shihtzu
0,12,1.33,2,brown,1,0,Tobye,shihtzu
0,10,2.81,3,brown,1,1,Mercedes,shihtzu
0,16,1.31,2,brown,0,0,Birdie,shihtzu
0,14,2.13,2,brown,0,0,Adolpho,shihtzu
0,10,2.53,3,brown,1,1,Michaelina,shihtzu
0,12,2.79,4,brown,1,0,Stevena,shihtzu
0,6,2.83,5,grey,1,0,Celie,shihtzu
0,13,2.87,5,grey,0,0,Mitchael,shihtzu
0,11,2.97,4,grey,1,0,Anselma,shihtzu
0,14,1.13,4,grey,0,1,Lorita,shihtzu
0,10,1.4,3,grey,1,1,Lola,shihtzu
0,15,3.48,4,grey,1,0,Jonah,shihtzu
0,14,2.93,4,grey,1,0,Fanny,shihtzu
0,9,0.57,2,grey,1,0,Cammie,shihtzu
0,11,1.91,3,grey,0,0,Niven,shihtzu
0,9,1.26,3,grey,1,0,Daile,shihtzu
0,11,2.21,2,grey,0,0,Felix,shihtzu
0,29,5.33,1,grey,1,1,Charla,terrier
0,21,3.05,1,grey,1,0,Coral,terrier
0,30,6.45,4,grey,1,0,Sybille,terrier
0,30,5.15,3,grey,0,0,Neils,terrier
0,28,4.51,1,grey,1,0,Daisi,terrier
0,50,3.27,2,grey,0,1,Haleigh,terrier
0,36,1.65,2,grey,0,0,Porter,terrier
0,24,5.57,3,grey,1,1,Eva,terrier
0,40,5.67,2,grey,0,1,Kaitlin,terrier
0,35,0.48,1,grey,0,0,Anni,terrier
0,13,3.95,1,grey,1,0,Asher,terrier
1,42,2.96,1,grey,0,0,Leoine,terrier
0,30,4.23,3,grey,1,0,Maritsa,terrier
0,23,3.38,1,grey,0,0,Klarrisa,terrier
0,39,3.69,2,grey,0,1,Constancia,terrier
0,32,4.64,3,grey,0,1,Hedy,terrier
0,23,0.56,1,grey,1,0,Ardith,terrier
0,22,1,2,grey,1,0,Meridith,terrier
0,23,7.5,4,grey,1,0,Farrell,terrier
0,27,5.28,1,grey,0,0,Ira,terrier
0,29,5.25,3,grey,0,0,Eloisa,terrier
0,31,16.25,2,grey,0,0,Trudy,whippet
1,56,10.38,7,grey,1,0,Morgan,whippet
0,68,14.89,6,grey,0,1,Engelbert,whippet
0,37,9.95,8,grey,0,0,Theo,whippet
0,24,14.34,6,grey,0,0,Galvin,whippet
0,42,7.04,12,grey,0,1,Cecilla,whippet
0,58,10.59,8,grey,1,0,Regina,whippet
0,62,18.83,1,grey,1,0,Ellwood,whippet
0,27,15.18,3,grey,0,0,Gard,whippet
  
  
#-------------------fetchmaker.py---------------


import pandas as pd
import numpy as np

dogs = pd.read_csv("dog_data.csv")

def get_attribute(breed, attribute):
  if breed in dogs.breed.unique():
    if attribute in dogs.columns:
			return dogs[dogs["breed"] == breed][attribute]
    else:
      raise NameError('Attribute {} does not exist.'.format(attribute))
  else:
    raise NameError('Breed {} does not exist.'.format(breed))
  

def get_weight(breed):
  return get_attribute(breed, 'weight')
  
def get_tail_length(breed):
  return get_attribute(breed, 'tail_length')

def get_color(breed):
	return get_attribute(breed, 'color')

def get_age(breed):
	return get_attribute(breed, 'age')

def get_is_rescue(breed):
	return get_attribute(breed, 'is_rescue')

def get_likes_children(breed):
	return get_attribute(breed, 'likes_children')

def get_is_hypoallergenic(breed):
	return get_attribute(breed, "is_hypoallergenic")

def get_name(breed):
	return get_attribute(breed, "name")


#----------------script.py---------------

import numpy as np
import fetchmaker
from scipy.stats import binom_test, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

rottweiler_tl = fetchmaker.get_tail_length("rottweiler")

#print (np.mean(rottweiler_tl))
#print(np.std(rottweiler_tl))

#-------------

whippet_rescue = fetchmaker.get_is_rescue('whippet')
num_whippet_rescues = np.count_nonzero(whippet_rescue)
num_whippets = np.size(whippet_rescue)

pval = binom_test(num_whippet_rescues, n=num_whippets, p=0.08)
print (pval)#0.58

whippets_wt = fetchmaker.get_weight('whippet')
terriers_wt = fetchmaker.get_weight('terrier')
pitbulls_wt = fetchmaker.get_weight('pitbull')

print(np.mean(whippets_wt))
print(np.mean(terriers_wt))
print(np.mean(pitbulls_wt))

_, pval2 = f_oneway(whippets_wt, terriers_wt, pitbulls_wt)
print(pval2)

v = np.concatenate([whippets_wt, terriers_wt, pitbulls_wt])
labels = ['whippet'] * len(whippets_wt) + ['terrier'] * len(terriers_wt) + ['pitbull'] * len(pitbulls_wt)

tukey_results = pairwise_tukeyhsd(v, labels, 0.05)
print(tukey_results)

#-------------

poodle_colors = fetchmaker.get_color('poodle')
shihtzu_colors = fetchmaker.get_color('shihtzu')

colors_list = ['brown', 'gold', 'Grey', 'White']

poodle_color_count = []
shihtzu_color_count = []
x = []

for color in colors_list:
  temp = []
  temp.append(np.count_nonzero(poodle_colors == color.lower() ))
  temp.append(np.count_nonzero(shihtzu_colors == color.lower() ))
  x.append(temp)

'''	
Poodle	Shih Tzu
Black	  x   	x
Brown	  x   	x
Gold	  x	    x
Grey	  x	    x
White	  x	    x'''

chi2, pval, dof, expected = chi2_contingency(x)
print pval






#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-HYPOTHESIS TESTING CHEATSHEET SCIPY / PYTHON SCIPY HYPOTHESIS TSSTING #-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''


Cheatsheets / Hypothesis Testing with SciPy

Hypothesis Testing
Print PDF icon
Print Cheatsheet


TOPICS

Hypothesis Testing
Sample Size Determination
Analyze FarmBurg's A/B Test
Hypothesis Test Errors
Type I errors, also known as false positives, is the error of rejecting a null hypothesis when it is actually true. This can be viewed as a miss being registered as a hit. The acceptable rate of this type of error is called significance level and is usually set to be 0.05 (5%) or 0.01 (1%).

Type II errors, also known as false negatives, is the error of not rejecting a null hypothesis when the alternative hypothesis is the true. This can be viewed as a hit being registered as a miss.

Depending on the purpose of testing, testers decide which type of error to be concerned. But, usually type I error is more important than type II.

Sample Vs. Population Mean
In statistics, we often use the mean of a sample to estimate or infer the mean of the broader population from which the sample was taken. In other words, the sample mean is an estimation of the population mean.

Central Limit Theorem
The central limit theorem states that as samples of larger size are collected from a population, the distribution of sample means approaches a normal distribution with the same mean as the population. No matter the distribution of the population (uniform, binomial, etc), the sampling distribution of the mean will approximate a normal distribution and its mean is the same as the population mean.

The central limit theorem allows us to perform tests, make inferences, and solve problems using the normal distribution, even when the population is not normally distributed.

Hypothesis Test P-value
Statistical hypothesis tests return a p-value, which indicates the probability that the null hypothesis of a test is true. If the p-value is less than or equal to the significance level, then the null hypothesis is rejected in favor of the alternative hypothesis. And, if the p-value is greater than the significance level, then the null hypothesis is not rejected.

Univariate T-test
A univariate T-test (or 1 Sample T-test) is a type of hypothesis test that compares a sample mean to a hypothetical population mean and determines the probability that the sample came from a distribution with the desired mean.

This can be performed in Python using the ttest_1samp() function of the SciPy library. The code block shows how to call ttest_1samp(). It requires two inputs, a sample distribution of values and an expected mean and returns two outputs, the t-statistic and the p-value.
'''
from scipy.stats import ttest_1samp

t_stat, p_val = ttest_1samp(example_distribution, expected_mean)'''
Tukey’s Range Hypothesis Tests
A Tukey’s Range hypothesis test can be used to check if the relationship between two datasets is statistically significant.

The Tukey’s Range test can be performed in Python using the StatsModels library function pairwise_tukeyhsd(). The example code block shows how to call pairwise_tukeyhsd(). It accepts a list of data, a list of labels, and the desired significance level.
'''
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_results = pairwise_tukeyhsd(data, labels, alpha=significance_level)



'''




'''

