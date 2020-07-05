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




