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