



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-PYTHON MATPLOTLIB PYTHON MATPLOTLIB PYTHON MATPLOTLIB PYTHON MATPLOTLIB PYTHON MATPLOTLIB#-#-#-#-#-#-#-#-#-
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
import codecademylib
from matplotlib import pyplot as plt

days = [0, 1, 2, 3, 4, 5, 6]
money_spend = [10, 12, 12, 10, 14, 22, 24]

plt.plot(days, money_spend)
plt.show()

#-----------------------

# Days of the week:
days = [0, 1, 2, 3, 4, 5, 6]
# Your Money:
money_spent = [10, 12, 12, 10, 14, 22, 24]
# Your Friend's Money:
money_spent_2 = [11, 14, 15, 15, 22, 21, 12]
# Plot your money:
plt.plot(days, money_spent)
# Plot your friend's money:
plt.plot(days, money_spent_2)
# Display the result:
plt.show()

#----------------------

plt.plot(days, money_spent, color='green')
plt.plot(days, money_spent_2, color='#AAAAAA')

# Dashed:
plt.plot(x_values, y_values, linestyle='--')
# Dotted:
plt.plot(x_values, y_values, linestyle=':')
# No line:
plt.plot(x_values, y_values, linestyle='')

# A circle:
plt.plot(x_values, y_values, marker='o')
# A square:
plt.plot(x_values, y_values, marker='s')
# A star:
plt.plot(x_values, y_values, marker='*')

plt.plot(days, money_spent, color='green', linestyle='--')
plt.plot(days, money_spent_2, color='#AAAAAA',  marker='o')

#-------------------------------

#For example, if we want to display a plot from x=0 to x=3 and from y=2 to y=5, we would call plt.axis([0, 3, 2, 5]).

x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]
plt.plot(x, y)
plt.axis([0, 3, 2, 5])
plt.show()

#-----------------------------------------------
#We can label the x- and y- axes by using plt.xlabel() and plt.ylabel(). The plot title can be set by using plt.title().

hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
happiness = [9.8, 9.9, 9.2, 8.6, 8.3, 9.0, 8.7, 9.1, 7.0, 6.4, 6.9, 7.5]
plt.plot(hours, happiness)
plt.xlabel('Time of day')
plt.ylabel('Happiness Rating (out of 10)')
plt.title('My Self-Reported Happiness While Awake')
plt.show()

#----------------------------------------

#We can create subplots using .subplot().

#The command plt.subplot() needs three arguments to be passed into it:

#The number of rows of subplots
#The number of columns of subplots
#The index of the subplot we want to create

#For instance, the command plt.subplot(2, 3, 4) would create “Subplot 4” from the figure above.


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

# Plot the ground truth

fig = plt.figure(fignum, figsize=(4, 3))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Robots', 0),
                    ('Cyborgs', 1),
                    ('Humans', 2)]:
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

ax.set_xlabel('Time to Heal')
ax.set_ylabel('Reading Speed')
ax.set_zlabel('EQ')

ax.set_title('')
ax.dist = 12

plt.show(head)

#Any plt.plot() that comes after plt.subplot() will create a line plot in the specified subplot. For instance:

# Data sets
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

# First Subplot
plt.subplot(1, 2, 1)
plt.plot(x, y, color='green')
plt.title('First Subplot')

# Second Subplot
plt.subplot(1, 2, 2)
plt.plot(x, y, color='steelblue')
plt.title('Second Subplot')

# Display both subplots
plt.show()

#----------------------------------------
#.subplots_adjust() has some keyword arguments that can move your plots within the figure:

#left — the left-side margin, with a default of 0.125. You can increase this number to make room for a y-axis label
#right — the right-side margin, with a default of 0.9. You can increase this to make more room for the figure, or decrease it to make room for a legend
#bottom — the bottom margin, with a default of 0.1. You can increase this to make room for tick mark labels or an x-axis label
#top — the top margin, with a default of 0.9
#wspace — the horizontal space between adjacent subplots, with a default of 0.2
#hspace — the vertical space between adjacent subplots, with a default of 0.2
#For example, if we were adding space to the bottom of a graph by changing the bottom margin to 0.2 (instead of the default of 0.1), we would use the command:

#plt.subplots_adjust(bottom=0.2)
#We can also use multiple keyword arguments, if we need to adjust multiple margins. For instance, we could adjust both the top and the hspace:

#plt.subplots_adjust(top=0.95, hspace=0.25)
#Let’s use wspace to fix the figure above:

# Left Plot
plt.subplot(1, 2, 1)
plt.plot([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])

# Right Plot
plt.subplot(1, 2, 2)
plt.plot([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])

# Subplot Adjust
plt.subplots_adjust(wspace=0.35)

plt.show()

#-------------------------------------------


plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16])
plt.plot([0, 1, 2, 3, 4], [0, 1, 8, 27, 64])
plt.legend(['parabola', 'cubic'])
plt.show()

#-----

Number Code	String
0	best
1	upper right
2	upper left
3	lower left
4	lower right
5	right
6	center left
7	center right
8	lower center
9	upper center
10	center
Note: If you decide not to set a value for loc, it will default to choosing the “best” location.	

#------

plt.legend(['parabola', 'cubic'], loc=6)
plt.show()

#------

plt.plot([0, 1, 2, 3, 4], [0, 1, 4, 9, 16],
         label="parabola")
plt.plot([0, 1, 2, 3, 4], [0, 1, 8, 27, 64],
         label="cubic")
plt.legend() # Still need this command!
plt.show()

#--------------------------------------------------
#for multiple subplots
ax = plt.subplot(1, 1, 1)
#for only one subplots
ax = plt.subplot()

#ax arguments

ax.set_xticks([1, 2, 4])
ax.set_yticks([0.1, 0.6, 0.8])
ax.set_yticklabels(['10%', '60%', '80%'])

#plot mark rather than line
ax = plt.subplot()
plt.plot([1, 3, 3.5], [0.1, 0.6, 0.8], 'o')
ax.set_yticks([0.1, 0.6, 0.8])
ax.set_yticklabels(['10%', '60%', '80%'])

#--------------------------------------------------
plt.close('all') to clear all existing plots

# Figure 2
plt.figure(figsize=(4, 10)) 
plt.plot(x, parabola)
plt.savefig('tall_and_narrow.png')

#---------------------------------------------------

import codecademylib
from matplotlib import pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

visits_per_month = [9695, 7909, 10831, 12942, 12495, 16794, 14161, 12762, 12777, 12439, 10309, 8724]

# numbers of limes of different species sold each month
key_limes_per_month = [92.0, 109.0, 124.0, 70.0, 101.0, 79.0, 106.0, 101.0, 103.0, 90.0, 102.0, 106.0]
persian_limes_per_month = [67.0, 51.0, 57.0, 54.0, 83.0, 90.0, 52.0, 63.0, 51.0, 44.0, 64.0, 78.0]
blood_limes_per_month = [75.0, 75.0, 76.0, 71.0, 74.0, 77.0, 69.0, 80.0, 63.0, 69.0, 73.0, 82.0]

x_values = range(len(months))


# create your figure here
plt.figure(figsize = (12, 8))

ax1 = plt.subplot(1, 2, 1)
plt.plot(x_values, visits_per_month, marker = "o")
plt.xlabel ('Months')
plt.ylabel ('Visits')
plt.title ('Total Visits by Months')
ax1.set_xticks(x_values)
ax1.set_xticklabels(months)

ax2 = plt.subplot(1, 2, 2)
plt.plot(x_values, key_limes_per_month, color = 'green')
plt.plot(x_values, persian_limes_per_month, color = 'yellow')
plt.plot(x_values, blood_limes_per_month, color = 'red')
plt.title ('Limes Sales by Months')
plt.xlabel ('Months')
plt.ylabel ('Limes')
plt.legend(['Key', 'persian', 'blood'])
ax2.set_xticks(x_values)
ax2.set_xticklabels(months)

plt.subplots_adjust(wspace = 0.3)
plt.show()
plt.savefig('lime_sales.png')


#
# DIFFERENT PLOT TYPES
# Simple Bar Chart
# The plt.bar function allows you to create simple bar charts to compare multiple categories of data.

# Some possible data that would be displayed with a bar chart:

# x-axis — famous buildings, y-axis — heights
# x-axis — different planets, y-axis — number of days in the year
# x-axis — programming languages, y-axis — lines of code written by you
# You call plt.bar with two arguments:

# the x-values — a list of x-positions for each bar
# the y-values — a list of heights for each bar
# In most cases, we will want our x-values to be a list that looks like [0, 1, 2, 3 ...] and has the same number of elements as our y-values list. We can create that list manually, but we can also use the following code:

heights = [88, 225, 365, 687, 4333, 10756, 30687, 60190, 90553]
x_values = range(len(heights))
# The range function creates a list of consecutive integers (i.e., [0, 1, 2, 3, ...]). It needs an argument to tell it how many numbers should be in the list. For instance, range(5) would make a list with 5 numbers. We want our list to be as long as our bar heights (heights in this example). len(heights) tell us how many elements are in the list heights.

# Here is an example of how to make a bar chart using plt.bar to compare the number of days in a year on the different planets:

days_in_year = [88, 225, 365, 687, 4333, 10756, 30687, 60190, 90553]
plt.bar(range(len(days_in_year)),
        # days_in_year)
plt.show()
# The result of this is:

# planet_bar_chart

# At this point, it’s hard to tell what this represents, because it’s unclearly labeled. We’ll fix that in later sections!

# In the instructions below, we’ll use plt.bar to create a chart for a fake cafe called MatplotSip. We will be comparing the sales of different beverages on a given day.

#----------------------------------------------------------------

# DIFFERENT PLOT TYPES
# Simple Bar Chart II
# When we create a bar chart, we want each bar to be meaningful and correspond to a category of data. In the drinks chart from the last exercise, we could see that sales were different for different drink items, but this wasn’t very helpful to us, since we didn’t know which bar corresponded to which drink.

# In the previous lesson, we learned how to customize the tick marks on the x-axis in three steps:

# Create an axes object
ax = plt.subplot()
# Set the x-tick positions using a list of numbers
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
# Set the x-tick labels using a list of strings
ax.set_xticklabels(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'])
# If your labels are particularly long, you can use the rotation keyword to rotate your labels by a specified number of degrees:
ax.set_xticklabels(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'],rotation=30)
# Note: We have to set the x-ticks before we set the x-labels because the default ticks won’t necessarily be one tick per bar, especially if we’re plotting a lot of bars. If we skip setting the x-ticks before the x-labels, we might end up with labels in the wrong place.

# Remember from Lesson I that we can label the x-axis (plt.xlabel) and y-axis (plt.ylabel) as well. Now, our graph is much easier to understand:labeled_planet_chart

# Let’s add the appropriate labels for the chart you made in the last exercise for the coffee shop, MatplotSip.

#------------------------------------------

import codecademylib
from matplotlib import pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales =  [91, 76, 56, 66, 52, 27]

plt.bar(range(len(drinks)), sales)

#create your ax object here

ax = plt.subplot()
ax.set_xticks(range(len(drinks)))
ax.set_xticklabels(drinks, rotation = 30)

plt.show()

#------------------------------------------

# DIFFERENT PLOT TYPES
# Side-By-Side Bars
# We can use a bar chart to compare two sets of data with the same types of axis values. To do this, we plot two sets of bars next to each other, so that the values of each category can be compared. For example, here is a chart with side-by-side bars for the populations of the United States and China over the age of 65 (in percentages):population_bars

# (data taken from World Bank)

# Some examples of data that side-by-side bars could be useful for include:

# the populations of two countries over time
# prices for different foods at two different restaurants
# enrollments in different classes for males and females
# In the graph above, there are 7 sets of bars, with 2 bars in each set. Each bar has a width of 0.8 (the default width for all bars in Matplotlib).

# If our first blue bar is at x=0, then we want the next blue bar to be at x=2, and the next to be at x=4, etc.
# Our first orange bar should be at x=0.8 (so that it is touching the blue bar), and the next orange bar should be at x=2.8, etc.
# This is a lot of math, but we can make Python do it for us by copying and pasting this code:

# China Data (blue bars)
n = 1  # This is our first dataset (out of 2)
t = 2 # Number of datasets
d = 7 # Number of sets of bars
w = 0.8 # Width of each bar
x_values1 = [t*element + w*n for element
             in range(d)]
# That just generated the first set of x-values. To generate the second set, paste the code again, but change n to 2, because this is the second dataset:

# US Data (orange bars)
n = 2  # This is our second dataset (out of 2)
t = 2 # Number of datasets
d = 7 # Number of sets of bars
w = 0.8 # Width of each bar
x_values2 = [t*element + w*n for element
             in range(d)]
# Let’s examine our special code:

[t*element + w*n for element in range(d)]
# This is called a list comprehension. It’s a special way of generating a list from a formula. You can learn more about it in this article. For making side-by-side bar graphs, you’ll never need to change this line; just paste it into your code and make sure to define n, t, d, and w correctly.

# In the instructions below, we’ll experiment with side-by-side bars to compare different locations of the MatplotSip coffee empire.

#------------------------------------------------

import codecademylib
from matplotlib import pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]

#Paste the x_values code here

n = 1  # This is our first dataset (out of 2)
t = 2 # Number of datasets
d = len(sales1) # Number of sets of bars
w = 0.8 # Width of each bar
store1_x = [t*element + w*n for element
             in range(d)]
plt.bar(store1_x, sales1)

n = 2  # This is our second dataset (out of 2)
t = 2 # Number of datasets
d = len(sales2) # Number of sets of bars
w = 0.8 # Width of each bar
store2_x = [t*element + w*n for element
             in range(d)]
plt.bar(store2_x, sales2)


plt.show()

#--------------------------------------------------------------

# DIFFERENT PLOT TYPES
# Stacked Bars
# If we want to compare two sets of data while preserving knowledge of the total between them, we can also stack the bars instead of putting them side by side. For instance, if someone was plotting the hours they’ve spent on entertaining themselves with video games and books in the past week, and wanted to also get a feel for total hours spent on entertainment, they could create a stacked bar chart:

# entertainment

# We do this by using the keyword bottom. The top set of bars will have bottom set to the heights of the other set of bars. So the first set of bars is plotted normally:

video_game_hours = [1, 2, 2, 1, 2]

plt.bar(range(len(video_game_hours)),
  video_game_hours) 
# and the second set of bars has bottom specified:

book_hours = [2, 3, 4, 2, 1]

plt.bar(range(len(book_hours)),
  book_hours,
  bottom=video_game_hours)
# This starts the book_hours bars at the heights of the video_game_hours bars. So, for example, on Monday the orange bar representing hours spent reading will start at a value of 1 instead of 0, because 1 hour was spent playing video games.

# Let’s try this out with the MatplotSip data from the last exercise.

#-------------------------------------------------

import codecademylib
from matplotlib import pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]

plt.bar(range(len(sales1)), sales1)
plt.bar(range(len(sales2)), sales2, bottom = sales1)

plt.legend(['Location 1','Location 2'])

plt.show()

#-----------------------------------
values = [10, 13, 11, 15, 20]
yerr = 2
plt.bar(range(len(values)), values, yerr=yerr, capsize=10)
plt.show()

values = [10, 13, 11, 15, 20]
yerr = [1, 3, 0.5, 2, 4]
plt.bar(range(len(values)), values, yerr=yerr, capsize=10)
plt.show()
#--------------------------------------

x_values = range(10)
y_values = [10, 12, 13, 13, 15, 19, 20, 22, 23, 29]
y_lower = [8, 10, 11, 11, 13, 17, 18, 20, 21, 27]
y_upper = [12, 14, 15, 15, 17, 21, 22, 24, 25, 31]

plt.fill_between(x_values, y_lower, y_upper, alpha=0.2) #this is the shaded error
plt.plot(x_values, y_values) #this is the line itself
plt.show()


# Having to calculate y_lower and y_upper by hand is time-consuming. If we try to just subtract 2 from y_values, we will get an error.

# TypeError: unsupported operand type(s) for -: 'list' and 'int'
# In order to correctly add or subtract from a list, we need to use list comprehension:

y_lower = [i - 2 for i in y_values]


#--------------------------------------------------

import codecademylib
from matplotlib import pyplot as plt

months = range(12)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
revenue = [16000, 14000, 17500, 19500, 21500, 21500, 22000, 23000, 20000, 19500, 18000, 16500]

y_lower = [i*0.9 for i in revenue]
y_upper = [i*1.1 for i in revenue]


#your work here
plt.plot(months, revenue)
plt.fill_between(months, y_lower, y_upper, alpha = 0.2)

ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)

plt.show()

#----------------------------------------------------------------------------
# Pie chart
budget_data = [500, 1000, 750, 300, 100]

plt.pie(budget_data)
plt.show()




import codecademylib
from matplotlib import pyplot as plt
import numpy as np

payment_method_names = ["Card Swipe", "Cash", "Apple Pay", "Other"]
payment_method_freqs = [270, 77, 32, 11]

#make your pie chart here
plt.pie(payment_method_freqs)
plt.axis('equal')

plt.show()

#-----------------------------------------------
# Pie Chart Labeling Method 1

budget_data = [500, 1000, 750, 300, 100]
budget_categories = ['marketing', 'payroll', 'engineering', 'design', 'misc']

plt.pie(budget_data)
plt.legend(budget_categories)

# Pie Chart Labeling Method 2

#option 2
plt.pie(budget_data, labels=budget_categories)

# Pie Chart percentages

'%0.2f' — 2 decimal places, like 4.08
'%0.2f%%' — 2 decimal places, but with a percent sign at the end, like 4.08%. You need two consecutive percent signs because the first one acts as an escape character, so that the second one gets displayed on the chart.
'%d%%' — rounded to the nearest int and with a percent sign at the end, like 4%.
So, a full call to plt.pie might look like:

plt.pie(budget_data,
        labels=budget_categories,
        autopct='%0.1f%%')

		
#--------------------------

DIFFERENT PLOT TYPES
Histogram
Sometimes we want to get a feel for a large dataset with many samples beyond knowing just the basic metrics of mean, median, or standard deviation. To get more of an intuitive sense for a dataset, we can use a histogram to display all the values.

A histogram tells us how many values in a dataset fall between different sets of numbers (i.e., how many numbers fall between 0 and 10? Between 10 and 20? Between 20 and 30?). Each of these questions represents a bin, for instance, our first bin might be between 0 and 10.

All bins in a histogram are always the same size. The width of each bin is the distance between the minimum and maximum values of each bin. In our example, the width of each bin would be 10.

Each bin is represented by a different rectangle whose height is the number of elements from the dataset that fall within that bin.

Here is an example:

histogram

To make a histogram in Matplotlib, we use the command plt.hist. plt.hist finds the minimum and the maximum values in your dataset and creates 10 equally-spaced bins between those values.

The histogram above, for example, was created with the following code:

plt.hist(dataset) 
plt.show()
If we want more than 10 bins, we can use the keyword bins to set how many bins we want to divide the data into. The keyword range selects the minimum and maximum values to plot. For example, if we wanted to take our data from the last example and make a new histogram that just displayed the values from 66 to 69, divided into 40 bins (instead of 10), we could use this function call:

plt.hist(dataset, range=(66,69), bins=40)
which would result in a histogram that looks like this:

histogram_range

Histograms are best for showing the shape of a dataset. For example, you might see that values are close together, or skewed to one side. With this added intuition, we often discover other types of analysis we want to perform.

#-------------------------------------------------

Multiple Histogram
######
use the keyword alpha, which can be a value between 0 and 1. This sets the transparency of the histogram. A value of 0 would make the bars entirely transparent. A value of 1 would make the bars completely opaque.

plt.hist(a, range=(55, 75), bins=20, alpha=0.5)
plt.hist(b, range=(55, 75), bins=20, alpha=0.5)
#######
use the keyword histtype with the argument 'step' to draw just the outline of a histogram:

plt.hist(a, range=(55, 75), bins=20, histtype='step')
plt.hist(b, range=(55, 75), bins=20, histtype='step')
#######
Another problem we face is that our histograms might have different numbers of samples, making one much bigger than the other. We can see how this makes it difficult to compare qualitatively, by adding a dataset b with a much bigger size value:

a = normal(loc=64, scale=2, size=10000)
b = normal(loc=70, scale=2, size=100000)

plt.hist(a, range=(55, 75), bins=20)
plt.hist(b, range=(55, 75), bins=20)
plt.show()

To solve this, we can normalize our histograms using normed=True. This command divides the height of each column by a constant such that the total shaded area of the histogram sums to 1.

a = normal(loc=64, scale=2, size=10000)
b = normal(loc=70, scale=2, size=100000)

plt.hist(a, range=(55, 75), bins=20, alpha=0.5, normed=True)
plt.hist(b, range=(55, 75), bins=20, alpha=0.5, normed=True)
plt.show()

#--------------------------------
import codecademylib
from matplotlib import pyplot as plt

past_years_averages = [82, 84, 83, 86, 74, 84, 90]
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006]
error = [1.5, 2.1, 1.2, 3.2, 2.3, 1.7, 2.4]

# Make your chart here
plt.figure(figsize = (10,8))
plt.bar(range(len(past_years_averages)), past_years_averages, yerr = error, capsize= 10)

plt.axis([-0.5, 6.5, 70, 95])
plt.xlabel('Year')
plt.ylabel('Test average')
plt.title('Final Exam Averages')

ax = plt.subplot()
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)


plt.show()
plt.savefig('my_bar_chart.png')

#----------------------------------------------------
import codecademylib
from matplotlib import pyplot as plt

unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']
middle_school_a = [80, 85, 84, 83, 86]
middle_school_b = [73, 78, 77, 82, 86]

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
#school_a_x = [0.8, 2.8, 4.8, 6.8, 8.8]
#school_b_x = [1.6, 3.6, 5.6, 7.6, 9.6]
# Make your chart here

school_a_x = create_x(2, 0.8, 1, len(middle_school_a))
school_b_x = create_x(2, 0.8, 2, len(middle_school_b))
middle_x = [(a+b)/2 for a,b in zip(school_a_x, school_b_x)] 
#middle_x for label position

plt.figure(figsize = (10, 8))

plt.xlabel('Unit')
plt.ylabel('Test Average')
plt.title('Test Averages on Different Units')
plt.axis([0, 10.6, 70, 90])

ax = plt.subplot()
ax.set_xticks(middle_x)
ax.set_xticklabels(unit_topics)

plt.bar(school_a_x, middle_school_a)
plt.bar(school_b_x, middle_school_b)
plt.legend(['Middle School A', 'Middle School B'])


plt.show()
plt.savefig('my_side_by_side.png')

------------------------------------------

import codecademylib
from matplotlib import pyplot as plt
import numpy as np

unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']
As = [6, 3, 4, 3, 5]
Bs = [8, 12, 8, 9, 10]
Cs = [13, 12, 15, 13, 14]
Ds = [2, 3, 3, 2, 1]
Fs = [1, 0, 0, 3, 0]

x = range(5)

c_bottom = np.add(As, Bs)
#create d_bottom and f_bottom here
d_bottom = np.add(c_bottom, Cs)
f_bottom = np.add(d_bottom, Ds)

#create your plot here
plt.figure(figsize = (10, 8))

plt.bar(range(len(unit_topics)), As)
plt.bar(range(len(unit_topics)), Bs, bottom = As)
plt.bar(range(len(unit_topics)), Cs, bottom = c_bottom)
plt.bar(range(len(unit_topics)), Ds, bottom = d_bottom)
plt.bar(range(len(unit_topics)), Fs, bottom = f_bottom)

ax = plt.subplot()
ax.set_xticks(range(len(unit_topics)))
ax.set_xticklabels(unit_topics)
plt.xlabel('Unit')
plt.ylabel('Number of Students')
plt.title('Grade distribution')

plt.savefig('my_stacked_bar.png')    

plt.show()

#-----------------------------------------
import codecademylib
from matplotlib import pyplot as plt

exam_scores1 = [62.58, 67.63, 81.37, 52.53, 62.98, 72.15, 59.05, 73.85, 97.24, 76.81, 89.34, 74.44, 68.52, 85.13, 90.75, 70.29, 75.62, 85.38, 77.82, 98.31, 79.08, 61.72, 71.33, 80.77, 80.31, 78.16, 61.15, 64.99, 72.67, 78.94]
exam_scores2 = [72.38, 71.28, 79.24, 83.86, 84.42, 79.38, 75.51, 76.63, 81.48,78.81,79.23,74.38,79.27,81.07,75.42,90.35,82.93,86.74,81.33,95.1,86.57,83.66,85.58,81.87,92.14,72.15,91.64,74.21,89.04,76.54,81.9,96.5,80.05,74.77,72.26,73.23,92.6,66.22,70.09,77.2]

# Make your plot here
plt.figure(figsize = (10,8))

plt.hist(exam_scores1, bins = 12, normed = True, histtype = 'step', linewidth = 2)
plt.hist(exam_scores2, bins = 12, normed = True, histtype = 'step', linewidth = 2)

plt.legend(['1st Yr Teaching', '2nd Yr Teaching'])
plt.xlabel('Percentage')
plt.ylabel('Frequency')
plt.title('Final Exam Score Distribution')

plt.show()

plt.savefig('my_histogram.png')
#----------------------------------

import codecademylib
from matplotlib import pyplot as plt

unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']
num_hardest_reported = [1, 3, 10, 15, 1]

#Make your plot here
plt.figure(figsize = (10, 8))

plt.pie(num_hardest_reported, labels = unit_topics, autopct = '%d%%')
plt.axis('equal')
plt.title('Hardest Topics')

plt.show()
plt.savefig('my_pie_chart.png')

#-----------------------------------------
import codecademylib
from matplotlib import pyplot as plt

hours_reported =[3, 2.5, 2.75, 2.5, 2.75, 3.0, 3.5, 3.25, 3.25,  3.5, 3.5, 3.75, 3.75,4, 4.0, 3.75,  4.0, 4.25, 4.25, 4.5, 4.5, 5.0, 5.25, 5, 5.25, 5.5, 5.5, 5.75, 5.25, 4.75]
exam_scores = [52.53, 59.05, 61.15, 61.72, 62.58, 62.98, 64.99, 67.63, 68.52, 70.29, 71.33, 72.15, 72.67, 73.85, 74.44, 75.62, 76.81, 77.82, 78.16, 78.94, 79.08, 80.31, 80.77, 81.37, 85.13, 85.38, 89.34, 90.75, 97.24, 98.31]

plt.figure(figsize=(10,8))

# Create your hours_lower_bound and hours_upper_bound lists here 

hours_lower_bound = [x*0.8 for x in hours_reported]
hours_upper_bound = [x*1.2 for x in hours_reported]


# Make your graph here
plt.plot(exam_scores, hours_reported, linewidth = 2)
plt.fill_between(exam_scores, hours_lower_bound, hours_upper_bound, alpha = 0.2 )

plt.xlabel('Score')
plt.ylabel('Hours studying (self-reported)')
plt.title('Time spent studying vs final exam scores')

plt.show()
plt.savefig('my_line_graph.png')

'''
How to Select a Meaningful Visualization
This article will guide you through the process of selecting a graph for a visualization.

Brainstorming your visualization
The three steps in the data visualization process are preparing, visualizing, and styling data. When faced with a blank canvas, the second step of the process, visualizing the data, can be overwhelming. To help, we’ve created a diagram to guide the selection of a chart based on what you want to explore in your data.

When planning out a visualization, you’ll usually have an idea of what questions you’ll want to explore. However, you may initially wonder exactly which chart to use. This moment is one of the most exciting parts of the process!

During your brainstorming phase, you should consider two things:

The focusing question you want to answer with your chart
The type of data that you want to visualize
Depending on the focusing questions you’re trying to answer, the type of chart you select should be different and intentional in its difference. In the diagram below, we have assigned Matplotlib visualizations to different categories. These categories explore common focusing questions and types of data you may want to display in a visualization.

A Diagram of Diagrams!
SVG

Chart categories
Composition charts
Focusing Question: What are the parts of some whole? What is the data made of?

Datasets that work well: Data pertaining to probabilities, proportions, and percentages can be visualized as with the graphs in this composition category. Charts in this category illustrate the different data components and their percentages as part of a whole.

Distribution Charts
Datasets that work well: Data in large quantities and/or with an array of attributes works well for these types of charts. Visualizations in this category will allow you to see patterns, re-occurrences, and a clustering of data points.

Note: In statistics, a commonly seen distribution is a bell curve, also known as a normal distribution. A bell curve is a bell-shaped distribution where most of the values in the dataset crowd around the average (also known as the mean), therefore causing the curve to form. If you want to see how values in the data are “distributed” across variables, the best way to do that would be with the visualizations in this category.

Relationship Charts
Focusing Question: How do variables relate to each other?

Datasets that work well: Data with two or more variables can be displayed in these charts. These charts typically illustrate a correlation between two or more variables. You can communicate this relationship by mapping multiple variables in the same chart. Correlation measures the strength of a relationship between variables.

Comparison Charts
Focusing Question: How do variables compare to each other?

Datasets that work well: Data must have multiple variables, and the visualizations in this category allow readers to compare those items against the others. For example, a line graph that has multiple lines, each belonging to a different variable. Multi-colored bar charts are also a great way to compare items in data.

Summary
When brainstorming a visualization, use the diagram above to guide the selection of your chart. Remember to be intentional in your selection by thinking about what type of data you’re dealing with and what focusing question you wish to answer.'''

#sample from https://matplotlib.org/gallery/mplot3d/subplot3d.html?highlight=add_subplot

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d.axes3d import get_test_data


# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

#===============
#  First subplot
#===============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

#===============
# Second subplot
#===============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
X, Y, Z = get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()

'''
Project: Visualizing the Orion Constellation

In this project you are Dr. Jillian Bellovary, a real-life astronomer for the Hayden Planetarium at the American Museum of Natural History. As an astronomer, part of your job is to study the stars. You've recently become interested in the constellation Orion, a collection of stars that appear in our night sky and form the shape of Orion, a warrior God from ancient Greek mythology. 

As a researcher on the Hayden Planetarium team, you are in charge of visualizing the Orion constellation in 3D using the Matplotlib function .scatter(). To learn more about the .scatter() you can see the Matplotlib documentation here. 

You will create a rotate-able visualization of the position of the Orion's stars and get a better sense of their actual positions. To achieve this, you will be mapping real data from outer space that maps the position of the stars in the sky

The goal of the project is to understand spatial perspective. Once you visualize Orion in both 2D and 3D, you will be able to see the difference in the constellation shape humans see from earth versus the actual position of the stars that make up this constellation. 

'''

'''
1. Set-Up

The following set-up is new and specific to the project. It is very similar to the way you have imported Matplotlib in previous lessons.

•Add %matplotlib notebook in the cell below. This is a new statement that you may not have seen before. It will allow you to be able to rotate your visualization in this jupyter notebook.


•We will be using a subset of Matplotlib: matplotlib.pyplot. Import the subset as you have been importing it in previous lessons: from matplotlib import pyplot as plt

•In order to see our 3D visualization, we also need to add this new line after we import Matplotlib: from mpl_toolkits.mplot3d import Axes3D
'''

#allow you to be able to rotate your visualization in this jupyter notebook
%matplotlib notebook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
2. Get familiar with real data¶

Astronomers describe a star's position in the sky by using a pair of angles: declination and right ascension. Declination is similar to longitude, but it is projected on the celestian fear. Right ascension is known as the "hour angle" because it accounts for time of day and earth's rotaiton. Both angles are relative to the celestial equator. You can learn more about star position here.

The x, y, and z lists below are composed of the x, y, z coordinates for each star in the collection of stars that make up the Orion constellation as documented in a paper by Nottingham Trent Univesity on "The Orion constellation as an installation" found here.

Spend some time looking at x, y, and z, does each fall within a range?
'''

# Orion
x = [-0.41, 0.57, 0.07, 0.00, -0.29, -0.32,-0.50,-0.23, -0.23]
y = [4.12, 7.71, 2.36, 9.10, 13.35, 8.13, 7.19, 13.25,13.43]
z = [2.06, 0.84, 1.56, 2.07, 2.36, 1.72, 0.66, 1.25,1.38]

'''
3. Create a 2D Visualization

Before we visualize the stars in 3D, let's get a sense of what they look like in 2D. 

Create a figure for the 2d plot and save it to a variable name fig. (hint: plt.figure())

Add your subplot .add_subplot() as the single subplot, with 1,1,1.(hint: add_subplot(1,1,1))

Use the scatter function to visualize your x and y coordinates. (hint: .scatter(x,y))

Render your visualization. (hint: plt.show())

Does the 2D visualization look like the Orion constellation we see in the night sky? Do you recognize its shape in 2D? There is a curve to the sky, and this is a flat visualization, but we will visualize it in 3D in the next step to get a better sense of the actual star positions. 
'''
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
plt.scatter(x,y)
plt.show()


'''
4. Create a 3D Visualization

Create a figure for the 3D plot and save it to a variable name fig_3d. (hint: plt.figure())

Since this will be a 3D projection, we want to make to tell Matplotlib this will be a 3D plot. 

To add a 3D projection, you must include a the projection argument. It would look like this:
projection="3d"

Add your subplot with .add_subplot() as the single subplot 1,1,1 and specify your projection as 3d:

fig_3d.add_subplot(1,1,1,projection="3d"))

Since this visualization will be in 3D, we will need our third dimension. In this case, our z coordinate. 

Create a new variable constellation3d and call the scatter function with your x, y and z coordinates. 

Include z just as you have been including the other two axes. (hint: .scatter(x,y,z))

Render your visualization. (hint plt.show().)
'''

fig_3d = plt.figure()
fig_3d.add_subplot(1,1,1, projection="3d")
constellation3d = plt.scatter(x,y,z)
plt.show()

'''

5. Rotate and explore

Use your mouse to click and drag the 3D visualization in the previous step. This will rotate the scatter plot. As you rotate, can you see Orion from different angles? 

Note: The on and off button that appears above the 3D scatter plot allows you to toggle rotation of your 3D visualization in your notebook.

Take your time, rotate around! Remember, this will never look exactly like the Orion we see from Earth. The visualization does not curve as the night sky does. There is beauty in the new understanding of Earthly perspective! We see the shape of the warrior Orion because of Earth's location in the universe and the location of the stars in that constellation.

Feel free to map more stars by looking up other celestial x, y, z coordinates here.'''


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- SAMPLES #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

import codecademylib3_seaborn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

games = ["LoL", "Dota 2", "CS:GO", "DayZ", "HOS", "Isaac", "Shows", "Hearth", "WoT", "Agar.io"]

viewers =  [1070, 472, 302, 239, 210, 171, 170, 90, 86, 71]

plt.bar(range(len(games)), viewers, color='slateblue')

plt.legend(["Twitch"])

plt.xlabel('Games')
plt.ylabel('Viewers')

ax = plt.subplot()

ax.set_xticks(range(0, 10))

ax.set_xticklabels(games, rotation=30)

plt.show()

# We can add plt.clf() to clear the current figure, our bar graph, before creating our next figure, the pie chart

plt.clf()



labels = ["US", "DE", "CA", "N/A", "GB", "TR", "BR", "DK", "PL", "BE", "NL", "Others"]

countries = [447, 66, 64, 49, 45, 28, 25, 20, 19, 17, 17, 279]

colors = ['lightskyblue', 'gold', 'lightcoral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna', 'khaki', 'gold', 'violet', 'yellowgreen']

# Make your pie chart here

explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

plt.pie(countries, explode=explode, colors=colors, shadow=True, startangle=345, autopct='%1.0f%%', pctdistance=1.15, labels = lavels)

plt.title("League of Legends Viewers' Whereabouts")

plt.legend(labels, loc="right")

plt.show()

# We can add plt.clf() to clear the current figure, our pie chart,before creating our next figure, the line graph

plt.clf()

hour = range(24)

viewers_hour = [30, 17, 34, 29, 19, 14, 3, 2, 4, 9, 5, 48, 62, 58, 40, 51, 69, 55, 76, 81, 102, 120, 71, 63]

plt.title("Time Series")

plt.xlabel("Hour")
plt.ylabel("Viewers")

plt.plot(hour, viewers_hour)

plt.legend(['2015-01-01'])

ax = plt.subplot()

ax.set_xticks(hour)
ax.set_yticks([0, 20, 40, 60, 80, 100, 120])

y_upper = [i + (i*0.15) for i in viewers_hour]
y_lower = [i - (i*0.15) for i in viewers_hour]

plt.fill_between(hour, y_lower, y_upper, alpha=0.2)

plt.show()



#-#-#-#-#-#-#-#-#-#-#-#-  matplotlib Tutorials #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

https://matplotlib.org/tutorials/index.html


