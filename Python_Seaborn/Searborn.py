
#-#-#-#-#-#-#-#-#-#-#-#-#-#-SEABORN #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

'''LEARN SEABORN INTRODUCTION
Plotting Bars with Seaborn
Take a look at the file called results.csv. You’ll plot that data soon, but before you plot it, take a minute to understand the context behind that data, which is based on a hypothetical situation we have created:

Suppose we are analyzing data from a survey: we asked 1,000 patients at a hospital how satisfied they were with their experience. Their response was measured on a scale of 1 - 10, with 1 being extremely unsatisfied, and 10 being extremely satisfied. We have summarized that data in a CSV file called results.csv.

To plot this data using Matplotlib, you would write the following:
'''

df = pd.read_csv("results.csv")
ax = plt.subplot()
plt.bar(range(len(df)),
        df["Mean Satisfaction"])
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df.Gender)
plt.xlabel("Gender")
plt.ylabel("Mean Satisfaction")
'''

That's a lot of work for a simple bar chart! Seaborn gives us a much simpler option. With Seaborn, you can use the `sns.barplot()` command to do the same thing.
The Seaborn function sns.barplot(), takes at least three keyword arguments:

data: a Pandas DataFrame that contains the data (in this example, data=df)
x: a string that tells Seaborn which column in the DataFrame contains otheur x-labels (in this case, x="Gender")
y: a string that tells Seaborn which column in the DataFrame contains the heights we want to plot for each bar (in this case y="Mean Satisfaction")
By default, Seaborn will aggregate and plot the mean of each category. In the next exercise you will learn more about aggregation and how Seaborn handles it.'''



import codecademylib3_seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load results.csv here:
df = pd.read_csv('results.csv')

print(df)

sns.barplot(
  data= df,
  x= 'Gender',
	y= 'Mean Satisfaction'
)

plt.show()





'''
LEARN SEABORN INTRODUCTION
Understanding Aggregates
Seaborn can also calculate aggregate statistics for large datasets. To understand why this is helpful, we must first understand what an aggregate is.

An aggregate statistic, or aggregate, is a single number used to describe a set of data. One example of an aggregate is the average, or mean of a data set. There are many other aggregate statistics as well.

Suppose we have a grade book with columns student, assignment_name, and grade, as shown below.

student	assignment_name	grade
Amy	Assignment 1	75
Amy	Assignment 2	82
Bob	Assignment 1	99
Bob	Assignment 2	90
Chris	Assignment 1	72
Chris	Assignment 2	66
…	…	…
To calculate a student’s current grade in the class, we need to aggregate the grade data by student. To do this, we’ll calculate the average of each student’s grades, resulting in the following data set:

student	grade
Amy	78.5
Bob	94.5
Chris	69
…	…
On the other hand, we may be interested in understanding the relative difficulty of each assignment. In this case, we would aggregate by assignment, taking the average of all student’s scores on each assignment:

assignment_name	grade
Assignment 1	82
Assignment 2	79.3
…	…
In both of these cases, the function we used to aggregate our data was the average or mean, but there are many types of aggregate statistics including:

Median
Mode
Standard Deviation
In Python, you can compute aggregates fairly quickly and easily using Numpy, a popular Python library for computing. You’ll use Numpy in this exercise to compute aggregates for a DataFrame.'''

#plot mean grade against assignment
import codecademylib3_seaborn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

gradebook = pd.read_csv("gradebook.csv")
#print(gradebook)

assignment1 = gradebook[gradebook['assignment_name'] == 'Assignment 1']
print(assignment1)

asn1_median = np.median(assignment1['grade'])
print(asn1_median)

#plot mean grade against assignment, done with seaborn

import codecademylib3_seaborn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

gradebook = pd.read_csv("gradebook.csv")

sns.barplot(
  data = gradebook,
  x = 'assignment_name',
  y = 'grade'
)

plt.show()

'''
LEARN SEABORN INTRODUCTION
Modifying Error Bars
By default, Seaborn will place error bars on each bar when you use the barplot() function.

Error bars are the small lines that extend above and below the top of each bar. Errors bars visually indicate the range of values that might be expected for that bar.


For example, in our assignment average example, an error bar might indicate what grade we expect an average student to receive on this assignment.


There are several different calculations that are commonly used to determine error bars.

By default, Seaborn uses something called a bootstrapped confidence interval. Roughly speaking, this interval means that “based on this data, 95% of similar situations would have an outcome within this range”.

In our gradebook example, the confidence interval for the assignments means “if we gave this assignment to many, many students, we’re confident that the mean score on the assignment would be within the range represented by the error bar”.

The confidence interval is a nice error bar measurement because it is defined for different types of aggregate functions, such as medians and mode, in addition to means.

If you’re calculating a mean and would prefer to use standard deviation for your error bars, you can pass in the keyword argument ci="sd" to sns.barplot() which will represent one standard deviation. It would look like this:

sns.barplot(data=gradebook, x="name", y="grade", ci="sd")'''


'''
LEARN SEABORN INTRODUCTION
Calculating Different Aggregates
In most cases, we’ll want to plot the mean of our data, but sometimes, we’ll want something different:

If our data has many outliers, we may want to plot the median.
If our data is categorical, we might want to count how many times each category appears (such as in the case of survey responses).
Seaborn is flexible and can calculate any aggregate you want. To do so, you’ll need to use the keyword argument estimator, which accepts any function that works on a list.

For example, to calculate the median, you can pass in np.median to the estimator keyword:
'''
sns.barplot(data=df,
  x="x-values",
  y="y-values",
  estimator=np.median)
  
 '''
Consider the data in results.csv. To calculate the number of times a particular value appears in the Response column , we pass in len:'''

sns.barplot(data=df,
  x="Patient ID",
  y="Response",
  estimator=len)

#--------------------------- script.py---------------

import codecademylib3_seaborn
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("survey.csv")

print(df)

#sns.barplot(data = df,
#  x = 'Gender',
#  y = 'Response',
#  estimator = len
#  )

sns.barplot(data = df,
  x = 'Gender',
  y = 'Response',
  estimator = np.median
  )

plt.show()

'''
LEARN SEABORN INTRODUCTION
Aggregating by Multiple Columns
Sometimes we’ll want to aggregate our data by multiple columns to visualize nested categorical variables.

For example, consider our hospital survey data. The mean satisfaction seems to depend on Gender, but it might also depend on another column: Age Range.

We can compare both the Gender and Age Range factors at once by using the keyword hue.
'''
sns.barplot(data=df,
            x="Gender",
            y="Response",
            hue="Age Range")
            
            '''
The hue parameter adds a nested categorical variable to the plot.

*Visualizing survey results by gender with age range nested*.
Notice that we keep the same x-labels, but we now have different color bars representing each Age Range. We can compare two bars of the same color to see how patients with the same Age Range, but different Gender rated the survey.'''

'''
LEARN SEABORN INTRODUCTION
Review
In this lesson you learned how to extend Matplotlib with Seaborn to create meaningful visualizations from data in DataFrames.

You’ve also learned how Seaborn creates aggregated charts and how to change the way aggregates and error bars are calculated.

Finally, you learned how to aggregate by multiple columns, and how the hue parameter adds a nested categorical variable to a visualization.

To review the seaborn workflow:
1. Ingest data from a CSV file to Pandas DataFrame.'''
df = pd.read_csv('file_name.csv')'''
2. Set sns.barplot() with desired values for x, y, and set data equal to your DataFrame.'''
sns.barplot(data=df, x='X-Values', y='Y-Values')'''
3. Set desired values for estimator and hue parameters.'''
sns.barplot(data=df, x='X-Values', y='Y-Values', estimator=len, hue='Value')'''
4. Render the plot using plt.show().'''
plt.show()

'''

LEARN SEABORN: DISTRIBUTIONS
Introduction
In this lesson, we will explore how to use Seaborn to graph multiple statistical distributions, including box plots and violin plots.

Seaborn is optimized to work with large datasets — from its ability to natively interact with Pandas DataFrames, to automatically calculating and plotting aggregates. One of the most powerful aspects of Seaborn is its ability to visualize and compare distributions. Distributions provide us with more information about our data — how spread out it is, its range, etc.

Calculating and graphing distributions is integral to analyzing massive amounts of data. We’ll look at how Seaborn allows us to move beyond the traditional distribution graphs to plots that enable us to communicate important statistical information.'''

'''
Plotting Distributions with Seaborn
Seaborn's strength is in visualizing statistical calculations. Seaborn includes several plots that allow you to graph univariate distribution, including KDE plots, box plots, and violin plots. Explore the Jupyter notebook below to get an understanding of how each plot works.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
First, we'll read in three datasets. In order to plot them in Seaborn, we'll combine them using NumPy's .concatenate() function into a Pandas DataFrame.'''

n = 500
dataset1 = np.genfromtxt("dataset1.csv", delimiter=",")
dataset2 = np.genfromtxt("dataset2.csv", delimiter=",")
dataset3 = np.genfromtxt("dataset3.csv", delimiter=",")


df = pd.DataFrame({
    "label": ["set_one"] * n + ["set_two"] * n + ["set_three"] * n,
    "value": np.concatenate([dataset1, dataset2, dataset3])
})

sns.set()

'''
First, let's plot each dataset as bar charts.'''

sns.barplot(data=df, x='label', y='value')
plt.show()

'''

We can use barplots to find out information about the mean - but it doesn't give us a sense of how spread out the data is in each set. To find out more about the distribution, we can use a KDE plot.'''

sns.kdeplot(dataset1, shade=True, label="dataset1")
sns.kdeplot(dataset2, shade=True, label="dataset2")
sns.kdeplot(dataset3, shade=True, label="dataset3")

plt.legend()
plt.show()

'''
A KDE plot will give us more information, but it's pretty difficult to read this plot.'''

sns.boxplot(data=df, x='label', y='value')
plt.show()

'''
A box plot, on the other hand, makes it easier for us to compare distributions. It also gives us other information, like the interquartile range and any outliers. However, we lose the ability to determine the shape of the data.'''

sns.violinplot(data=df, x="label", y="value")
plt.show()

'''
A violin plot brings together shape of the KDE plot with additional information that a box plot provides. It's understandable why many people like this plot!'''



'''
LEARN SEABORN: DISTRIBUTIONS
Bar Charts Hide Information
Before we dive into these new charts, we need to understand why we’d want to use them. To best illustrate this idea, we need to revisit bar charts.

We previously learned that Seaborn can quickly aggregate data to plot bar charts using the mean.

Here is a bar chart that uses three different randomly generated sets of data:

sns.barplot(data=df, x="label", y="value")
plt.show()
alt

These three datasets look identical! As far as we can tell, they each have the same mean and similar confidence intervals.

We can get a lot of information from these bar charts, but we can’t get everything. For example, what are the minimum and maximum values of these datasets? How spread out is this data?

While we may not see this information in our bar chart, these differences might be significant and worth understanding better.'''

import codecademylib3_seaborn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Take in the data from the CSVs as NumPy arrays:
set_one = np.genfromtxt("dataset1.csv", delimiter=",")
set_two = np.genfromtxt("dataset2.csv", delimiter=",")
set_three = np.genfromtxt("dataset3.csv", delimiter=",")
set_four = np.genfromtxt("dataset4.csv", delimiter=",")

# Creating a Pandas DataFrame:
n=500
df = pd.DataFrame({
    "label": ["set_one"] * n + ["set_two"] * n + ["set_three"] * n + ["set_four"] * n,
    "value": np.concatenate([set_one, set_two, set_three, set_four])
})

# Setting styles:
sns.set_style("darkgrid")
sns.set_palette("pastel")

# Add your code below:
sns.barplot(data = df, 
           x = 'label',
           y = 'value')

plt.show()

'''
LEARN SEABORN: DISTRIBUTIONS
KDE Plots, Part I
Bar plots can tell us what the mean of our dataset is, but they don’t give us any hints as to the distribution of the dataset values. For all we know, the data could be clustered around the mean or spread out evenly across the entire range.

To find out more about each of these datasets, we’ll need to examine their distributions. A common way of doing so is by plotting the data as a histogram, but histograms have their drawback as well.

Seaborn offers another option for graphing distributions: KDE Plots.

KDE stands for Kernel Density Estimator. A KDE plot gives us the sense of a univariate as a curve. A univariate dataset only has one variable and is also referred to as being one-dimensional, as opposed to bivariate or two-dimensional datasets which have two variables.

KDE plots are preferable to histograms because depending on how you group the data into bins and the width of the bins, you can draw wildly different conclusions about the shape of the data. Using a KDE plot can mitigate these issues, because they smooth the datasets, allow us to generalize over the shape of our data, and aren’t beholden to specific data points.'''

#good youtube video explained what is KDE  https://www.youtube.com/watch?v=x5zLaWT5KPs

'''
To plot a KDE in Seaborn, we use the method sns.kdeplot().

A KDE plot takes the following arguments:

data - the univariate dataset being visualized, like a Pandas DataFrame, Python list, or NumPy array
shade - a boolean that determines whether or not the space underneath the curve is shaded
Let’s examine the KDE plots of our three datasets:
'''
sns.kdeplot(dataset1, shade=True)
sns.kdeplot(dataset2, shade=True)
sns.kdeplot(dataset3, shade=True)
plt.legend()
plt.show()

'''
For this lesson, the KDE plots we work will be using univariate data. So, only one of the axes will represent actual values in the data.

The horizontal or x-axis of a KDE plot is the range of values in the data set. This is similar to the x axis for histograms.

The vertical or y-axis of a KDE plot represents the Kernel Density Estimate of the Probability Density Function of a random variable, which is interpreted as a probability differential. The probability of a value being between the points x1 and x2 is the total shaded area of the curve under the two points.
'''

'''
LEARN SEABORN: DISTRIBUTIONS
Box Plots, Part I
While a KDE plot can tell us about the shape of the data, it’s cumbersome to compare multiple KDE plots at once. They also can’t tell us other statistical information, like the values of outliers.

The box plot (also known as a box-and-whisker plot) can’t tell us about how our dataset is distributed, like a KDE plot. But it shows us the range of our dataset, gives us an idea about where a significant portion of our data lies, and whether or not any outliers are present.

Let’s examine how we interpret a box plot:

The box represents the interquartile range
The line in the middle of the box is the median
The end lines are the first and third quartiles
The diamonds show outliers'''

'''
To plot a box plot in Seaborn, we use the method sns.boxplot().

A box plot takes the following arguments:

data - the dataset we’re plotting, like a DataFrame, list, or an array
x - a one-dimensional set of values, like a Series, list, or array
y - a second set of one-dimensional data
If you use a Pandas Series for the x and y values, the Series will also generate the axis labels. For example, if you use the value Series as your y value data, Seaborn will automatically apply that name as the y-axis label.'''


'''
LEARN SEABORN: DISTRIBUTIONS
Violin Plots, Part I
As we saw in the previous exercises, while it’s possible to plot multiple histograms, it is not a great option for comparing distributions. Seaborn gives us another option for comparing distributions - a violin plot. Violin plots provide more information than box plots because instead of mapping each individual data point, we get an estimation of the dataset thanks to the KDE.

Violin plots are less familiar and trickier to read, so let’s break down the different parts:

There are two KDE plots that are symmetrical along the center line.
A white dot represents the median.
The thick black line in the center of each violin represents the interquartile range.
The lines that extend from the center are the confidence intervals - just as we saw on the bar plots, a violin plot also displays the 95% confidence interval.'''

'''LEARN SEABORN: DISTRIBUTIONS
Violin Plots, Part II
Violin Plots are a powerful graphing tool that allows you to compare multiple distributions at once.

Let’s look at how our original three data sets look like as violin plots:

sns.violinplot(data=df, x="label", y="value")
plt.show()
alt

As we can see, violin plots allow us to graph and compare multiple distributions. It also retains the shape of the distributions, so we can easily tell that Dataset 1 is skewed left and that Dataset 3 is bimodal.

To plot a violin plot in Seaborn, use the method sns.violinplot().

There are several options for passing in relevant data to the x and y parameters:

data - the dataset that we’re plotting, such as a list, DataFrame, or array
x, y, and hue - a one-dimensional set of data, such as a Series, list, or array
any of the parameters to the function sns.boxplot()'''




'''
Seaborn Styling, Part 1: Figure Style and Scale
Learn how to customize your figures and scale plots for different presentation settings.

Introduction
When creating a data visualization, your goal is to communicate the insights found in the data. While visualizing communicates important information, styling will influence how your audience understands what you’re trying to convey.

After you have formatted and visualized your data, the third and last step of data visualization is styling. Styling is the process of customizing the overall look of your visualization, or figure. Making intentional decisions about the details of the visualization will increase their impact and set your work apart.

In this article, we’ll look at how to do the following techniques in Seaborn:

customize the overall look of your figure, using background colors, grids, spines, and ticks
scale plots for different contexts, such as presentations and reports
Customizing the Overall Look of Your Figure
Seaborn enables you to change the presentation of your figures by changing the style of elements like the background color, grids, and spines. When deciding how to style your figures, you should take into consideration your audience and the context. Is your visualization part of a report and needs to convey specific information? Or is it part of a presentation? Or is your visualization meant as its own stand-alone, with no narrator in front of it, and no other visualizations to compare it to?

In this section, we’ll explore three main aspects of customizing figures in Seaborn - background color, grids, and spines - and how these elements can change the look and meaning of your visualizations.

Built-in Themes
Seaborn has five built-in themes to style its plots: darkgrid, whitegrid, dark, white, and ticks. Seaborn defaults to using the darkgrid theme for its plots, but you can change this styling to better suit your presentation needs.

To use any of the preset themes pass the name of it to sns.set_style().
'''
sns.set_style("darkgrid")
sns.stripplot(x="day", y="total_bill", data=tips)
'''
image1

We’ll explore the rest of the themes in the examples below.

Background Color
When thinking about the look of your visualization, one thing to consider is the background color of your plot. The higher the contrast between the color palette of your plot and your figure background, the more legible your data visualization will be. Fun fact: dark blue on white is actually more legible than black on white!

The dark background themes provide a nice change from the Matplotlib styling norms, but doesn’t have as much contrast:
'''
sns.set_style("dark")
sns.stripplot(x="day", y="total_bill", data=tips)'''
image2

The white and tick themes will allow the colors of your dataset to show more visibly and provides higher contrast so your plots are more legible:
'''
sns.set_style("ticks")
sns.stripplot(x="day", y="total_bill", data=tips)'''
image3

Grids
In addition to being able to define the background color of your figure, you can also choose whether or not to include a grid. Remember that the default theme includes a grid.

It’s a good choice to use a grid when you want your audience to be able to draw their own conclusions about data. A grid allows the audience to read your chart and get specific information about certain values. Research papers and reports are a good example of when you would want to include a grid.
'''
sns.set_style("whitegrid")
sns.stripplot(x="day", y="total_bill", data=tips)'''
image4

There are also instances where it would make more sense to not use a grid. If you’re delivering a presentation, simplifying your charts in order to draw attention to the important visual details may mean taking out the grid. If you’re interested in making more specific design choices, then leaving out the grids might be part of that aesthetic decision.
'''
sns.set_style("white")
sns.stripplot(x="day", y="total_bill", data=tips)'''
image5

In this case, a blank background would allow your plot to shine.

Despine
In addition to changing the color background, you can also define the usage of spines. Spines are the borders of the figure that contain the visualization. By default, an image has four spines.

You may want to remove some or all of the spines for various reasons. A figure with the left and bottom spines resembles traditional graphs. You can automatically take away the top and right spines using the sns.despine()function. Note: this function must be called after you have called your plot.
'''
sns.set_style("white")
sns.stripplot(x="day", y="total_bill", data=tips)
sns.despine()'''
image6

Not including any spines at all may be an aesthetic decision. You can also specify how many spines you want to include by calling despine() and passing in the spines you want to get rid of, such as: left, bottom, top, right.
'''
sns.set_style("whitegrid")
sns.stripplot(x="day", y="total_bill", data=tips)
sns.despine(left=True, bottom=True)'''
image7

Scaling Figure Styles for Different Mediums
Matplotlib allows you to generate powerful plots, but styling those plots for different presentation purposes is difficult. Seaborn makes it easy to produce the same plots in a variety of different visual formats so you can customize the presentation of your data for the appropriate context, whether it be a research paper or a conference poster.

You can set the visual format, or context, using sns.set_context()

Within the usage of sns.set_context(), there are three levels of complexity:

Pass in one parameter that adjusts the scale of the plot
Pass in two parameters - one for the scale and the other for the font size
Pass in three parameters - including the previous two, as well as the rc with the style parameter that you want to override
Scaling Plots
Seaborn has four presets which set the size of the plot and allow you to customize your figure depending on how it will be presented.

In order of relative size they are: paper, notebook, talk, and poster. The notebook style is the default.
'''
sns.set_style("ticks")

# Smallest context:
sns.set_context("paper")
sns.stripplot(x="day", y="total_bill", data=tips)


sns.set_style("ticks")

# Largest Context:
sns.set_context("poster")
sns.stripplot(x="day", y="total_bill", data=tips)'''


Scaling Fonts and Line Widths
You are also able to change the size of the text using the font_scale parameter for sns.set_context()

You may want to also change the line width so it matches. We do this with the rc parameter, which we’ll explain in detail below.
'''
# Set font scale and reduce grid line width to match
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})
sns.stripplot(x="day", y="total_bill", data=tips)'''
image10

While you’re able to change these parameters, you should keep in mind that it’s not always useful to make certain changes. Notice in this example that we’ve changed the line width, but because of it’s relative size to the plot, it distracts from the actual plotted data.'''

# Set font scale and increase grid line width to match
sns.set_context("poster", font_scale = 1, rc={"grid.linewidth": 5})
sns.stripplot(x="day", y="total_bill", data=tips)'''
image11

The RC Parameter
As we mentioned above, if you want to override any of these standards, you can use sns.set_context and pass in the parameter rc to target and reset the value of an individual parameter in a dictionary. rc stands for the phrase ‘run command’ - essentially, configurations which will execute when you run your code.
'''
sns.set_style("ticks")
sns.set_context("poster")
sns.stripplot(x="day", y="total_bill", data=tips)
sns.plotting_context()
Returns:

{'axes.labelsize': 17.6,
 'axes.titlesize': 19.200000000000003,
 'font.size': 19.200000000000003,
 'grid.linewidth': 1.6,
 'legend.fontsize': 16.0,
 'lines.linewidth': 2.8000000000000003,
 'lines.markeredgewidth': 0.0,
 'lines.markersize': 11.200000000000001,
 'patch.linewidth': 0.48,
 'xtick.labelsize': 16.0,
 'xtick.major.pad': 11.200000000000001,
 'xtick.major.width': 1.6,
 'xtick.minor.width': 0.8,
 'ytick.labelsize': 16.0,
 'ytick.major.pad': 11.200000000000001,
 'ytick.major.width': 1.6,
 'ytick.minor.width': 0.8}'''
 
Conclusion
As you can see, Seaborn offers a lot of opportunities to customize your plots and have them show a distinct style. The color of your background, background style such as lines and ticks, and the size of your font all play a role in improving legibility and aesthetics.'''

'''
Seaborn Styling, Part 2: Color
Learn how to work with color in Seaborn and choose appropriate color palettes for your datasets.

Introduction
When creating a data visualization, your goal is to communicate the insights found in the data. While visualizing communicates important information, styling will influence how your audience understands what you’re trying to convey.

After you have formatted and visualized your data, the third and last step of data visualization is styling. Styling is the process of customizing the overall look of your visualization, or figure. Making intentional decisions about the details of the visualization will increase their impact and set your work apart.

In this article, we’ll look at how we can effectively use color to convey meaning. We’ll cover:

How to set a palette
Seaborn default and built-in color palettes
Color Brewer Palettes
Selecting palettes for your dataset
Commands for Working with Palettes
You can build color palettes using the function sns.color_palette(). This function can take any of the Seaborn built-in palettes (see below). You can also build your own palettes by passing in a list of colors in any valid Matplotlib format, including RGB tuples, hex color codes, or HTML color names.

If you want to quickly see what a palette looks like, use the function sns.palplot() to plot a palette as an array of colors:

# Save a palette to a variable:
palette = sns.color_palette("bright")

# Use palplot and pass in the variable:
sns.palplot(palette)
image1

To select and set a palette in Seaborn, use the command sns.set_palette() and pass in the name of the palette that you would like to use:

# Set the palette using the name of a palette:
sns.set_palette("Paired")

# Plot a chart:
sns.stripplot(x="day", y="total_bill", data=tips)
image2

Seaborn Default Color Palette
If you do not pass in a color palette to sns.color_palette() or sns.set_palette(), Seaborn will use a default set of colors. These defaults improve upon the Matplotlib default color palettes and are one significant reason why people choose to use Seaborn for their data visualizations. Here’s a comparison of the two default palettes:

image3

image4

Seaborn also allows you to style Matplotlib plots. So even if you’re using a plot that only exists in Matplotlib, such as a histogram, you can do so using Seaborn defaults.

To do so, call the sns.set() function before your plot:

# Call the sns.set() function 
sns.set()
for col in 'xy':
  plt.hist(data[col], normed=True, alpha=0.5)
image5

Not only does this function allow you the ability to use Seaborn default colors, but also any of Seaborn’s other styling techniques.

Seaborn has six variations of its default color palette: deep, muted, pastel, bright, dark, and colorblind.

image6

To use one of these palettes, pass the name into sns.set_palette():

# Set the palette to the "pastel" default palette:
sns.set_palette("pastel")

# plot using the "pastel" palette
sns.stripplot(x="day", y="total_bill", data=tips)
image7

Using Color Brewer Palettes
In addition to the default palette and its variations, Seaborn also allows the use of Color Brewer palettes. Color Brewer is the name of a set of color palettes inspired by the research of cartographer Cindy Brewer. The color palettes are specifically chosen to be easy to interpret when used to represent ordered categories. They are also colorblind accessible, as each color differs from its neighbors in lightness or tint.

To use, pass the name of any Color Brewer palette directly to any of the color functions:

custom_palette = sns.color_palette("Paired", 9)
sns.palplot(custom_palette)
image8

Here is a list of the the Color Brewer palettes, with their names for easy reference:

image9

Check out http://colorbrewer2.org (http://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3)for more information about color palette configuration options.

Selecting Color Palettes for Your Dataset
Qualitative Palettes for Categorical Datasets
When using a dataset that uses distinct but non-ordered categories, it’s good to use qualitative palettes. Qualitative palettes are sets of distinct colors which make it easy to distinguish the categories when plotted but don’t imply any particular ordering or meaning.

An example of categorical data is breed of dog. Each of these values, such as Border Collie or Poodle, are distinct from each other but there isn’t any inherent order to these categories.

Here’s an example of a qualitative Color Brewer palette:

qualitative_colors = sns.color_palette("Set3", 10)
sns.palplot(qualitative_colors)
image10

Sequential Palettes
Just as the name implies, sequential palettes are a set of colors that move sequentially from a lighter to a darker color. Sequential color palettes are appropriate when a variable exists as ordered categories, such as grade in school, or as continuous values that can be put into groups, such as yearly income. Because the darkest colors will attract the most visual attention, sequential palettes are most useful when only high values need to be emphasized.

Here’s an example of a sequential Color Brewer palette:

sequential_colors = sns.color_palette("RdPu", 10)
sns.palplot(sequential_colors)
image11

Diverging Palettes
Diverging palettes are best suited for datasets where both the low and high values might be of equal interest, such as hot and cold temperatures.

In the example below, both ends of the spectrum — fire red and deep blue — are likely to attract attention.

diverging_colors = sns.color_palette("RdBu", 10)
sns.palplot(diverging_colors)
image12

Here is a quick diagram that depicts each of the palette types:

image13

Credit: Michael Waskom
Summary
The ability to use easily choose different color palettes is one of the important affordances of styling your plots with Seaborn. Seaborn gives you a range of built-in plots to choose from: whether it’s variations on the defaults or access to all of the Color Brewer palettes. It’s easy to choose a palette that is well suited to your dataset, thanks to Color Brewer, as it supports palettes for qualitative, sequential, and diverging datasets.

For more on using color in Seaborn, check out their documentation.
'''

https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=color

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-  SNS PROJECT #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('all_data.csv')
'''
#general inquiry of df 
print(df.head())
df.info()
print(df.describe())
print(df.Year.value_counts)
print(df.Year.unique())
print(df.Country.unique())'''

#option to display all rows for printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#print(df)

#sory by country first , then by year.
df = df.sort_values(by=['Country', 'Year'], ascending= True)
#print(df)


df = df.rename(columns = {'Life expectancy at birth (years)' : 'LEABY'})

#plot GDP per Country

labels = df.Year.unique()

chile_gdp = df[df.Country == 'Chile'].GDP.values
china_gdp = df[df.Country == 'China'].GDP.values
germany_gdp = df[df.Country == 'Germany'].GDP.values
mexico_gdp = df[df.Country == 'Mexico'].GDP.values
mexico_gdp = df[df.Country == 'Mexico'].GDP.values
us_gdp = df[df.Country == 'United States of America'].GDP.values
zimbabwe_gdp = df[df.Country == 'Zimbabwe'].GDP.values

x = np.arange(len(labels)) # the label locations
width = 1/(7)  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 2.5*width, chile_gdp, width, label='Chile')
rects2 = ax.bar(x - 1.5*width, china_gdp, width, label='China')
rects3 = ax.bar(x - width/2, germany_gdp, width, label='Germany')
rects4 = ax.bar(x + width/2, mexico_gdp, width, label='Mexico')
rects5 = ax.bar(x + 1.5*width, us_gdp, width, label='United States of America')
rects6 = ax.bar(x + 2.5*width, zimbabwe_gdp, width, label='Zimbabwe')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('GDP')
ax.set_title('GDP per Year Group By Country')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#autolabel(rects4)
#autolabel(rects5)
#autolabel(rects6)

fig.tight_layout()

#plt.show()

#plot LEABY per Country

print(df.head())

labels = df.Year.unique()
countries = df.Country.unique()

x = np.arange(len(labels)) # the label locations
width = 1/(7)  # the width of the bars

fig1, ax1 = plt.subplots()

i = -2.5

for country in countries:
    country_leaby = df[df.Country == country].LEABY.values
    rec = ax1.bar(x - i * width, country_leaby, width, label=country)
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Life Expectation by year of Born')
ax1.set_title('Life Expectation by Countries')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_yticks(np.arange(40, 100, step=10))
ax1.set_ylim([40,100])
ax1.legend()

#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#autolabel(rects4)
#autolabel(rects5)
#autolabel(rects6)

fig.tight_layout()

#use seabron to plot

fig2, ax2 = plt.subplots()

sns.set_palette("Blues")
ax2 = sns.barplot(x="Year", y="GDP", hue="Country", data=df)

plt.xticks(rotation=90)

fig3, ax3 = plt.subplots()

# WORDBANK:
# "Year"
# "Country"
# "GDP"
# "LEABY"
# plt.scatter


# Uncomment the code below and fill in the blanks
ax3 = sns.FacetGrid(df, col='Year', hue='Country', col_wrap=4, size=2)
ax3 = (ax3.map(plt.scatter, "GDP", "LEABY", edgecolor="w").add_legend())

fig4, ax4 = plt.subplots()

# WORDBANK:
# plt.plot
# "LEABY"
# "Year"
# "Country"


# Uncomment the code below and fill in the blanks
g3 = sns.FacetGrid(df, col="Country", col_wrap=3, size=4)
g3 = (g3.map(plt.plot, "Year", "LEABY").add_legend())



plt.show()







  






#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-SEABORN CHEATSHEET SEABORN CHEATSHEET SEABORN CHEATSHEET  #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Seaborn
Seaborn is a Python data visualization library that builds off the functionalities of Matplotlib and integrates nicely with Pandas DataFrames. It provides a high-level interface to draw statistical graphs, and makes it easier to create complex visualizations.

Estimator argument in barplot
The estimator argument of the barplot() method in Seaborn can alter how the data is aggregated. By default, each bin of a barplot displays the mean value of a variable. Using the estimator argument this behaviour would be different.

The estimator argument can receive a function such as np.sum, len, np.median or other statistical function. This function can be used in combination with raw data such as a list of numbers and display in a barplot the desired statistic of this list.

Seaborn barplot
In Seaborn, drawing a barplot is simple using the function sns.barplot(). This function takes in the paramaters data, x, and y. It then plots a barplot using data as the dataframe, or dataset for the plot. x is the column of the dataframe that contains the labels for the x axis, and y is the column of the dataframe that contains the data to graph (aka what will end up on the y axis).

Using the Seaborn sample data “tips”, we can draw a barplot having the days of the week be the x axis labels, and the total_bill be the y axis values:

sns.barplot(data = tips, x = "day", y = "total_bill")

Barplot error bars
By default, Seaborn’s barplot() function places error bars on the bar plot. Seaborn uses a bootstrapped confidence interval to calculate these error bars.

The confidence interval can be changed to standard deviation by setting the parameter ci = "sd".

Seaborn hue
For the Seaborn function sns.barplot(), the hue parameter can be used to create a bar plot with more than one dimension, or, in other words, such that the data can be divided into more than one set of columns.

Using the Seaborn sample data “tips”, we can draw a barplot with the days of the week as the labels of the columns on the x axis, and the total_bill as the y axis values as follows:
'''
sns.barplot(data = tips, x = "day", y = "total_bill", hue = "sex")
'''
As you can see, hue divides the data into two columns based on the “sex” - male and female.

Seaborn function plots means by default
By default, the seaborn function sns.barplot() plots the means of each category on the x axis.

In the example code block, the barplot will show the mean satisfaction for every gender in the dataframe df.

sns.barplot(data = df, x = "Gender", y = "Satisfaction")
Box and Whisker Plots in Seaborn
A box and whisker plot shows a dataset’s median value, quartiles, and outliers. The box’s central line is the dataset’s median, the upper and lower lines marks the 1st and 3rd quartiles, and the “diamonds” shows the dataset’s outliers. With Seaborn, multiple data sets can be plotted as adjacent box and whisker plots for easier comparison.

Seaborn Package
Seaborn is a suitable package to plot variables and compare their distributions. With this package users can plot univariate and bivariate distributions among variables. It has superior capabilities than the popular methods of charts such as the barchart. Seaborn can show information about outliers, spread, lowest and highest points that otherwise would not be shown on a traditional barchart.

'''





		
		




























    

#-#-#-#-#-#-#-#-#-#-DATA CLEANING WITH PANDAS #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
DATA CLEANING WITH PANDAS
Introduction
A huge part of data science involves acquiring raw data and getting it into a form ready for analysis. Some have estimated that data scientists spend 80% of their time cleaning and manipulating data, and only 20% of their time actually analyzing it or building models from it.

When we receive raw data, we have to do a number of things before we’re ready to analyze it, possibly including:

diagnosing the “tidiness” of the data — how much data cleaning we will have to do
reshaping the data — getting right rows and columns for effective analysis
combining multiple files
changing the types of values — how we fix a column where numerical values are stored as strings, for example
dropping or filling missing values - how we deal with data that is incomplete or missing
manipulating strings to represent the data better
We will go through the techniques data scientists use to accomplish these goals by looking at some “unclean” datasets and trying to get them into a good, clean state.'''

'''
DATA CLEANING WITH PANDAS
Diagnose the Data
We often describe data that is easy to analyze and visualize as “tidy data”. What does it mean to have tidy data?

For data to be tidy, it must have:

Each variable as a separate column
Each row as a separate observation
For example, we would want to reshape a table like:

Account	Checkings	Savings
“12456543”	8500	8900
“12283942”	6410	8020
“12839485”	78000	92000

Into a table that looks more like:

Account	Account Type	Amount
“12456543”	“Checking”	8500
“12456543”	“Savings”	8900
“12283942”	“Checking”	6410
“12283942”	“Savings”	8020
“12839485”	“Checking”	78000
“12839485”	“Savings”	920000

The first step of diagnosing whether or not a dataset is tidy is using pandas functions to explore and probe the dataset.

You’ve seen most of the functions we often use to diagnose a dataset for cleaning. Some of the most useful ones are:

.head() — display the first 5 rows of the table
.info() — display a summary of the table
.describe() — display the summary statistics of the table
.columns — display the column names of the table
.value_counts() — display the distinct values for a column'''

import codecademylib3_seaborn
import pandas as pd

df1 = pd.read_csv("df1.csv")
df2 = pd.read_csv("df2.csv")

print(df1)

print(df1.head())
#print(df2.head())
print('\n')

print(df1.info())

print('\n')

print(df1.describe())

clean = 2

'''
DATA CLEANING WITH PANDAS
Dealing with Multiple Files
Often, you have the same data separated out into multiple files.

Let’s say that we have a ton of files following the filename structure: 'file1.csv', 'file2.csv', 'file3.csv', and so on. The power of pandas is mainly in being able to manipulate large amounts of structured data, so we want to be able to get all of the relevant information into one table so that we can analyze the aggregate data.

We can combine the use of glob, a Python library for working with files, with pandas to organize this data better. glob can open multiple files by using regex matching to get the filenames:
'''
import glob

files = glob.glob("file*.csv")

df_list = []
for filename in files:
  data = pd.read_csv(filename)
  df_list.append(data)

df = pd.concat(df_list)

print(files)'''
This code goes through any file that starts with 'file' and has an extension of .csv. It opens each file, reads the data into a DataFrame, and then concatenates all of those DataFrames together.'''

import codecademylib3_seaborn
import pandas as pd
import glob

student_files  =  glob.glob("exams*.csv")

df_list = []
for filename in student_files:
  data = pd.read_csv(filename)
  df_list.append(data)

students = pd.concat(df_list)

print(students)
print(len(students))

'''

DATA CLEANING WITH PANDAS
Reshaping your Data
Since we want

Each variable as a separate column
Each row as a separate observation
We would want to reshape a table like:

Account	Checking	Savings
“12456543”	8500	8900
“12283942”	6410	8020
“12839485”	78000	92000

Into a table that looks more like:
Account	Account Type	Amount
“12456543”	“Checking”	8500
“12456543”	“Savings”	8900
“12283942”	“Checking”	6410
“12283942”	“Savings”	8020
“12839485”	“Checking”	78000
“12839485”	“Savings”	920000

We can use pd.melt() to do this transformation. .melt() takes in a DataFrame, and the columns to unpack:
'''
pd.melt(frame=df, id_vars='name', value_vars=['Checking','Savings'], value_name="Amount", var_name="Account Type")
'''
The parameters you provide are:

frame: the DataFrame you want to melt
id_vars: the column(s) of the old DataFrame to preserve
value_vars: the column(s) of the old DataFrame that you want to turn into variables
value_name: what to call the column of the new DataFrame that stores the values
var_name: what to call the column of the new DataFrame that stores the variables
The default names may work in certain situations, but it’s best to always have data that is self-explanatory. Thus, we often use .columns() to rename the columns after melting:
'''
df.columns(["Account", "Account Type", "Amount"])
'''
'''


import codecademylib3_seaborn
import pandas as pd
from students import students

print(students.columns)
students = pd.melt(frame=students, id_vars=['full_name','gender_age','grade'], value_vars=['fractions', 'probability'], value_name='score', var_name='exam')

print(students.head())
print(students.columns)
print(students.exam.value_counts())


'''
DATA CLEANING WITH PANDAS
Dealing with Duplicates
Often we see duplicated rows of data in the DataFrames we are working with. This could happen due to errors in data collection or in saving and loading the data.

To check for duplicates, we can use the pandas function .duplicated(), which will return a Series telling us which rows are duplicate rows.

Let’s say we have a DataFrame fruits that represents this table:

item	price	calories
“banana”	“$1”	105
“apple”	“$0.75”	95
“apple”	“$0.75”	95
“peach”	“$3”	55
“peach”	“$4”	55
“clementine”	“$2.5”	35

If we call '''fruits.duplicated()''', we would get the following table:

id	value
0	False
1	False
2	True
3	False
4	False
5	False

We can see that row 2, which represents an "apple" with price "$0.75" and 95 calories, is a duplicate row. Every value in this row is the same as in another row.

We can use the pandas .drop_duplicates() function to remove all rows that are duplicates of another row.

If we call '''fruits.drop_duplicates()''', we would get the table:

item	price	calories
“banana”	“$1”	105
“apple”	“$0.75”	95
“peach”	“$3”	55
“peach”	“$4”	55
“clementine”	“$2.5”	35

The "apple" row was deleted because it was exactly the same as another row. But the two "peach" rows remain because there is a difference in the price column.

If we wanted to remove every row with a duplicate value in the item column, we could specify a subset:
'''
fruits = fruits.drop_duplicates(subset=['item'])'''
By default, this keeps the first occurrence of the duplicate:

item	price	calories
“banana”	“$1”	105
“apple”	“$0.75”	95
“peach”	“$3”	55
“clementine”	“$2.5”	35

Make sure that the columns you drop duplicates from are specifically the ones where duplicates don’t belong. You wouldn’t want to drop duplicates with the price column as a subset, for example, because it’s okay if multiple items cost the same amount!'''


import codecademylib3_seaborn
import pandas as pd
from students import students

#print(students)

duplicates = students.duplicated()
print(duplicates)
print(duplicates.value_counts)
print(duplicates.unique())

students = students.drop_duplicates()
print(students)

duplicates = students.duplicated()

'''
DATA CLEANING WITH PANDAS
Splitting by Index
In trying to get clean data, we want to make sure each column represents one type of measurement. Often, multiple measurements are recorded in the same column, and we want to separate these out so that we can do individual analysis on each variable.

Let’s say we have a column “birthday” with data formatted in MMDDYYYY format. In other words, “11011993” represents a birthday of November 1, 1993. We want to split this data into day, month, and year so that we can use these columns as separate features.

In this case, we know the exact structure of these strings. The first two characters will always correspond to the month, the second two to the day, and the rest of the string will always correspond to year. We can easily break the data into three separate columns by splitting the strings using .str:'''

# Create the 'month' column
df['month'] = df.birthday.str[0:2]

# Create the 'day' column
df['day'] = df.birthday.str[2:4]

# Create the 'year' column
df['year'] = df.birthday.str[4:]'''
The first command takes the first two characters of each value in the birthday column and puts it into a month column. The second command takes the second two characters of each value in the birthday column and puts it into a day column. The third command takes the rest of each value in the birthday column and puts it into a year column.

This would transform a table like:

id	birthday
1011	“12241989”
1112	“10311966”
1113	“01052011”

into a table like:

id	birthday	month	day	year
1011	“12241989”	“12”	“24”	“1989”
1112	“10311966”	“10”	“31”	“1966”
1113	“01052011”	“01”	“05”	“2011”

We will practice changing string columns into numerical columns (like converting "10" to 10) in a future exercise.'''

'''
Instructions
1.
Print out the columns of the students DataFrame.

2.
The column gender_age sounds like it contains both gender and age!

Print out the .head() of the column to see what kind of data it contains.


Hint
To print out the head of a column, we can use the pandas syntax:

print(df.column_name.head())
or:

print(df['column_name'].head())
3.
It looks like the first character of the values in gender_age contains the gender, while the rest of the string contains the age. Let’s separate out the gender data into a new column called gender.


Hint
To create a new column in pandas, you can use the syntax:

df.new_column_name = new_values
or:

df['new_column_name'] = new_values
4.
Now, separate out the age data into a new column called age.


Hint
The age is the rest of the gender_age string!

students.gender_age.str[1:]
This line of code takes everything after the first character of each string in gender_age.

5.
Good job! Let’s print the .head() of students to see how the DataFrame looks after our creation of new columns.

6.
Now, we don’t need that gender_age column anymore.

Let’s set the students DataFrame to be the students DataFrame with all columns except gender_age.


Hint
To select a subset of columns in pandas, you can use the notation:
'''
df[['column1', 'column2', 'column3']]'''
This will return a DataFrame containing just column1, column2 and column3 of the original DataFrame.'''


import codecademylib3_seaborn
import pandas as pd
from students import students

print(students.columns)

print(students.head())

students['gender'] = students.gender_age.str[0]
students['age'] = students.gender_age.str[1:]

print(students.head())
print(students.columns)

students = students [['full_name',  'grade', 'exam', 'score', 'gender', 'age']]

'''
DATA CLEANING WITH PANDAS
Splitting by Character
Let’s say we have a column called “type” with data entries in the format "admin_US" or "user_Kenya". Just like we saw before, this column actually contains two types of data. One seems to be the user type (with values like “admin” or “user”) and one seems to be the country this user is in (with values like “US” or “Kenya”).

We can no longer just split along the first 4 characters because admin and user are of different lengths. Instead, we know that we want to split along the "_". Using that, we can split this column into two separate, cleaner columns:
'''

# Create the 'str_split' column
df['str_split'] = df.type.str.split('_')

# Create the 'usertype' column
df['usertype'] = df.str_split.str.get(0)

# Create the 'country' column
df['country'] = df.str_split.str.get(1)'''
This would transform a table like:

id	type
1011	“user_Kenya”
1112	“admin_US”
1113	“moderator_UK”

into a table like:

id	type	country	usertype
1011	“user_Kenya”	“Kenya”	“user”
1112	“admin_US”	“US”	“admin”
1113	“moderator_UK”	“UK”	“moderator”'''

'''
Instructions
1.
The students’ names are stored in a column called full_name.

We want to separate this data out into two new columns, first_name and last_name.

First, let’s create a Series object called name_split that splits the full_name by the " " character.

2.
Now, let’s create a column called first_name that takes the first item in name_split.

3.
Finally, let’s create a column called last_name that takes the second item in name_split.

4.
Print out the .head() of students to see how the DataFrame has changed.'''

import codecademylib3_seaborn
import pandas as pd
from students import students

print(students)

name_split = students.full_name.str.split()

students['first_name'] = name_split.str[0]
students['last_name'] = name_split.str[1]

print(students.head())

'''
DATA CLEANING WITH PANDAS
Looking at Types
Each column of a DataFrame can hold items of the same data type or dtype. The dtypes that pandas uses are: float, int, bool, datetime, timedelta, category and object. Often, we want to convert between types so that we can do better analysis. If a numerical category like "num_users" is stored as a Series of objects instead of ints, for example, it makes it more difficult to do something like make a line graph of users over time.

To see the types of each column of a DataFrame, we can use:

print(df.dtypes)
For a DataFrame like this:

item	price	calories
“banana”	“$1”	105
“apple”	“$0.75”	95
“peach”	“$3”	55
“clementine”	“$2.5”	35

the .dtypes attribute would be:

item        object
price       object
calories     int64
dtype: object
We can see that the dtype of the dtypes attribute itself is an object! It is a Series object, which you have already worked with. Series objects compose all DataFrames.

We can see that the price column is made up of objects, which will probably make our analysis of price more difficult. We’ll look at how to convert columns into numeric values in the next few exercises.'''

'''DATA CLEANING WITH PANDAS
String Parsing
Sometimes we need to modify strings in our DataFrames to help us transform them into more meaningful metrics. For example, in our fruits table from before:

item	price	calories
“banana”	“$1”	105
“apple”	“$0.75”	95
“peach”	“$3”	55
“peach”	“$4”	55
“clementine”	“$2.5”	35

We can see that the 'price' column is actually composed of strings representing dollar amounts. This column could be much better represented in floats, so that we could take the mean, calculate other aggregate statistics, or compare different fruits to one another in terms of price.

First, we can use what we know of regex to get rid of all of the dollar signs:

fruit.price = fruit['price'].replace('[\$,]', '', regex=True)
Then, we can use the pandas function .to_numeric() to convert strings containing numerical values to integers or floats:

fruit.price = pd.to_numeric(fruit.price)
Now, we have a DataFrame that looks like:

item	price	calories
“banana”	1	105
“apple”	0.75	95
“peach”	3	55
“peach”	4	55
“clementine”	2.5	35

'''

import codecademylib3_seaborn
import pandas as pd
from students import students

print(students.head())

students['score'] = students['score'].replace('[\%]', '', regex = True)

students['score'] = pd.to_numeric(students['score'])

'''
DATA CLEANING WITH PANDAS
More String Parsing
Sometimes we want to do analysis on numbers that are hidden within string values. We can use regex to extract this numerical data from the strings they are trapped in. Suppose we had this DataFrame df representing a workout regimen:

date	exerciseDescription
10/18/2018	“lunges - 30 reps”
10/18/2018	“squats - 20 reps”
10/18/2018	“deadlifts - 25 reps”
10/18/2018	“jumping jacks - 30 reps”
10/19/2018	“lunges - 40 reps”
10/19/2018	“chest flyes - 15 reps”
…	…

It would be helpful to separate out data like "30 lunges" into 2 columns with the number of reps, "30", and the type of exercise, "lunges". Then, we could compare the increase in the number of lunges done over time, for example.

To extract the numbers from the string we can use pandas’ .str.split() function:
'''
split_df = df['exerciseDescription'].str.split('(\d+)', expand=True)
'''
which would result in this DataFrame split_df:

* *	0	1	2
0	“lunges - “	“30”	“reps”
1	“squats - “	“20”	“reps”
2	“deadlifts - “	“25”	“reps”
3	“jumping jacks - “	“30”	“reps”
4	“lunges - “	“40”	“reps”
5	“chest flyes - “	“15”	“reps”
…	…	…	…

Then, we can assign columns from this DataFrame to the original df:
'''
df.reps = pd.to_numeric(split_df[1])
df.exercise = split_df[2].replace('[\- ]', '', regex=True)'''
Now, our df looks like this:

date	exerciseDescription	reps	exercise
10/18/2018	“lunges - 30 reps”	30	“lunges”
10/18/2018	“squats - 20 reps”	20	“squats”
10/18/2018	“deadlifts - 25 reps”	25	“deadlifts”
10/18/2018	“jumping jacks - 30 reps”	30	“jumping jacks”
10/19/2018	“lunges - 40 reps”	40	“lunges”
10/19/2018	“chest flyes - 15 reps”	15	“chest flyes”
…	…	…	…
'''

import codecademylib3_seaborn
import pandas as pd
from students import students

#print(students)
print(students.grade.head())

grade_split = students['grade'].str.split('(\d+)', expand = True)
students['grade'] = grade_split[1]

print(students.dtypes)

students.grade = pd.to_numeric(students.grade)
avg_grade =students.grade.mean()

'''
DATA CLEANING WITH PANDAS
Missing Values
We often have data with missing elements, as a result of a problem with the data collection process or errors in the way the data was stored. The missing elements normally show up as NaN (or Not a Number) values:

day	bill	tip	num_guests
“Mon”	10.1	1	1
“Mon”	20.75	5.5	2
“Tue”	19.95	5.5	NaN
“Wed”	44.10	15	3
“Wed”	NaN	1	1

The num_guests value for the 3rd row is missing, and the bill value for the 5th row is missing. Some calculations we do will just skip the NaN values, but some calculations or visualizations we try to perform will break when a NaN is encountered.

Most of the time, we use one of two methods to deal with missing values.

Method 1: drop all of the rows with a missing value
We can use .dropna() to do this:
'''
bill_df = bill_df.dropna()'''
This command will result in the DataFrame without the incomplete rows:

day	bill	tip	num_guests
“Mon”	10.1	1	1
“Mon”	20.75	5.5	2
“Wed”	44.10	15	3

If we wanted to remove every row with a NaN value in the num_guests column only, we could specify a subset:
'''
bill_df = bill_df.dropna(subset=['num_guests'])'''
Method 2: fill the missing values with the mean of the column, or with some other aggregate value.
We can use .fillna() to do this:
'''
bill_df = bill_df.fillna(value={"bill":bill_df.bill.mean(), "num_guests":bill_df.num_guests.mean()})'''
This command will result in the DataFrame with the respective mean of the column in the place of the original NaNs:

bill_df = bill_df.fillna(100)
this command will fill all NaN to 100

day	bill	tip	num_guests
“Mon”	10.1	1	1
“Mon”	20.75	5.5	2
“Tue”	19.95	5.5	1.75
“Wed”	44.10	15	3
“Wed”	23.725	1	1
'''
import codecademylib3_seaborn
import pandas as pd
from students import students

print(students)

score_mean = students.score.mean()

students = students.fillna(value = {'score':0})

score_mean_2 = students.score.mean()

print(score_mean, score_mean_2)


#-------------- project ------------------ 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecademylib3_seaborn
import glob

files = glob.glob('states*.csv')

df_list = []
for filename in files:
  data = pd.read_csv(filename)
  df_list.append(data)

df = pd.concat(df_list)

#print(df.head(20))
#print(df.info())
#print(df.describe())
#print(df.columns)

df['Income'] = df['Income'].replace('[\$]', '', regex = True)
df['Income'] = pd.to_numeric(df['Income'])



split_df = df['GenderPop'].str.split('_', expand = True)
split_df[0] = split_df[0].replace('[M]', '', regex = True)
split_df[1] = split_df[1].replace('[F]', '', regex = True)
split_df[0] = pd.to_numeric(split_df[0])
split_df[1] = pd.to_numeric(split_df[1])

df['Male'] = split_df[0]
df['Female'] = split_df[1]



df = df.reset_index()

df = df.drop(columns = [ 'Unnamed: 0', 'GenderPop'])


values = {'Female': (df.TotalPop - df.Male)}
df = df.fillna(value=values)


print(df['State'].duplicated())
df = df.drop_duplicates(['State'])


print(df)

plt.scatter(df.TotalPop,df.Income)
plt.show()
#print (df.Female, df.Income)


