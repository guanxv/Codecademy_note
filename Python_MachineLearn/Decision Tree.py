


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

#-#-#-#-#-#-#-#-#-#-#-#-#- DECISION TREE PROEJCT #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

'''

MACHINE LEARNING: SUPERVISED LEARNING 🤖
Find the Flag
Can you guess which continent this flag comes from?

Flag of Reunion
What are some of the features that would clue you in? Maybe some of the colors are good indicators. The presence or absence of certain shapes could give you a hint. In this project, we’ll use decision trees to try to predict the continent of flags based on several of these features.

We’ll explore which features are the best to use and the best way to create your decision tree.

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
12/14Complete
Mark the tasks as complete by checking them off
Investigate the Data
1.
Let’s start by seeing what the data looks like. Begin by loading the data into a variable named flags using Panda’s pd.read_csv() function. The function should take the name of the CSV file you want to load. In this case, our file is named "flags.csv".

We also want row 0 to be used as the header, so include the parameter header = 0.


Stuck? Get a hint
2.
Take a look at the names of the columns in our DataFrame. These are the features we have available to us. Print flags.columns.

Let’s also take a look at the first few rows of the dataset. Print flags.head().

3.
Many columns contain numbers that don’t make a lot of sense. For example, the third row, which represents Algeria, has a Language of 8. What exactly does that mean?

Take a look at the Attribute Information for this dataset from UCI’s Machine Learning Repository.

Using that information along with the printout of flags.head(), can you figure out what landmass Andora is on?

Creating Your Data and Labels
4.
We’re eventually going to use create a decision tree to classify what Landmass a country is on.

Create a variable named labels and set it equal to only the "Landmass" column from flags.

You can grab specific columns from a DataFrame using this syntax:

one_column = df[["A"]]
two_columns = df[["B", "C"]]
In this example, one_column will be a DataFrame of only df‘s "A" column. two_columns will be a DataFrame of the "B" and "C" columns from df.


Stuck? Get a hint
5.
We have our labels. Now we want to choose which columns will help our decision tree correctly classify those labels.

You could spend a lot of time playing with groups of columns to find the that work best. But for now, let’s see if we can predict where a country is based only on the colors of its flag.

Create a variable named data and set it equal to a DataFrame containing the following columns from flags:

"Red"
"Green"
"Blue"
"Gold"
"White"
"Black"
"Orange"

Stuck? Get a hint
6.
Finally, let’s split these DataFrames into a training set and test set using the train_test_split() function. This function should take data and labels as parameters. Also include the parameter random_state = 1.

This function returns four values. Name those values train_data, test_data, train_labels, and test_labels in that order.


Stuck? Get a hint
Make and Test the Model
7.
Create a DecisionTreeClassifier and name it tree. When you create the tree, give it the parameter random_state = 1.


Stuck? Get a hint
8.
Call tree‘s .fit() method using train_data and train_labels to fit the tree to the training data.


Stuck? Get a hint
9.
Call .score() using test_data and test_labels. Print the result.

Since there are six possible landmasses, if we randomly guessed, we’d expect to be right about 16% of the time. Did our decision tree beat randomly guessing?

Tuning the Model
10.
We now have a good baseline of how our model performs with these features. Let’s see if we can prune the tree to make it better!

Put your code that creates, trains, and tests the tree inside a for loop that has a variable named i that increases from 1 to 20.

Inside your for loop, when you create tree, give it the parameter max_depth = i.

We’ll now see a printout of how the accuracy changes depending on how large we allow the tree to be.


Stuck? Get a hint
11.
Rather than printing the score of each tree, let’s graph it! We want the x-axis to show the depth of the tree and the y-axis to show the tree’s score.

To do this, we’ll need to create a list containing all of the scores. Before the for loop, create an empty list named scores. Inside the loop, instead of printing the tree’s score, use .append() to add it to scores.


Stuck? Get a hint
12.
Let’s now plot our points. Call plt.plot() using two parameters. The first should be the points on the x-axis. In this case, that is range(1, 21). The second should be scores.

Then call plt.show().


Stuck? Get a hint
13.
Our graph doesn’t really look like we would expect it to. It seems like the depth of the tree isn’t really having an impact on its performance. This might be a good indication that we’re not using enough features.

Let’s add all the features that have to do with shapes to our data. data should now be set equal to:

flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]
What does your graph look like after making this change?

Explore on Your Own
14.
Nice work! That graph looks more like what we’d expect. If the tree is too short, we’re underfitting and not accurately representing the training data. If the tree is too big, we’re getting too specific and relying too heavily on the training data.

There are a few different ways to extend this project:

Try to classify something else! Rather than predicting the "Landmass" feature, could predict something like the "Language"?
Find a subset of features that work better than what we’re currently using. An important note is that a feature that has categorical data won’t work very well as a feature. For example, we don’t want a decision node to split nodes based on whether the value for "Language" is above or below 5.
Tune more parameters of the model. You can find a description of all the parameters you can tune in the Decision Tree Classifier documentation. For example, see what happens if you tune max_leaf_nodes. Think about whether you would be overfitting or underfitting the data based on how many leaf nodes you allow.
'''


import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


flags = pd.read_csv('flags.csv', header = 0)

#print(flags)
#print(flags.columns)

labels = flags[['Landmass']]

#data = flags[['Red', 'Green', 'Blue', 'Gold', 'White','Black', 'Orange']]
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

train_data, test_data, train_label, test_label = train_test_split(data, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

scores = []


for i in range(1, 21):

  tree = DecisionTreeClassifier(max_depth = i)
  tree.fit(train_data, train_label)
  print('Max Depth = ', i, end = '')
  print("  The Score is ", tree.score(test_data, test_label))
  scores.append(tree.score(test_data, test_label))

plt.plot(range(1,21), scores)
plt.show()

print(data.columns)

flag_to_check = pd.DataFrame ([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], columns =  ['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange','Circles','Crosses', 'Saltires', 'Quarters', 'Sunstars', 'Crescent', 'Triangle'])
#flag_to_check = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

print(tree.predict(flag_to_check))




#-#-#-#-#-#-#-#-#-#-#-#-#- END OF DECISION TREE PROEJCT #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-



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


