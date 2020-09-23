


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-PYTHON SORTING -- PYTHON SORTING -- PYTHON SORTING -- PYTHON SORTING -- PYTHON SORTING#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

'''BUBBLE SORT: CONCEPTUAL
Bubble Sort Introduction
Bubble sort is an introductory sorting algorithm that iterates through a list and compares pairings of adjacent elements.

According to the sorting criteria, the algorithm swaps elements to shift elements towards the beginning or end of the list.

By default, a list is sorted if for any element e and position 1 through N:

e1 <= e2 <= e3 … eN, where N is the number of elements in the list.

For example, bubble sort transforms a list:

[5, 2, 9, 1, 5]
to an ascending order, from lowest to highest:

[1, 2, 5, 5, 9]
We implement the algorithm with two loops.

The first loop iterates as long as the list is unsorted and we assume it’s unsorted to start.

Within this loop, another iteration moves through the list. For each pairing, the algorithm asks:

In comparison, is the first element larger than the second element?

If it is, we swap the position of the elements. The larger element is now at a greater index than the smaller element.

When a swap is made, we know the list is still unsorted. The outer loop will run again when the inner loop concludes.

The process repeats until the largest element makes its way to the last index of the list. The outer loop runs until no swaps are made within the inner loop.'''

'''BUBBLE SORT: CONCEPTUAL
Bubble Sort
As mentioned, the bubble sort algorithm swaps elements if the element on the left is larger than the one on the right.

How does this algorithm swap these elements in practice?

Let’s say we have the two values stored at the following indices index_1 and index_2. How would we swap these two elements within the list?

It is tempting to write code like:

list[index_1] = list[index_2]
list[index_2] = list[index_1]
However, if we do this, we lose the original value at index_1. The element gets replaced by the value at index_2. Both indices end up with the value at index_2.

Programming languages have different ways of avoiding this issue. In some languages, we create a temporary variable which holds one element during the swap:

temp = list[index_1]
list[index_1] = list[index_2]
list[index_2] = temp 
The GIF illustrates this code snippet.

Other languages provide multiple assignment which removes the need for a temporary variable.

list[index_1], list[index_2] = list[index_2], list[index_1]'''

'''BUBBLE SORT: CONCEPTUAL
Algorithm Analysis
Given a moderately unsorted data-set, bubble sort requires multiple passes through the input before producing a sorted list. Each pass through the list will place the next largest value in its proper place.

We are performing n-1 comparisons for our inner loop. Then, we must go through the list n times in order to ensure that each item in our list has been placed in its proper order.

The n signifies the number of elements in the list. In a worst case scenario, the inner loop does n-1 comparisons for each n element in the list.

Therefore we calculate the algorithm’s efficiency as:

\mathcal{O}(n(n-1)) = \mathcal{O}(n(n)) = \mathcal{O}(n^2)O(n(n−1))=O(n(n))=O(n 
2
 )
The diagram analyzes the pseudocode implementation of bubble sort to show how we draw this conclusion.

When calculating the run-time efficiency of an algorithm, we drop the constant (-1), which simplifies our inner loop comparisons to n.

This is how we arrive at the algorithm’s runtime: O(n^2).'''

'''
BUBBLE SORT: CONCEPTUAL
Bubble Sort Review
Bubble sort is an algorithm to sort a list through repeated swaps of adjacent elements. It has a runtime of O(n^2).

For nearly sorted lists, bubble sort performs relatively few operations since it only performs a swap when elements are out of order.

Bubble sort is a good introductory algorithm which opens the door to learning more complex algorithms. It answers the question, “How can we algorithmically sort a list?” and encourages us to ask, “How can we improve this sorting algorithm?”'''


'''BuBBLE SORT: PYTHON
Bubble Sort: Swap
The Bubble Sort algorithm works by comparing a pair of neighbor elements and shifting the larger of the two to the right. Bubble Sort completes this by swapping the two elements’ positions if the first element being compared is larger than the second element being compared.

Below is a quick pseudocode example of what we will create:

for each pair(elem1, elem2):
  if elem1 > elem2:
    swap(elem1, elem2)
  else:
    # analyze next set of pairs
This swap() sub-routine is an essential part of the algorithm. Bubble sort swaps elements repeatedly until the largest element in the list is placed at the greatest index. This looping continues until the list is sorted.

This GIF illustrates how swap() method works.'''

nums = [5, 2, 9, 1, 5, 6]

# Define swap() below:

def swap(arr, index_1, index_2):
  temp = arr[index_1]
  arr[index_1] = arr[index_2]
  arr[index_2] = temp

swap(nums, 3, 5)
print(nums)

'''BUBBLE SORT: PYTHON
Bubble Sort: Compare
Now that we know how to swap items in an array, we need to set up the loops which check whether a swap is necessary.

Recall that Bubble Sort compares neighboring items and if they are out of order, they are swapped.

What does it mean to be “out of order”? Since bubble sort is a comparison sort, we’ll use a comparison operator: <.

We’ll have two loops:

One loop will iterate through each element in the list.

Within the first loop, we’ll have another loop for each element in the list.

Inside the second loop, we’ll take the index of the loop and compare the element at that index with the element at the next index. If they’re out of order, we’ll make a swap!'''

'''Instructions
1.
Below the body of swap(), define a new function: bubble_sort() which has the parameter arr.

Write pass in the body of bubble_sort to start.

2.
Inside bubble_sort(), replace pass with a for loop that iterates up until the last element of the list.

Inside the for loop, check if the value in arr at index is > the value in arr at index + 1.

If it is, use swap() and pass arr, index, and index + 1 as arguments.

We can loop through every element except the last using:

pets = ['donkey', 'rabbit', 'snake', 'cat']

for index in range(len(pets) - 1):
  print(pets[index])

# donkey
# rabbit
# snake
We can make a comparison check like so:

nums = [4, 2, 7]

for index in range(len(nums) - 1):
  if nums[index] > nums[index + 1]:
    print("Elements are out of order!")
  else:
    print("Elements are in order!")

# Elements are out of order!
# Elements are in order!
3.
As you can see by the output, our list is not sorted!

One loop through the list is only sufficient to move the largest value to its correct placement.

Create another loop which iterates for each element in arr.

Move the entire contents of the function within this loop:

def bubble_sort(arr):
  for el in arr:
    # previous code goes here!
Run the code again, your list should be sorted!

Your code should now look like the following

def bubble_sort(arr):
  for el in arr:
    for index in range(len(arr)-1):
      if arr[index] > arr[index + 1]:
        swap(arr, index, index + 1)
'''

nums = [5, 2, 9, 1, 5, 6]

def swap(arr, index_1, index_2):
  temp = arr[index_1]
  arr[index_1] = arr[index_2]
  arr[index_2] = temp
  
# define bubble_sort():
def bubble_sort(arr):
  for el in arr:
    for i in range(len(arr)-1):
      if arr[i] > arr[i+1]:
        swap(arr, i, i+1)


##### test statements

print("Pre-Sort: {0}".format(nums))      
bubble_sort(nums)
print("Post-Sort: {0}".format(nums))

'''BUBBLE SORT: PYTHON
Bubble Sort: Optimized
As you were writing Bubble Sort, you may have realized that we were doing some unnecessary iterations.

Consider the first pass through the outer loop. We’re making n-1 comparisons.

nums = [5, 4, 3, 2, 1]
# 5 element list: N is 5
bubble_sort(nums)
# 5 > 4
# [4, 5, 3, 2, 1]
# 5 > 3
# [4, 3, 5, 2, 1]
# 5 > 2
# [4, 3, 2, 5, 1]
# 5 > 1
# [4, 3, 2, 1, 5]
# four comparisons total
We know the last value in the list is in its correct position, so we never need to consider it again. The second time through the loop, we only need n-2 comparisons.

As we correctly place more values, fewer elements need to be compared. An optimized version doesn’t make n^2-n comparisons, it does (n-1) + (n-2) + ... + 2 + 1 comparisons, which can be simplified to (n^2-n) / 2 comparisons.

This is fewer than n^2-n comparisons but the algorithm still has a big O runtime of O(N^2).

As the input, N, grows larger, the division by two has less significance. Big O considers inputs as they reach infinity so the higher order term N^2 completely dominates.

We can’t make Bubble Sort better than O(N^2), but let’s take a look at the optimized code and compare iterations between implementations!

We’re also taking advantage of parallel assignment in Python and abstracting away the swap() function!'''

nums = [9, 8, 7, 6, 5, 4, 3, 2, 1]
print("PRE SORT: {0}".format(nums))

def swap(arr, index_1, index_2):
  temp = arr[index_1]
  arr[index_1] = arr[index_2]
  arr[index_2] = temp

def bubble_sort_unoptimized(arr):
  iteration_count = 0
  for el in arr:
    for index in range(len(arr) - 1):
      iteration_count += 1
      if arr[index] > arr[index + 1]:
        swap(arr, index, index + 1)

  print("PRE-OPTIMIZED ITERATION COUNT: {0}".format(iteration_count))

def bubble_sort(arr):
  iteration_count = 0
  for i in range(len(arr)):
    # iterate through unplaced elements
    for idx in range(len(arr) - i - 1):
      iteration_count += 1
      if arr[idx] > arr[idx + 1]:
        # replacement for swap function
        arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
        
  print("POST-OPTIMIZED ITERATION COUNT: {0}".format(iteration_count))

bubble_sort_unoptimized(nums.copy())
bubble_sort(nums)
print("POST SORT: {0}".format(nums))


'''MERGE SORT: CONCEPTUAL
What Is A Merge Sort?
Merge sort is a sorting algorithm created by John von Neumann in 1945. Merge sort’s “killer app” was the strategy that breaks the list-to-be-sorted into smaller parts, sometimes called a divide-and-conquer algorithm.

In a divide-and-conquer algorithm, the data is continually broken down into smaller elements until sorting them becomes really simple.

Merge sort was the first of many sorts that use this strategy, and is still in use today in many different applications.

How To Merge Sort:
Merge sorting takes two steps: splitting the data into “runs” or smaller components, and the re-combining those runs into sorted lists (the “merge”).

When splitting the data, we divide the input to our sort in half. We then recursively call the sort on each of those halves, which cuts the halves into quarters. This process continues until all of the lists contain only a single element. Then we begin merging.

When merging two single-element lists, we check if the first element is smaller or larger than the other. Then we return the two-element list with the smaller element followed by the larger element.

MERGE SORT: CONCEPTUAL
Merging
When merging larger pre-sorted lists, we build the list similarly to how we did with single-element lists.

Let’s call the two lists left and right. Bothleft and right are already sorted. We want to combine them (to merge them) into a larger sorted list, let’s call it both. To accomplish this we’ll need to iterate through both with two indices, left_index and right_index.

At first left_index and right_index both point to the start of their respective lists. left_index points to the smallest element of left (its first element) and right_index points to the smallest element of right.

Compare the elements at left_index and right_index. The smaller of these two elements should be the first element of both because it’s the smallest of both! It’s the smallest of the two smallest values.

Let’s say that smallest value was in left. We continue by incrementing left_index to point to the next-smallest value in left. Then we compare the 2nd smallest value in left against the smallest value of right. Whichever is smaller of these two is now the 2nd smallest value of both.

This process of “look at the two next-smallest elements of each list and add the smaller one to our resulting list” continues on for as long as both lists have elements to compare. Once one list is exhausted, say every element from left has been added to the result, then we know that all the elements of the other list, right, should go at the end of the resulting list (they’re larger than every element we’ve added so far).

Merge Sort Performance
Merge sort was unique for its time in that the best, worst, and average time complexity are all the same: Θ(N*log(N)). This means an almost-sorted list will take the same amount of time as a completely out-of-order list. This is acceptable because the worst-case scenario, where a sort could stand to take the most time, is as fast as a sorting algorithm can be.

Some sorts attempt to improve upon the merge sort by first inspecting the input and looking for “runs” that are already pre-sorted. Timsort is one such algorithm that attempts to use pre-sorted data in a list to the sorting algorithm’s advantage. If the data is already sorted, Timsort runs in Θ(N) time.

Merge sort also requires space. Each separation requires a temporary array, and so a merge sort would require enough space to save the whole of the input a second time. This means the worst-case space complexity of merge sort is O(N).'''

###################################################################################################

'''MERGE SORT: PYTHON
Separation
What is sorted by a sort? A sort takes in a list of some data. The data can be words that we want to sort in dictionary order, or people we want to sort by birth date, or really anything else that has an order. For the simplicity of this lesson, we’re going to imagine the data as just numbers.

The first step in a merge sort is to separate the data into smaller lists. Then we break those lists into even smaller lists. Then, when those lists are all single-element lists, something amazing happens! Well, kind of amazing. Well, you might have expected it, we do call it a “merge sort”. We merge the lists.'''

'''Instructions
1.
Define a function called merge_sort(). Give merge_sort() one parameter: items.

2.
We’re going to use merge_sort() to break down items into smaller and smaller lists, and then write a merge() function that will combine them back together.

For now, check the length of items. If items has length one or less, return items.'''

def merge_sort(items):
  if len(items) <=1:
    return items

'''MERGE SORT: PYTHON
Partitions
How do we break up the data in a merge sort? We split it in half until there’s no more data to split. Our first step is to break down all of the items of the list into their own list.

Instructions
1.
After returning all inputs that have less than 2 elements, we split everything up that’s longer.

Create the variable middle_index which is the index to the middle element in the list.

You can find the middle_index of a list by calculating its length (using the len() function) and then dividing it by 2 using integer division.

Altogether this should look like this:

middle_index = len(items) // 2
2.
Create another variable called left_split. This should be a list of all elements in the input list starting at the first up to but not including the middle_index element.

You can use the list slice operator:

a_list = ['peas', 'carrots', 'potatoes']
slice_til_here = 2
a_list[:slice_til_here]
# ['peas', 'carrots']
3.
Create one more variable called right_split which includes all elements in items from the middle_index to the end of the list.

Using a starting index to slice a list includes the element at the starting index, but an ending index does not include that element. This makes halving a list (like we’re doing) easier!

cool_list = [1, 2, 3, 4, 5, 6]
start = cool_list[:3]
end = cool_list[3:]

print(start)
# [1, 2, 3]
print(end)
# [4, 5, 6]
4.
For now, return all three of these at the bottom of the function in a single return statement. Like this:

return middle_index, left_split, right_split'''

def merge_sort(items):
  if len(items) < 2:
    return items
  
  middle_index = len(items)//2
  left_split = items[:middle_index]
  right_split = items[middle_index:]

  return middle_index, left_split, right_split

'''MERGE SORT: PYTHON
Creating the Merge Function
Our merge_sort() function so far only separates the input list into many different parts — pretty much the opposite of what you’d expect a merge sort to do. To actually perform the merging, we’re going to define a helper function that joins the data together.'''

def merge_sort(items):
  if len(items) <= 1:
    return items

  middle_index = len(items) // 2
  left_split = items[:middle_index]
  right_split = items[middle_index:]

  return middle_index, left_split, right_split

def merge(left, right):
  result = []
  return result
  
'''
Merging
Now we need to build out our result list. When we’re merging our lists together, we’re creating ordered lists that combine the elements of two lists.

Instructions
1.
Since we’re going to be removing the contents of each list until they’re both depleted, let’s start with a while loop!

Create a loop that will continue iterating while both left and right have elements. When one of those two are empty we’ll want to move on.

Remember a list is truthy if it has elements and falsy if it’s empty, so writing:

while(left and right):
Will continue until one of those two is depleted.

2.
Now we do our comparison! Check if the first element (index 0, remember) of left is smaller than the first element of right.

3.
If left[0] is smaller than right[0], we want to add it to our result! Append left[0] to our result list.

Since we’ve added it to our results we’ll want to remove it from left. Use left.pop() to remove the first element from the left list.

Add the element to result and pop it off from the front of the list:

result.append(left[0])
left.pop(0)
4.
If left[0] is larger than right[0], we want to add right[0] to our result! Append right[0] to result and then pop it out of right.'''

def merge_sort(items):
    if len(items) <= 1:
        return items

    middle_index = len(items) // 2
    left_split = items[:middle_index]
    right_split = items[middle_index:]

    return middle_index, left_split, right_split

def merge(left, right):
    result = []
    
    while (left and right):
      if left[0] < right[0]:
        result.append(left[0])
        left.pop(0)
      else:
        result.append(right[0])
        right.pop(0)
        
    return result

'''Finishing the Merge
Since we’ve only technically depleted one of our two inputs to merge(), we want to add in the rest of the values to finish off our merge() function and return the sorted list.

Instructions
1.
After our while loop, check if there are any elements still in left.

If there are, add those elements to the end of result.

You can add two lists using +:

[1, 2, 3] + [4, 5, 6] == [1, 2, 3, 4, 5, 6]
This means we can update our result list using +=

a = [1, 2, 3]
a += [4, 5, 6]

print(a)
# Prints "[1, 2, 3, 4, 5, 6]'''

def merge_sort(items):
  if len(items) <= 1:
    return items

  middle_index = len(items) // 2
  left_split = items[:middle_index]
  right_split = items[middle_index:]

  return middle_index, left_split, right_split

def merge(left, right):
  result = []

  while (left and right):
    if left[0] < right[0]:
      result.append(left[0])
      left.pop(0)
    else:
      result.append(right[0])
      right.pop(0)
    
    if left:
      result += left
    if right:
      result += right  

  return result

'''MERGE SORT: PYTHON
Finishing the Sort
Let’s update our merge_sort() function so that it returns a sorted list finally!

Instructions
1.
In merge_sort() create two new variables: left_sorted and right_sorted.

left_sorted should be the result of calling merge_sort() recursively on left_split.

right_sorted should be the result of calling merge_sort() recursively on right_split.

2.
Erase the “return” line and change it to return the result of calling merge() on left_sorted and right_sorted.'''


def merge_sort(items):
  if len(items) <= 1:
    return items

  middle_index = len(items) // 2
  left_split = items[:middle_index]
  right_split = items[middle_index:]

  #return middle_index, left_split, right_split

  left_sorted = merge_sort(left_split)
  right_sorted = merge_sort(right_split)

  return merge(left_sorted, right_sorted)

def merge(left, right):
  result = []

  while (left and right):
    if left[0] < right[0]:
      result.append(left[0])
      left.pop(0)
    else:
      result.append(right[0])
      right.pop(0)

  if left:
    result += left
  if right:
    result += right

  return result

'''MERGE SORT: PYTHON
Finishing the Sort
Let’s update our merge_sort() function so that it returns a sorted list finally!

Instructions
1.
In merge_sort() create two new variables: left_sorted and right_sorted.

left_sorted should be the result of calling merge_sort() recursively on left_split.

right_sorted should be the result of calling merge_sort() recursively on right_split.

2.
Erase the “return” line and change it to return the result of calling merge() on left_sorted and right_sorted.'''

def merge_sort(items):
  if len(items) <= 1:
    return items

  middle_index = len(items) // 2
  left_split = items[:middle_index]
  right_split = items[middle_index:]

  #return middle_index, left_split, right_split

  left_sorted = merge_sort(left_split)
  right_sorted = merge_sort(right_split)

  return merge(left_sorted, right_sorted)

def merge(left, right):
  result = []

  while (left and right):
    if left[0] < right[0]:
      result.append(left[0])
      left.pop(0)
    else:
      result.append(right[0])
      right.pop(0)

  if left:
    result += left
  if right:
    result += right

  return result

'''MERGE SORT: PYTHON
Testing the Sort
We’ve written our merge sort! The whole sort takes up two functions:

merge_sort() which is called recursively breaks down an input list to smaller, more manageable pieces. merge() which is a helper function built to help combine those broken-down lists into ordered combination lists.

merge_sort() continues to break down an input list until it only has one element and then it joins that with other single element lists to create sorted 2-element lists. Then it combines 2-element sorted lists into 4-element sorted lists. It continues that way until all the items of the lists are sorted!

Only one thing left to do, test it out!'''

def merge_sort(items):
  if len(items) <= 1:
    return items

  middle_index = len(items) // 2
  left_split = items[:middle_index]
  right_split = items[middle_index:]

  left_sorted = merge_sort(left_split)
  right_sorted = merge_sort(right_split)

  return merge(left_sorted, right_sorted)

def merge(left, right):
  result = []

  while (left and right):
    if left[0] < right[0]:
      result.append(left[0])
      left.pop(0)
    else:
      result.append(right[0])
      right.pop(0)

  if left:
    result += left
  if right:
    result += right

  return result

unordered_list1 = [356, 746, 264, 569, 949, 895, 125, 455]
unordered_list2 = [787, 677, 391, 318, 543, 717, 180, 113, 795, 19, 202, 534, 201, 370, 276, 975, 403, 624, 770, 595, 571, 268, 373]
unordered_list3 = [860, 380, 151, 585, 743, 542, 147, 820, 439, 865, 924, 387]

ordered_list1 = merge_sort(unordered_list1)
ordered_list2 = merge_sort(unordered_list2)
ordered_list3 = merge_sort(unordered_list3)

print(ordered_list1)
print(ordered_list2)
print(ordered_list3)


'''QUICKSORT: CONCEPTUAL
Introduction to Quicksort
Quicksort is an efficient recursive algorithm for sorting arrays or lists of values. The algorithm is a comparison sort, where values are ordered by a comparison operation such as > or <.

Quicksort uses a divide and conquer strategy, breaking the problem into smaller sub-problems until the solution is so clear there’s nothing to solve.

The problem: many values in the array which are out of order.

The solution: break the array into sub-arrays containing at most one element. One element is sorted by default!

We choose a single pivot element from the list. Every other element is compared with the pivot, which partitions the array into three groups.

A sub-array of elements smaller than the pivot.
The pivot itself.
A sub-array of elements greater than the pivot.
The process is repeated on the sub-arrays until they contain zero or one element. Elements in the “smaller than” group are never compared with elements in the “greater than” group. If the smaller and greater groupings are roughly equal, this cuts the problem in half with each partition step!

[6,5,2,1,9,3,8,7]
6 # The pivot
[5, 2, 1, 3] # lesser than 6
[9, 8, 7] # greater than 6


[5,2,1,3]  # these values
# will never be compared with 
[9,8,7] # these values
Depending on the implementation, the sub-arrays of one element each are recombined into a new array with sorted ordering, or values within the original array are swapped in-place, producing a sorted mutation of the original array.'''

'''QUICKSORT: CONCEPTUAL
Quicksort Runtime
The key to Quicksort’s runtime efficiency is the division of the array. The array is partitioned according to comparisons with the pivot element, so which pivot is the optimal choice to produce sub-arrays of roughly equal length?

The graphic displays two data sets which always use the first element as the pivot. Notice how many more steps are required when the quicksort algorithm is run on an already sorted input. The partition step of the algorithm hardly divides the array at all!

The worst case occurs when we have an imbalanced partition like when the first element is continually chosen in a sorted data-set.

One popular strategy is to select a random element as the pivot for each step. The benefit is that no particular data set can be chosen ahead of time to make the algorithm perform poorly.

Another popular strategy is to take the first, middle, and last elements of the array and choose the median element as the pivot. The benefit is that the division of the array tends to be more uniform.

Quicksort is an unusual algorithm in that the worst case runtime is O(N^2), but the average case is O(N * logN).

We typically only discuss the worst case when talking about an algorithm’s runtime, but for Quicksort it’s so uncommon that we generally refer to it as O(N * logN).'''

'''QUICKSORT: CONCEPTUAL
Quicksort Review
Quicksort is an efficient algorithm for sorting values in a list. A single element, the pivot, is chosen from the list. All the remaining values are partitioned into two sub-lists containing the values smaller than and greater than the pivot element.

Ideally, this process of dividing the array will produce sub-lists of nearly equal length, otherwise, the runtime of the algorithm suffers.

When the dividing step returns sub-lists that have one or less elements, each sub-list is sorted. The sub-lists are recombined, or swaps are made in the original array, to produce a sorted list of values.'''

'''QUICKSORT: PYTHON
Quicksort Introduction
We’ll be implementing a version of the quicksort algorithm in Python. Quicksort is an efficient way of sorting a list of values by partitioning the list into smaller sub-lists based on comparisons with a single “pivot” element.

Our algorithm will be recursive, so we’ll have a base case and an inductive step that takes us closer to the base case. We’ll also sort our list in-place to keep it as efficient as possible.

Sorting in place means we’ll need to keep track of the sub-lists in our algorithm using pointers and swap values inside the list rather than create new lists.

We’ll use pointers a lot in this algorithm so it’s worth spending a little time practicing. Pointers are indices that keep track of a portion of a list. Here’s an example of using pointers to represent the entire list:

my_list = ['pizza', 'burrito', 'sandwich', 'salad', 'noodles']
start_of_list = 0
end_of_list = len(my_list) - 1

my_list[start_of_list : end_of_list + 1]
# ['pizza', 'burrito', 'sandwich', 'salad', 'noodles']
Now, what if we wanted to keep my_list the same, but make a sub-list of only the first half?

end_of_half_sub_list = len(my_list) // 2
# 2

my_list[start_of_list : end_of_half_sub_list + 1]
# ['pizza', 'burrito', 'sandwich']
Finally, let’s make a sub-list that excludes the first and last elements…

start_of_sub_list = 1
end_of_sub_list = len(my_list) - 2

my_list[start_of_sub_list : end_of_sub_list]
# ['burrito', 'sandwich', 'salad']
Nice work! We’ll use two pointers, start and end to keep track of sub-lists in our algorithm. Let’s get started!'''

# Define your quicksort function
  
def quicksort(list, start, end):
  if start >= end:
    return

  print(list[start])
  start += 1
  quicksort(list, start, end)

colors = ["blue", "red", "green", "purple", "orange"]
quicksort(colors, 0, len(colors) - 1)

'''QUICKSORT: PYTHON
Pickin' Pivots
Quicksort works by selecting a pivot element and dividing the list into two sub-lists of values greater than or less than the pivot element’s value. This process of “partitioning” the list breaks the problem down into two smaller sub-lists.

For the algorithm to remain efficient, those sub-lists should be close to equal in length. Here’s an example:

[9, 3, 21, 4, 50, 8, 11]
# pick the first element, 9, as the pivot
# "lesser_than_list" becomes [3, 4, 8]
# "greater_than_list" becomes [21, 50, 11]
In this example the two sub-lists are equal in length, but what happens if we pick the first element as a pivot in a sorted list?

[1, 2, 3, 4, 5, 6]
# pick the first element, 1, as the pivot
# "lesser_than_list" becomes []
# "greater_than_list" becomes [2,3,4,5,6]
Our partition step has produced severely unbalanced sub-lists! While it may seem silly to sort an already sorted list, this is a common enough occurrence that we’ll need to make an adjustment.

We can avoid this problem by randomly selecting a pivot element from the list each time we partition. The benefit of random selection is that no particular data set will consistently cause trouble for the algorithm! We’ll then swap this random element with the last element of the list so our code consistently knows where to find the pivot.'''

'''
Instructions
1.
We’ve imported the randrange() function to assist with the random pivot. Check the documentation for how it works.

Use this function to create the variable pivot_idx, a random index between start and end.

randrange() can take two arguments which we’ll use to give us the bounds of a random number.

The second argument is not inclusive.

randrange(1, 10) # anything from 1 to 9
randrange(1, 11) # anything from 1 to 10

randrange(start, end) 
# anything from start to end - 1
2.
Make another variable pivot_element and use pivot_idx to retrieve the value located in the list which was passed in as an argument.

3.
Random is great because it protects our algorithm against inefficient runtimes, but our code will be simpler for the remainder of the algorithm if we know the pivot will always be in the same place.

Swap the end element of the list with the pivot_idx so we know the pivot element will always be located at the end of the list.'''

# use randrange to produce a random index
from random import randrange

def quicksort(list, start, end):
  if start >= end:
    return list
	# Define your pivot variables below
  pivot_idx = randrange(start, end)
  # Swap the elements in list below
  pivot_element = list[pivot_idx]
  list[pivot_idx] = list[-1]
  list[-1] = pivot_element

  # Leave these lines for testing
  print(list[start])
  start += 1
  return quicksort(list, start, end)


my_list = [32, 22]
print("BEFORE: ", my_list)
sorted_list = quicksort(my_list, 0, len(my_list) - 1)
print("AFTER: ", sorted_list)

'''QUICKSORT: PYTHON
Partitioning Party
We need to partition our list into two sub-lists of greater than or smaller than elements, and we’re going to do this “in-place” without creating new lists. Strap in, this is the most complex portion of the algorithm!

Because we’re doing this in-place, we’ll need two pointers. One pointer will keep track of the “lesser than” elements. We can think of it as the border where all values at a lower index are lower in value to the pivot. The other pointer will track our progress through the list.

Let’s explore how this will work in an example:

[5, 6, 2, 3, 1, 4]
# we randomly select "3" and swap with the last element
[5, 6, 2, 4, 1, 3]

# We'll use () to mark our "lesser than" pointer
# We'll use {} to mark our progress through the list

[{(5)}, 6, 2, 4, 1, 3]
# {5} is not less than 3, so the "lesser than" pointer doesn't move

[(5), {6}, 2, 4, 1, 3]
# {6} is not less than 3, so the "lesser than" pointer doesn't move

[(5), 6, {2}, 4, 1, 3]
# {2} is less than 3, so we SWAP the values...
[(2), 6, {5}, 4, 1, 3]
# Then we increment the "lesser than" pointer
[2, (6), {5}, 4, 1, 3]

[2, (6), 5, {4}, 1, 3]
# {4} is not less than 3, so the "lesser than" pointer doesn't move

[2, (6), 5, 4, {1}, 3]
# {1} is less than 3, so we SWAP the values...
[2, (1), 5, 4, {6}, 3]
# Then we increment the "lesser than" pointer
[2, 1, (5), 4, {6}, 3]

# We've reached the end of the non-pivot values
[2, 1, (5), 4, 6, {3}]
# Swap the "lesser than" pointer with the pivot...
[2, 1, (3), 4, 6, {5}]
Tada! We have successfully partitioned this list. Note that the “sub-lists” are not necessarily sorted, we’ll need to recursively run the algorithm on each sub-list, but the pivot has arrived at the correct location within the list.'''

'''
Create the variable lesser_than_pointer and assign it to the start of the list.

2.
Create a for loop that iterates from start to end, and set the iterating variable to idx. This will track our progress through the list (or sub-list) we’re partitioning.

To start, write continue in the for loop.

3.
Within the loop, remove continue and replace it with a conditional.

We need to do something if the element at idx is less than pivot_element.

If so:

Use parallel assignment to swap the values at lesser_than_pointer and idx.
Increment the lesser_than_pointer
We can check if list[idx] < pivot_element: to see if any alterations should be made to the list.

Here’s an example of swapping values with parallel assignment:

colors = ["blue", "red", "green"]
colors[0], colors[1] = colors[1], colors[0]
# ["red", "blue", "green"]
4.
Once the loop concludes, use parallel assignment to swap the pivot element with the value located at lesser_than_pointer.

Here’s an example:

colors = ["blue", "red", "green"]
colors[0], colors[1] = colors[1], colors[0]
# ["red", "blue", "green"]'''

from random import randrange

def quicksort(list, start, end):
  if start >= end:
    return list

  pivot_idx = randrange(start, end)
  pivot_element = list[pivot_idx]
  list[end], list[pivot_idx] = list[pivot_idx], list[end]

  # Create the lesser_than_pointer
  lesser_than_pointer = start

  # Start a for loop, use 'idx' as the variable
  for idx in range(start, end):
    # Check if the value at idx is less than the pivot
      if list[idx] < pivot_element:
      # If so: 
        # 1) swap lesser_than_pointer and idx values
        # 2) increment lesser_than_pointer
        list[idx], list[lesser_than_pointer] = list[lesser_than_pointer], list[idx]
        lesser_than_pointer += 1
      
  # After the loop is finished...
  # swap pivot with value at lesser_than_pointer
  list[end], list[lesser_than_pointer] = list[lesser_than_pointer], list[end]
  


  print(list[start])
  start += 1
  return quicksort(list, start, end)

my_list = [42, 103, 22]
print("BEFORE: ", my_list)
sorted_list = quicksort(my_list, 0, len(my_list) - 1)
print("AFTER: ", sorted_list)

'''QUICKSORT: PYTHON
Recurse, Rinse, Repeat
We’ve made it through the trickiest portion of the algorithm, but we’re not quite finished. We’ve partitioned the list once, but we need to continue partitioning until the base case is met.

Let’s revisit our example from the previous exercise where we had finished a single partition step:

# the pivot, 3, is correctly placed
whole_list = [2, 1, (3), 4, 6, 5]

less_than_pointer = 2
start = 0
end = len(whole_list) - 1
# start and end are pointers encompassing the entire list
# pointers for the "lesser than" sub-list
left_sub_list_start = start
left_sub_list_end = less_than_pointer - 1

lesser_than_sub_list = whole_list[left_sub_list_start : left_sub_list_end]
# [2, 1]

# pointers for the "greater than" sub-list
right_sub_list_start = less_than_pointer + 1
right_sub_list_end = end
greater_than_sub_list = whole_list[right_sub_list_start : right_sub_list_end]
# [4, 6, 5]
The key insight is that we’ll recursively call quicksort and pass along these updated pointers to mark the various sub-lists. Make sure you’re excluding the index that stores the newly placed pivot value or we’ll never hit the base case!'''

from random import randrange, shuffle 
def quicksort(list, start, end):
  # this portion of listay has been sorted
  if start >= end:
    return

  # select random element to be pivot
  pivot_idx = randrange(start, end + 1)
  pivot_element = list[pivot_idx]

  # swap random element with last element in sub-listay
  list[end], list[pivot_idx] = list[pivot_idx], list[end]

  # tracks all elements which should be to left (lesser than) pivot
  less_than_pointer = start
  
  for i in range(start, end):
    # we found an element out of place
    if list[i] < pivot_element:
      # swap element to the right-most portion of lesser elements
      list[i], list[less_than_pointer] = list[less_than_pointer], list[i]
      # tally that we have one more lesser element
      less_than_pointer += 1
  # move pivot element to the right-most portion of lesser elements
  list[end], list[less_than_pointer] = list[less_than_pointer], list[end]
  
  # Call quicksort on the "left" and "right" sub-lists

  quicksort(list, start, less_than_pointer - 1)
  quicksort(list, less_than_pointer + 1, end)
  
unsorted_list = [3,7,12,24,36,42]
shuffle(unsorted_list)
print(unsorted_list)
# use quicksort to sort the list, then print it out!
quicksort(unsorted_list,0,-1)
print(unsorted_list)

'''QUICKSORT: PYTHON
Quicksort Review
Congratulations on implementing the quicksort algorithm in Python. To review:

We established a base case where the algorithm will complete when the start and end pointers indicate a list with one or zero elements
If we haven’t hit the base case, we randomly selected an element as the pivot and swapped it to the end of the list
We then iterate through that list and track all the “lesser than” elements by swapping them with the iteration index and incrementing a lesser_than_pointer.
Once we’ve iterated through the list, we swap the pivot element with the element located at lesser_than_pointer.
With the list partitioned into two sub-lists, we repeat the process on both halves until base cases are met.'''

from random import randrange, shuffle

def quicksort(list, start, end):
  # this portion of list has been sorted
  if start >= end:
    return
  print("Running quicksort on {0}".format(list[start: end + 1]))
  # select random element to be pivot
  pivot_idx = randrange(start, end + 1)
  pivot_element = list[pivot_idx]
  print("Selected pivot {0}".format(pivot_element))
  # swap random element with last element in sub-lists
  list[end], list[pivot_idx] = list[pivot_idx], list[end]

  # tracks all elements which should be to left (lesser than) pivot
  less_than_pointer = start
  
  for i in range(start, end):
    # we found an element out of place
    if list[i] < pivot_element:
      # swap element to the right-most portion of lesser elements
      print("Swapping {0} with {1}".format(list[i], pivot_element))
      list[i], list[less_than_pointer] = list[less_than_pointer], list[i]
      # tally that we have one more lesser element
      less_than_pointer += 1
  # move pivot element to the right-most portion of lesser elements
  list[end], list[less_than_pointer] = list[less_than_pointer], list[end]
  print("{0} successfully partitioned".format(list[start: end + 1]))
  # recursively sort left and right sub-lists
  quicksort(list, start, less_than_pointer - 1)
  quicksort(list, less_than_pointer + 1, end)


    
  
list = [5,3,1,7,4,6,2,8]
shuffle(list)
print("PRE SORT: ", list)
print(quicksort(list, 0, len(list) -1))
print("POST SORT: ", list)

'''
RADIX SORT: CONCEPTUAL
What Is A Radix
Quick, which number is bigger: 1489012 or 54? It’s 1489012, but how can you tell? It has more digits so it has to be larger, but why exactly is that the case?

Our number system was developed by 8th century Arabic mathematicians and was successful because it made arithmetic operations more sensible and larger numbers easier to write and comprehend.

The breakthrough those mathematicians made required defining a set of rules for how to depict every number. First we decide on an alphabet: different glyphs, or digits, that we’ll use to write our numbers with. The alphabet that we use to depict numbers in this system are the ten digits 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9. We call the length of this alphabet our radix (or base). So for our decimal system, we have a radix of 10.

Next we need to understand what those digits mean in different positions. In our system we have a ones place, a tens place, a hundreds place and so on. So what do digits mean in each of those places?

This is where explaining gets a little complicated because the actual knowledge might feel very fundamental. There’s a difference, for instance, between the digit ‘6’ and the actual number six that we represent with the digit ‘6’. This difference is similar to the difference between the letter ‘a’ (which we can use in lots of words) and the word ‘a’.

But the core of the idea is that we use these digits to represent different values when they’re used in different positions. The digit 6 in the number 26 represents the value 6, but the digit 6 used in the number 86452 represents the value 6000.

RADIX SORT: CONCEPTUAL
Base Numbering Systems
The value of different positions in a number increases by a multiplier of 10 in increasing positions. This means that a digit ‘8’ in the rightmost place of a number is equal to the value 8, but that same digit when shifted left one position (i.e., in 80) is equal to 10 * 8. If you shift it again one position you get 800, which is 10 * 10 * 8.

This is where it’s useful to incorporate the shorthand of exponential notation. It’s important to note that 100 is equal to 1. Each position corresponds to a different exponent of 10.

So why 10? It’s a consequence of how many digits are in our alphabet for numbering. Since we have 10 digits (0-9) we can count all the way up to 9 before we need to use a different position. This system that we used is called base-10 because of that.

Sorting By Radix
So how does a radix sort use this base numbering system to sort integers? First, there are two different kinds of radix sort: most significant digit, or MSD, and least significant digit, or LSD.

Both radix sorts organize the input list into ten “buckets”, one for each digit. The numbers are placed into the buckets based on the MSD (left-most digit) or LSD (right-most digit). For example, the number 2367 would be placed into the bucket “2” for MSD and into “7” for LSD.

This bucketing process is repeated over and over again until all digits in the longest number have been considered. The order within buckets for each iteration is preserved. For example, the numbers 23, 25 and 126 are placed in the “3”, “5”, and “6” buckets for an initial LSD bucketing. On the second iteration of the algorithm, they are all placed into the “2” bucket, but the order is preserved as 23, 25, 126.

Radix Sort Performance
The most amazing feature of radix sort is that it manages to sort a list of integers without performing any comparisons whatsoever. We call this a non-comparison sort.

This makes its performance a little difficult to compare to most other comparison-based sorts. Consider a list of length n. For each iteration of the algorithm, we are deciding which bucket to place each of the n entries into.

How many iterations do we have? Remember that we continue iterating until we examine each digit. This means we need to iterate for how ever many digits we have. We’ll call this average number of digits the word-size or w.

This means the complexity of radix sort is O(wn). Assuming the length of the list is much larger than the number of digits, we can consider w a constant factor and this can be reduced to O(n).

Radix Review
Take a moment to review radix sort:

A radix is the base of a number system. For the decimal number system, the radix is 10.
Radix sort has two variants - MSD and LSD
Numbers are bucketed based on the value of digits moving left to right (for MSD) or right to left (for LSD)
Radix sort is considered a non-comparison sort
The performance of radix sort is O(n)

RADIX SORT: PYTHON
Finding the Max Exponent
In our version of least significant digit radix sort, we’re going to utilize the string representation of each integer. This, combined with negative indexing, will allow us to count each digit in a number from right-to-left.

Some other implementations utilize integer division and modular arithmetic to find each digit in a radix sort, but our goal here is to build an intuition for how the sort works.

Our first step is going to be finding the max_exponent, which is the number of digits long the largest number is. We’re going to find the largest number, cast it to a string, and take the length of that string.

Instructions
1.
Define your function radix_sort() that takes a list as input and call that input to_be_sorted.

2.
In order to determine how many digits are in the longest number in the list, we’ll need to find the longest number.

Declare a new variable maximum_value and assign the max() of to_be_sorted to it.

3.
Now we want to define our max_exponent.

First, cast maximum_value to a string.
Then take the len() of that string.
Then assign that len() to a variable called max_exponent.
Then return max_exponent.
Use str(number) to convert number into a string.'''

def radix_sort(to_be_sorted):
  maximum_value = max(to_be_sorted)
  max_exponent = len(str(maximum_value))
  return max_exponent

'''
RADIX SORT: PYTHON
Setting Up For Sorting
In this implementation, we’re going to build the radix sort naturally, from the inside out. The first step we’re going to take is going to be our inner-most loop, so that we know we’ll be solid when we’re iterating through each of the exponents.

Instructions
1.
By the nature of a radix sort we need to erase and rewrite our output list a number of times. It would be bad practice to mutate the input list — in case something goes wrong with our code, or someone using our sort decides they don’t want to wait anymore. We wouldn’t want anyone out there to have to deal with the surprise of having their precious list of integers mangled.

Create a copy of to_be_sorted and save the copy into a new list called being_sorted.

Remember you can make a copy by using the slice (:) syntax without any arguments:

cool_list = [1, 19, 22]

# create a copy of cool_list
imitation_is_flattery = cool_list[:]
2.
A radix sort goes through each position of each number and puts all of the inputs in different buckets depending on the value . Since there are 10 different values (that is, 0 through 9) that a position can have, we need to create ten different buckets to put each number in.

Create a list of ten empty lists and assign the result to a variable called digits. Then return digits.

There are plenty of ways to create a list of ten empty lists, but a list comprehension would be an efficient way to do this:

six_empty_lists = [[] for digit in range(6)]'''

def radix_sort(to_be_sorted):
  maximum_value = max(to_be_sorted)
  max_exponent = len(str(maximum_value))

  # create copy of to_be_sorted here
  being_sorted = to_be_sorted[:]
  
  digits = [[] for i in range(10)]
  return digits
  
'''
RADIX SORT: PYTHON
Bucketing Numbers
The least significant digit radix sort algorithm takes each number in the input list, looks at the digits of that number in order from right to left, and incrementally stuffs each number into the bucket corresponding to the value of that digit.

First we’re going to write this logic for the least significant digit, then we’re going to loop over the code we write to do that for every digit.

Instructions
1.
We’ll need to iterate over being_sorted. Grab each value of being_sorted and save it as the temporary variable number.

Use the invocation

for temporary_variable in list_being_iterated:
To create a temporary variable called temporary_variable that stores each value of list_being_iterated.

2.
Now convert number to a string and save that as number_as_a_string.

You can convert almost anything in Python to a string using the str() function.

3.
How do we get the last element of a string? This would correspond to the least significant digit of the number. For strings, this is simple, we can use a negative index.

Save the last element of number_as_a_string to the variable digit.

We can get the last digit of number_as_a_string from the following:

number_as_a_string[-1]
4.
Now that we have a string containing the least significant digit of number saved to the variable digit. We want to use digit as a list index for digits. Unfortunately, it needs to be an integer to do that. But that should be easy for us to do:

Set digit equal to the integer form of digit.

You can use int() to cast a string to an integer. So something like:

an_int = int(a_string)
5.
We know that digits[digit] is an empty list (because digits has ten lists and digit is a number from 0 to 9). So let’s add our number to that list!

Call .append() on digits[digit] with the argument number.

6.
Now break out of the for loop and return digits.'''

def radix_sort(to_be_sorted):
  maximum_value = max(to_be_sorted)
  max_exponent = len(str(maximum_value))

  being_sorted = to_be_sorted[:]
  digits = [[] for i in range(10)]

  # create for loop here:
  for number in being_sorted:
    number_as_a_string = str(number)
    digit = number_as_a_string[-1]
    digit = int(digit)
    digits[digit].append(number)
  return digits

'''
Rendering the List
For every iteration, radix sort renders a version of the input list that is sorted based on the digit that was just looked at. At first pass, only the ones digit is sorted. At the second pass, the tens and the ones are sorted. This goes on until the digits in the largest position of the largest number in the list are all sorted, and the list is sorted at that time.

Here we’ll be rendering the list, at first, it will just return the list sorted so just the ones digit is sorted.

Instructions
1.
Outside of our for loop which appends the numbers in our input list to the different buckets in digits, let’s render the list.

Since we know that all of our input numbers are in digits we can safely clear out being_sorted. We’ll make it an empty list and then add back in all the numbers from digits as they appear.

Assign an empty list to being_sorted.

2.
Now, create a for loop that iterates through each of our lists in digits.

Call each of these lists numeral because they each correspond to one specific numeral from 0 to 9.

3.
Now use the .extend() method (which appends all the elements of a list, instead of appending the list itself) to add the elements of numeral to being_sorted.

Use the .extend() method:

big_list = [1, 51, 801, 42, 302]
smaller_list = [15, 905]
big_list.extend(smaller_list)
4.
Unindent out of the for loop and return being_sorted.'''

def radix_sort(to_be_sorted):
  maximum_value = max(to_be_sorted)
  max_exponent = len(str(maximum_value))

  being_sorted = to_be_sorted[:]
  digits = [[] for i in range(10)]

  for number in being_sorted:
    number_as_a_string = str(number)
    digit = number_as_a_string[-1]
    digit = int(digit)
    
    digits[digit].append(number)

  # reassign being_sorted here:
    being_sorted = []

    for numeral in digits:
      being_sorted.extend(numeral)
  return being_sorted



'''
RADIX SORT: PYTHON
Iterating through Exponents
We have the interior of our radix sort, which right now goes through a list and sorts it by the first digit in each number. That’s a pretty great start actually. It won’t be hard for us to go over every digit in a number now that we can already sort by a given digit.

Instructions
1.
After defining being_sorted for the first time in the function (and before defining digits which we’ll need per iteration), create a new for loop that iterates through the range() of max_exponent.

Use the variable name exponent as a temporary variable in your for loop, it will count the current exponent we’re looking at for each number.

2.
Now indent the rest of your function after this new for loop.

(Tip: You can highlight the text in your code editor and use the Tab key to increase the indentation of code.)

3.
In our for loop we’re going to want to create the index we’ll use to get the appropriate position in the numbers we’re sorting.

First we’re going to create the position variable, which keeps track of what exponent we’re looking at. Since exponent is zero-indexed our position is always going to be one more than the exponent. Assign to it the value of exponent + 1.

4.
Now we want to create our index that we’ll be using to index into each number! This index is going to be roughly the same as position, but since we’re going to be indexing the string in reverse it needs to be negative!

Set index equal to -position.

5.
Now in the body of our loop, let’s update our digit lookup to get the digit at the given index. Where we before used number_as_a_string[-1] we’ll want to start accessing [index] instead.

Update the line of code where we first define digit to access index in number_as_a_string.

6.
Now we’ve got a sort going! At the very end of our function, de-indenting out of all the for loops (but not the function itself), return being_sorted. It will be sorted by this point!'''
def radix_sort(to_be_sorted):
  maximum_value = max(to_be_sorted)
  max_exponent = len(str(maximum_value))
  being_sorted = to_be_sorted[:]

  # Add new for-loop here:

  for exponent in range(max_exponent):

    position = exponent + 1

    index = -position

    digits = [[] for i in range(10)]

    for number in being_sorted:
      number_as_a_string = str(number)
      digit = number_as_a_string[index]
      digit = int(digit)
      
      digits[digit].append(number)

    being_sorted = []
    for numeral in digits:
      being_sorted.extend(numeral)

  return being_sorted

'''
RADIX SORT: PYTHON
Review (and Bug Fix!)
Now that we’ve finished writing our radix sort we’re finished for the day… or are we?

Instructions
1.
Now that we’ve gotten our sort working let’s test it out with some new data.

Run radix_sort on unsorted_list.

2.
What? IndexError? Did we forget something?

We did! Some of the numbers that we’re sorting are going to be shorter than other numbers.

We can fix it though! First, we should comment out the line we added to test the sort.

Add a comment with #:

neat_code = 20 + cool_function()
# This is a comment
3.
Where we defined digit to be the value of number_as_a_string at index index we need to now wrap that definition in a try block.

Add a try block and, indented in that block, leave your original definition of digit.

Add a try block to your code by adding the try keyword and an indented block:

try:
  cool_list = [0, 30, 18]

  # raises IndexError
  cool_value = cool_list[-5]
4.
After the try block, we’ll want to handle the possibility of an IndexError. What does it mean if we get an index error here?

It means the value for number at index is actually 0.

Handle the exception by adding an except IndexError block, in this case assigning digit to be 0.

Introduce an except block to your existing try block:

try:
  cool_list = [0, 30, 18]

  # raises IndexError
  cool_value = cool_list[-5]

except IndexError:
  # 5 is a cool value that will have to do.
  cool_value = 5
5.
Excellent! Now let’s try uncommenting the line where we sort unordered_list. Print out the results.

6.
Great job! We created an algorithm that:

Takes numbers in an input list.
Passes through each digit in those numbers, from least to most significant.
Looks at the values of those digits.
Buckets the input list according to those digits.
Renders the results from that bucketing.
Repeats this process until the list is sorted.
And that’s what a radix sort does! Feel free to play around with the solution code, see if there’s anything you can improve about the code or a different way of writing it you want to try.'''

def radix_sort(to_be_sorted):
  maximum_value = max(to_be_sorted)
  max_exponent = len(str(maximum_value))
  being_sorted = to_be_sorted[:]

  for exponent in range(max_exponent):
    position = exponent + 1
    index = -position

    digits = [[] for i in range(10)]

    for number in being_sorted:
      number_as_a_string = str(number)

      try:
        digit = number_as_a_string[index]
      
      except IndexError:

        digit = 0
      
      digit = int(digit)

      digits[digit].append(number)

    being_sorted = []
    for numeral in digits:
      being_sorted.extend(numeral)

  return being_sorted

unsorted_list = [830, 921, 163, 373, 961, 559, 89, 199, 535, 959, 40, 641, 355, 689, 621, 183, 182, 524, 1]

print(radix_sort(unsorted_list))


#--------------------------- practice project ---------------------------------------

'''SORTING ALGORITHMS IN PYTHON
A Sorted Tale
You recently began employment at “A Sorted Tale”, an independent bookshop. Every morning, the owner decides to sort the books in a new way.

Some of his favorite methods include:

By author name
By title
By number of characters in the title
By the reverse of the author’s name
Throughout the day, patrons of the bookshop remove books from the shelf. Given the strange ordering of the store, they do not always get the books placed back in exactly the correct location.

The owner wants you to research methods of fixing the book ordering throughout the day and sorting the books in the morning. It is currently taking too long!

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
19/20Complete
Mark the tasks as complete by checking them off
Get to know the data
1.
The owner provides the current state of the bookshelf in a comma-separated values, or csv, file. To get you started, we have provided a function load_books, defined in utils.py.

Within script.py, we are loading the books from books_small.csv. This list of 10 books makes it easier to determine how the algorithms are behaving. we’ll use this to develop the algorithms and then we’ll try them out later on the larger file.

Add a for loop to print the titles within the bookshelf.

Save your code and run it using python3 script.py in the terminal.

for book in bookshelf:
  print(book['title'])
2.
Today’s sorting is by title and looking at the bookshelf it’s pretty close. Some patrons have placed books back in slightly the wrong place.

Before we start solving the problem, we need to do a bit of data manipulation to ensure that we compare books correctly. Python’s built-in comparison operators compare letters lexicographically based on their Unicode code point. You can determine the code point of characters using Python’s built-in ord function. For example to calculate the code point for “z” you would use the following code:

ord("z")
Try this in script.py using print statements:

What is the code point of “a”?
What about “ “?
What about “A”?
print(ord("a"))
print(ord(" "))
print(ord("A"))
3.
You may have noticed that the uppercase letters have values less than their lowercase counterparts. When sorting, we don’t want to take into account the case of the letters. For example, “cats” should come before “Dogs”, even though ord("c") > ord("D") is True.

We’ll make this happen by converting everything to lowercase prior to comparison. Since we need to do this often, lets save the lowercase author and title as attributes while loading the bookshelf in utils.py:

book['author_lower']
book['title_lower']
For the author, the statement is:

book['author_lower'] = book['author'].lower()
Fix the midday errors
4.
As we noted, our books are pretty close to being sorted by title. From the sorting lessons, you may remember that bubble sort performs well for nearly sorted data such as this.

The code for performing bubble sort on an array of numbers is provided in sorts.py. However, we are sorting on books which are Python dictionaries. Further, the owner likes to change the ordering of books daily. To make the sort order flexible, add an argument comparison_function. This will allow us to pass in a custom function for comparing the order of two books.

Within sorts.py, change the bubble sort first line to:

def bubble_sort(arr, comparison_function):
5.
Our comparison_function will take two arguments, and return True if the first one is “greater than” the second.

Within the body of the bubble sort function, modify the comparison conditional statement to use the comparison_function instead of the built in operators (if arr[idx] > arr[idx + 1]:).

if comparison_function(arr[idx], arr[idx + 1]):
6.
Now that we have a bubble sort algorithm that can work on books, let’s give it a shot. Within script.py define a sort comparison function, by_title_ascending.

It should take book_a and book_b as arguments.

It should return True if the title_lower of book_a is “greater than” the title_lower of book_b and False otherwise.

def by_title_ascending(book_a, book_b):
  return book_a['title_lower'] > book_b['title_lower']
7.
Sort the bookshelf using bubble sort. Save the result as sort_1 and print the titles to the console to verify the order.

How many swaps were necessary?

sort_1 = sorts.bubble_sort(bookshelf, by_title_ascending)

for book in sort_1:
  print(book['title'])
A new sorting order
8.
The owner of the bookshop wants to sort by the author’s full name tomorrow. Define a new comparison function, by_author_ascending, within script.py.

It should take book_a and book_b as arguments.

It should return True if the author_lower of book_a is “greater than” the author_lower of book_b and False otherwise.

def by_author_ascending(book_a, book_b):
  return book_a['author_lower'] > book_b['author_lower']
9.
Our sorting algorithms will alter the original bookshelf, so create a new copy of this data, bookshelf_v1.

This does NOT create a copy:

bookshelf_v1 = bookshelf
Use:

bookshelf_v1 = bookshelf.copy()
10.
Try sorting the list of books, bookshelf_v1 using the new comparison function and bubble sort. Save the result as sort_2 and print the authors to the console to verify the order.

How many swaps are needed now?

sort_2 = sorts.bubble_sort(bookshelf_v1, by_author_ascending)
A new sorting algorithm
11.
The number of swaps is getting to be high for even a small list like this. Let’s try implementing a different type of search: quicksort.

The code for quicksort of a numeric array is in sorts.py. We need to modify it in a similar fashion that we modified bubble sort.

Add comparison_function as the final argument to the quicksort function.

def quicksort(list, start, end, comparison_function):
12.
Within the quicksort function, be sure to pass the argument for the comparison_function for the recursive calls.

Update the recursive calls to:

quicksort(list, start, less_than_pointer - 1, comparison_function)
quicksort(list, less_than_pointer + 1, end, comparison_function)
13.
The last modification we need to make to quicksort is to update the comparison conditional. It is currently using the built in comparison:

if pivot_element > list[i]:
Update this to use comparison_function.

if comparison_function(pivot_element, list[i]):
14.
Within script.py create another copy of bookshelf as bookshelf_v2.

bookshelf_v2 = bookshelf.copy()
15.
Perform quicksort on bookshelf_v2 using by_author_ascending. This implementation operates on the input directly, so does not return a list.

Print the authors in bookshelf_v2 to ensure they are now sorted correctly.

sorts.quicksort(bookshelf_v2, 0, len(bookshelf_v2) - 1, by_author_ascending)
The last sort
16.
The owner has asked for one last sorting order, sorting by the length of the sum of the number of characters in the book and author’s name.

Create a new comparison function, by_total_length. It should return True if the sum of characters in the title and author of book_a is “greater than” the sum in book_b and False otherwise.

def by_total_length(book_a, book_b):
  return len(book_a['author_lower']) + len(book_a['title_lower']) > len(book_b['author_lower']) + len(book_b['title_lower'])
17.
Load the long list of books into a new variable, long_bookshelf.

long_bookshelf = utils.load_books('books_large.csv')
18.
Run bubble sort on this algorithm using by_total_length as the comparison function. Does it seem slow?

It should take a couple of seconds to run.

19.
Comment out the bubble sort attempt and now try quicksort. Does it live up to its name?

Quicksort should be noticeably faster.

More sorting
20.
You’ve met the requirements of the project by the bookshop owner. If you’d like, play with creating your own comparison operators or other sorting functions.'''

#--------------------------- script.py------------------------------------------

import utils
import sorts

def by_title_ascending(book_a, book_b):
  if book_a['title_lower'] > book_b['title_lower']:
    return True
  else:
    #return False
    pass

def by_author_ascending(book_a, book_b):
  if book_a['author_lower'] > book_b['author_lower']:
    return True
  else:
    #return False
    pass

def by_total_length(book_a, book_b):
  sum_a = len(book_a['title']) + len(book_a['author'])
  sum_b = len(book_b['title']) + len(book_b['author'])
  if sum_a > sum_b:
    return True
  else:
    pass


bookshelf = utils.load_books('books_small.csv')
long_bookshelf = utils.load_books('books_large.csv')
# make a copy of the original list, because the bubble sorting will alter the original list.
bookshelf_v1 = bookshelf.copy()
bookshelf_v2 = bookshelf.copy()
bookshelf_v3 = bookshelf.copy()


#for book in bookshelf:
#  print(book['title'])

#print(ord("z"))
#print(ord(" "))
#print(ord("a"))
#print(ord("A"))


#bookshelf['author_lower'] = bookshelf['author'].lower()
#bookshelf['title_lower'] = bookshelf['title'].lower()

#print(type(bookshelf))


sort_1 = sorts.bubble_sort(bookshelf, by_title_ascending)
sort_2 = sorts.bubble_sort(bookshelf_v1, by_author_ascending)
#sort_3 = sorts.quicksort(bookshelf_v2, 0, len(bookshelf_v2)-1, by_author_ascending)

for book in sort_1:
  pass
  #print('sorted List')
  #print(book['title'])

for book in sort_2:
  pass
  #print('sorted List')
  #print('{}  by  {}'.format(book['title'],book['author']))

sorts.quicksort(bookshelf_v2, 0, len(bookshelf_v2)-1, by_author_ascending)
#print(bookshelf_v2)
for book in bookshelf_v2:
  #print('{}  by  {}'.format(book['title'],book['author']))
  pass

#sort4 = sorts.bubble_sort(long_bookshelf, by_total_length)
sorts.quicksort(bookshelf_v3, 0, len(bookshelf_v2)-1, by_total_length)
#print(bookshelf_v3)
for book in bookshelf_v3:
  #print('{}  by  {}'.format(book['title'],book['author']))
  pass


#-----------------------  sort.py------------------------------------------


import random

def bubble_sort(arr, comparison_function):
  swaps = 0
  sorted = False
  while not sorted:
    sorted = True
    for idx in range(len(arr) - 1):
      if comparison_function(arr[idx], arr[idx + 1]):
        sorted = False
        arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
        swaps += 1
  print("Bubble sort: There were {0} swaps".format(swaps))
  return arr

def quicksort(list, start, end, comparison_function):
  if start >= end:
    return
  pivot_idx = random.randrange(start, end + 1)
  pivot_element = list[pivot_idx]
  list[end], list[pivot_idx] = list[pivot_idx], list[end]
  less_than_pointer = start
  for i in range(start, end):
    if comparison_function(pivot_element, list[i]):
      list[i], list[less_than_pointer] = list[less_than_pointer], list[i]
      less_than_pointer += 1
  list[end], list[less_than_pointer] = list[less_than_pointer], list[end]
  quicksort(list, start, less_than_pointer - 1, comparison_function)
  quicksort(list, less_than_pointer + 1, end, comparison_function)


#------------------------ utils.py------------------------------------------


import csv

# This code loads the current book
# shelf data from the csv file
def load_books(filename):
  bookshelf = []
  with open(filename) as file:
      shelf = csv.DictReader(file)
      for book in shelf:
          # add your code here
          book['author_lower'] = book['author'].lower()
          book['title_lower'] = book['title'].lower()          
          bookshelf.append(book)

          
  return bookshelf
  
  
#-------------------------- books_small.csv ---------------------------------

title,author
Adventures of Huckleberry Finn,Mark Twain
Best Served Cold,Joe Abercrombie
Dear Emily,Fern Michaels
Collected Poems,Robert Hayden
End Zone,Don DeLillo
Forrest Gump,Winston Groom
Gravity,Tess Gerritsen
Hiromi's Hands,Lynne Barasch
Borwegian Wood,Haruki Murakami
Middlesex: A Novel (Oprah's Book Club),Jeffrey Eugenides

#------------------------ books_large.csv------------------------------------

title,author
100 Selected Poems,e. e. cummings
100 Years of The Best American Short Stories,Lorrie Moore
"1001 Beds: Performances, Essays, and Travels (Living Out: Gay and Lesbian Autobiog)",Tim Miller
1001 Children's Books You Must Read Before You Grow Up,Julia Eccleshare
101 Famous Poems,Roy Cook
"1020 Haiku in Translation: The Heart of Basho, Buson and Issa",William R. Nelson
11/22/63: A Novel,Stephen King
12 Plays: A Portable Anthology,Janet E. Gardner
1356: A Novel,Bernard Cornwell
187 Reasons Mexicanos Can't Cross the Border: Undocuments 1971-2007,Juan Felipe Herrera
"19 Book Set: The Aubrey Maturin Series - Master and Commander, Post Captain, HMS Surprise, The Mauritius Command, Desolation Island, The Fortune of War, The Surgeon's Mate, The Ionian Mission, Treason's Harbour, The Far Side of the World + 9 More (The Aubrey - Maturin Series Set, Vol. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)",Patrick O'Brian
1956 and All That: The Making of Modern British Drama,Dan Rebellato
1Q84: 3 Volume Boxed Set (Vintage International),Haruki Murakami
2.5 Minute Ride and 101 Most Humiliating Stories,Lisa Kron
"20,000 Leagues Under the Sea (Dramatized)",Jules Verne
2000x: R.U.R. (Dramatized),Karel Capek
21 Speeches That Shaped Our World: The People and Ideas That Changed the Way We Think,Chris Abbott
21: The Final Unfinished Voyage of Jack Aubrey (Vol. Book 21)  (Aubrey/Maturin Novels),Patrick O'Brian
2666 (Spanish Edition),Roberto Bolao
3 - La cura mortal - Maze Runner (Maze Runner Trilogy) (Spanish Edition),James Dashner
30 Ten-Minute Plays from the Actors Theatre of Louisville for 2 Actors,Michael Bigelow Dixon
30 Under 30: An Anthology of Innovative Fiction by Younger Writers,Blake Butler
36 Arguments for the Existence of God: A Work of Fiction (Vintage Contemporaries),Rebecca Goldstein
365 Days / 365 Plays,Suzan-Lori Parks
"4 Book Set : The Aubrey-Maturin Series - Master and Commander, Post Captain, HMS Surprise, The Mauritius Command (The Aubrey-Maturin Series, Vol. 1, 2, 3, 4)",Patrick O'Brian
40 Short Stories: A Portable Anthology,Beverly Lawn
4000 Miles,Amy Herzog
4000 Miles and After the Revolution: Two Plays,Amy Herzog
"44 Scotland Street (44 Scotland Street Series, Book 1)",Alexander McCall Smith
47 Ronin,John Allyn
491 Days: Prisoner Number 1323/69 (Modern African Writing Series),Winnie Madikizela-Mandela
50 Essays: A Portable Anthology,Samuel Cohen
501 Must-Know Speeches,Bounty
51 Shades of Chocolate,Christopher Milano
52 Bible Characters Dramatized: Easy-to-Use Monologues for All Occasions,Kenneth W. Osbeck
60 Seconds to Shine Volume 2: 221 One-minute Monologues For Women,John Capecci
777 And Other Qabalistic Writings of Aleister Crowley: Including Gematria & Sepher Sephiroth,Aleister Crowley
"9/11 Fiction, Empathy, and Otherness",Tim Gauthier
A 30-Minute Summary of Sue Monk Kidd's The Invention of Wings,InstaRead Summaries
A Baby for Christmas (Love at The Crossroads) (Volume 2),Pat Simmons
A Banner of Love,Josephine Garner
A Barthes Reader,Roland Barthes
A Beautiful Place to Die: An Emmanuel Cooper Mystery,Malla Nunn
A Beautiful Wedding: A Novella (Beautiful Disaster Series),Jamie McGuire
A Blaze of Glory: A Novel of the Battle of Shiloh (the Civil War in the West),Jeff Shaara
A Blossom of Bright Light (A Jimmy Vega Mystery),Suzanne Chazin
A Bollywood Affair,Sonali Dev
A Book of Middle English,J. A. Burrow
A Book That Was Lost: Thirty Five Stories (Hebrew Classics),S. Y. Agnon
A Boy and His Soul (Oberon Modern Plays),Colman Domingo
A Battle Won,S. Thomas Russell
A Boy of Good Breeding: A Novel,Miriam Toews
A Brave Man Seven Storeys Tall: A Novel (P.S.),Will Chancellor
A Brief History of Seven Killings: A Novel,Marlon James
A Brilliant Madness,Robert M Drake
A Broken Girl's Journey 4: Kylie's Song (Volume 4),Niki Jilvontae
A Brownsville Tale 2 (Volume 2),Sonovia Alexander
A Carrot a Day: A Daily Dose of Recognition for Your Employees,Adrian Gostick
A Cartoon History of Texas,Patrick M. Reynolds
A Chain of Thunder: A Novel of the Siege of Vicksburg (the Civil War in the West),Jeff Shaara
A Child's Garden of Verses (Large Print Edition),Robert Louis Stevenson
A Child's Portrait of Shakespeare (Shakespeare Can Be Fun series),Lois Burdett
A Christmas Carol,Charles Dickens
A Christmas Carol,Charles Dickens
A Christmas Carol and Other Christmas Books (Oxford World's Classics),Charles Dickens
A Clockwork Orange,Anthony Burgess
A Clockwork Orange Easton Press Leatherbound,Anthony Burgess
A Colebridge Quilted Christmas (Colebridge Community),Hazelwood
A Common Life: The Wedding Story (The Mitford Years #6),Jan Karon
A Confederacy of Dunces,John Kennedy Toole
A Confederacy of Dunces Cookbook: Recipes from Ignatius J. Reilly's New Orleans,Cynthia LeJeune Nobles
A Connecticut Yankee In King Arthur's Court,Mark Twain
A Connecticut Yankee in King Arthur's Court (Dover Thrift Editions),Mark Twain
A Constellation of Vital Phenomena: A Novel,Anthony Marra
"A Corresponding Renaissance: Letters Written by Italian Women, 1375-1650",Lisa Kaborycha
A Critical Companion to Beowulf,Andy Orchard
A Cup of Christmas Tea,Tom Hegg
A Dance of Blades (Shadowdance 2),David Dalglish
A Dance of Mirrors (Shadowdance 3),David Dalglish
A Daughter of Zion (Zion Chronicles),Bodie Thoene
A Day In August,Reis Armstrong
A Death in the Family: A Detective Kubu Mystery,Michael Stanley
A Decent Ride: A Novel,Irvine Welsh
A Deeper Love Inside: The Porsche Santiaga Story,Sister Souljah
"A Diary With Reminiscences Of The War And Refugee Life In The Shenandoah Valley, 1860-1865",Cornelia McDonald
A Dictionary of Critical Theory (Oxford Paperback Reference),Ian Buchanan
A Dictionary of Haiku: Second Edition,Jane Reichhold
A Dictionary of Northern Mythology,Rudolf Simek
A Different Kind of Courage,Sarah Holman
A Dirty Job: A Novel,Christopher Moore
"A Dirty Shame: J.J. Graves Mystery, Book 2",Liliana Hart
A Dog's Purpose,W. Bruce Cameron
A Doll's House (Dover Thrift Editions),Henrik Ibsen
A Dollar Outta Fifteen Cent 3: Mo' Money...Mo' Problems,Caroline McGill
A Dream Play,August Strindberg
A Face in the Crowd,Stephen King
A Faint Cold Fear (Grant County Mysteries),Karin Slaughter
A Family Guide To Narnia: Biblical Truths in C.S. Lewis's The Chronicles of Narnia,Christin Ditchfield
A Farewell To Arms,Ernest Hemingway
A Farewell to Arms: The Hemingway Library Edition,Ernest Hemingway
"A Fatal Grace (Three Pines Mysteries, No. 2)",Louise Penny
A Feast of Snakes: A Novel,Harry Crews
A Free State: A Novel,Tom Piazza
A Funny Thing Happened to Me on My Way Through the Bible: A Collection of Humorous Sketches and Monologues Based on Familiar Bible Stories (Lillenas Drama Resources),Martha Bolton
A Garden of Marvels: Tales of Wonder from Early Medieval China,"Robert, Ford Campany"
A Gentle Creature and Other Stories: White Nights; A Gentle Creature; The Dream of a Ridiculous Man (Oxford World's Classics),Fyodor Dostoevsky
A German General on the Eastern Front: The Letters and Diaries of Gotthard Heinrici 1941-1942,Johanne Hurter
A Glossary of Literary Terms,M.H. Abrams
A Glossary of Literary Terms,M.H. Abrams
A Good Man Is Hard to Find and Other Stories,Flannery O'Connor
A Good Yarn (A Blossom Street Novel),Debbie Macomber
A Grain of Wheat (Penguin African Writers),Ngugi wa Thiong'o
"A Gravity's Rainbow Companion: Sources and Contexts for Pynchon's Novel, 2nd Edition",Steven Weisenburger
A Guide for the Perplexed: A Novel,Dara Horn
A Haiku Garden: Selections from the Everyday Photo Haiku Project,Patrick J Harris
A Haiku Perspective,Annette Rochelle Aben
A Handbook of Critical Approaches to Literature,Wilfred Guerin
A Handbook to Literature (12th Edition),William Harmon
A Handful of Dust,Evelyn Waugh
A Hanging at Cinder Bottom: A Novel,Glenn Taylor
"A Hard, Cruel Shore: An Alan Lewrie Naval Adventure (Alan Lewrie Naval Adventures)",Dewey Lambdin
A Heart's Betrayal (A Journey of the Heart),Colleen Coble
A Heart's Disguise (A Journey of the Heart),Colleen Coble
A Heart's Obsession (A Journey of the Heart),Colleen Coble
A Herzen Reader,Alexander Herzen
A High Wind in Jamaica (New York Review Books Classics),Richard Hughes
A History of Illuminated Manuscripts,Christopher De Hamel
A History of Old Norse Poetry and Poetics,Margaret Clunies Ross
A History of the English Speaking Peoples (4 Volume Set),Winston S. Churchill
A History of the Present Illness: Stories,Louise Aronson
A House Divided (A Reverend Curtis Black Novel),Kimberla Lawson Roby
A House of My Own: Stories from My Life,Sandra Cisneros
A Hustler's Wife (Urban Books),Nikki Turner
"A Jane Austen Education: How Six Novels Taught Me About Love, Friendship, and the Things That Really Matter",William Deresiewicz
"A Jesuit Off-Broadway: Behind the Scenes with Faith, Doubt, Forgiveness, and More",James Martin SJ
A Jewish Spirit in the Wild: Stories of Jewish life in South Africa from the late 19th century to 1979,Abraham Rosen
A Journal of the Plague Year (Dover Thrift Editions),Daniel Defoe
A Journey to the End of the Millennium - A Novel of the Middle Ages,A. B. Yehoshua
A King's Ransom: A Novel,Sharon Kay Penman
A Lesson in Hope: A Novel,Philip Gulley
A King's Trade: An Alan Lewrie Naval Adventure (Alan Lewrie Naval Adventures),Dewey Lambdin
A Kiss Remembered,Sandra Brown
A Legacy (New York Review Books Classics),Sybille Bedford
A Lesson Before Dying (Oprah's Book Club),Ernest J. Gaines
"A Lick of Frost (Meredith Gentry, Book 6)",Laurell K. Hamilton
A Life in Letters (Penguin Classics),Anton Chekhov
A Light in the Wilderness: A Novel,Jane Kirkpatrick
A Lineage of Grace: Five Stories of Unlikely Women Who Changed Eternity,Francine Rivers
A Little Life: A Novel,Hanya Yanagihara
A Little Tour through European Poetry,John Taylor
A Man Called Ove: A Novel,Fredrik Backman
A Man for All Seasons: A Play in Two Acts,Robert Bolt
A Man of the People,Chinua Achebe
A Man Without a Country,Kurt Vonnegut
A Man's Worth (Urban Christian),Nikita Lynnette Nichols
A Manual for Cleaning Women: Selected Stories,Lucia Berlin
A Map of Betrayal: A Novel (Vintage International),Ha Jin
A Matter of Heart (Lone Star Brides) (Volume 3),Tracie Peterson
A Memory of Violets: A Novel of London's Flower Sellers,Hazel Gaynor
A Mercy,Toni Morrison
A Midsummer Night's Dream,William Shakespeare
A Midsummer Night's Dream,William Shakespeare
A Midsummer Night's Dream (Barnes & Noble Shakespeare),William Shakespeare
A Midsummer Night's Dream (Calla Editions),William Shakespeare
A Midsummer Night's Dream (Cambridge School Shakespeare),Rex Gibson
A Midsummer Night's Dream (Folger Shakespeare Library),William Shakespeare
A Midsummer Night's Dream (Shakespeare Made Easy),William Shakespeare
A Midsummer Night's Dream (The Pelican Shakespeare),William Shakespeare
A Midsummer Night's Dream for Kids (Shakespeare Can Be Fun!),Lois Burdett
A Midsummer Night's Dream The Graphic Novel: Original Text (Shakespeare Range),William Shakespeare
A Midsummer Night's Dream: Sixty-Minute Shakespeare Series,Cass Foster
A Midsummer Night's Dream: The Oxford Shakespeare,William Shakespeare
"A Million Guilty Pleasures: Million Dollar Duet, Book 2",C. L. Parker
A Mind at Peace,Ahmet Hamdi Tanpinar
A Mind Awake: An Anthology of C. S. Lewis,C. S. Lewis
A Modern Grammar for Biblical Hebrew Workbook,Duane A. Garrett
"A Modern Witch: A Modern Witch, Book 1",Debora Geary
A Modest Proposal and Other Satirical Works (Dover Thrift Editions),Jonathan Swift
A Moment in Time (Lone Star Brides) (Volume 2),Tracie Peterson
A Month in the Country (New York Review Books Classics),J.L. Carr
A Mother's Kisses: A Novel,Bruce Jay Friedman
A Naked Tree: Love Sonnets to C. S. Lewis and Other Poems,Joy Davidman
"A Narrative of a Revolutionary Soldier: Some Adventures, Dangers, and Sufferings of Joseph Plumb Martin (Signet Classics)",Joseph Plumb Martin
A Nasty Bit of Rough: A Novel,David Feherty
A Number,Caryl Churchill
A Pale View of Hills,Kazuo Ishiguro
A Perfect Crime,A Yi
A Perfect Life: A Novel,Danielle Steel
A Perfect Life: A Novel (Random House Large Print),Danielle Steel
A Philosophical Walking Tour with C.S. Lewis: Why It Did Not Include Rome,Stewart Goetz
A Place Where the Sea Remembers and Related Readings (Literature connections),Sandra Benitez
A Pledge of Silence,Flora J. Solomon
A Poet of the Invisible World: A Novel,Michael Golding
A Poetry Handbook,Mary Oliver
A Portrait of the Artist As a Young Man,James Joyce
A Portrait of the Artist as a Young Man (Penguin Classics),James Joyce
A Possibility of Violence: A Novel,D. A. Mishani
A Prayer for Owen Meany: A Novel,John Irving
A Private Revenge,Richard Woodman
A Promise Kept,Robin Lee Hatcher
A Question of Death: An Illustrated Phryne Fisher Anthology,Kerry Greenwood
"A Question of Mercy: A Play Based on the Essay by Richard Selzer (Rabe, David)",David Rabe
A Question of Proof,Joseph Amiel
A Quilter's Holiday: An Elm Creek Quilts Novel (The Elm Creek Quilts),Jennifer Chiaverini
A Rag Doll's Heart,Maggie Simmons
A Raisin in the Sun,Lorraine Hansberry
A Raisin In The Sun: And Related Readings,Lorraine Hansberry
A Raisin in the Sun: with Connections (HRW Library),Lorraine Hansberry
A Redbird Christmas: A Novel,Fannie Flagg
A Remarkable Kindness: A Novel,Diana Bletter
A Replacement Life: A Novel (P.S.),Boris Fishman
A Reunion of Ghosts: A Novel,Judith Claire Mitchell
A Rhetoric for Writing Teachers,Erika Lindemann
A River Runs Through It,Norman Maclean
"A River Runs Through It and Other Stories, Twenty-fifth Anniversary Edition",Norman Maclean
"A Roman Army Reader: Twenty-One Selections from Literary, Epigraphic, and Other Documents (Bc Latin Readers)",Dexter Hoyos
A Room of One's Own,Virginia Woolf
A Room of One's Own (Annotated),Virginia Woolf
"A Sailor of Austria: In Which, Without Really Intending to, Otto Prohaska Becomes Official War Hero No. 27 of the Habsburg Empire (The Otto Prohaska Novels)",John Biggins
A Sand County Almanac and Sketches Here and There,Aldo Leopold
A Sand County Almanac: And Sketches Here and There (Outdoor Essays & Reflections),Aldo Leopold
A Science Fiction Cookbook: And Guide to Edible Niceties,Nicole Lynn Roach
"A Scots Quair: Sunset Song, Cloud Howe, Grey Granite",Lewis Grassic Gibbon
"A Sea of Words, Third Edition: A Lexicon and Companion to the Complete Seafaring Tales of Patrick O'Brian",Dean King
A Second Daniel (In the Den of the English Lion) (Volume 1),Neal Roberts
A Second Helping: A Blessings Novel (Blessings Series),Beverly Jenkins
A Sentimental Journey: Sentimental Journey: Memoirs 1917-1922 (Russian Literature Series),Viktor Shklovsky
A Separate Peace,John Knowles
A Short History of Ancient Greece (I.B. Tauris Short Histories),P. J. Rhodes
A Short History of Indians in Canada: Stories,Thomas King
A Short History of Tractors in Ukrainian,Marina Lewycka
A Single Man: A Novel (FSG Classics),Christopher Isherwood
A Single Thread (Cobbled Court Quilts),Marie Bostwick
A Skeleton Key to Finnegans Wake: Unlocking James Joyce's Masterwork (The Collected Works of Joseph Campbell),Joseph Campbell
A Sky Without Eagles,Jack Donovan
A Slip of the Keyboard: Collected Nonfiction,Terry Pratchett
A Small Greek World: Networks in the Ancient Mediterranean (Greeks Overseas),Irad Malkin
A Small Story about the Sky,Alberto Ros
A Soft Place to Land: A Novel,Susan Rebecca White
A Soldier's Play (Dramabook),Charles Fuller
"A Solemn Pleasure: To Imagine, Witness, and Write (The Art of the Essay)",Melissa Pritchard
A Southern Girl: A Novel (Story River Books),John Warley
A Spear of Summer Grass,Deanna Raybourn
A Spool of Blue Thread: A novel,Anne Tyler
A Sportsman's Notebook (Everyman's Library),Ivan Turgenev
A Spot of Bother,Mark Haddon
"A Storm of Swords (A Song of Ice and Fire, Book 3)",George R. R. Martin
A Story as Sharp as a Knife: The Classical Haida Mythtellers and Their World (Masterworks of the Classical Haida Mythtellers),Robert Bringhurst
A Strangeness in My Mind: A novel,Orhan Pamuk
A Streetcar Named Desire.,Tennessee Williams
A Suitable Boy: A Novel,Vikram Seth
A Suitable Boy: A Novel (Modern Classics),Vikram Seth
A Tale of three Kings: A Study in Brokenness,Gene Edwards
A Taste for Chaos: The Art of Literary Improvisation,Randy Fertel
A Taste of Honey: Stories,Jabari Asim
A Texas Hill Country Christmas,William W. Johnstone
A Theatre of Envy,Rene Girard
A Thousand Mornings: Poems,Mary Oliver
A Thousand Pardons: A Novel,Jonathan Dee
A Thousand Splendid Suns,Khaled Hosseini
A Thread of Truth (Cobbled Court Quilts),Marie Bostwick
A Thread So Thin (Cobbled Court Quilts),Marie Bostwick
A Timeshare,Margaret Ross
A Town Called Valentine: A Valentine Valley Novel,Emma Cane
A Traveled First Lady: Writings of Louisa Catherine Adams,Louisa Catherine Adams
A Treacherous Paradise,Henning Mankell
"A Treasury of Irish Myth, Legend & Folklore (Fairy and Folk Tales of the Irish Peasantry / Cuchulain of Muirthemne)",Isabella Augusta Gregory
A Treasury of Jewish Folklore,Nathan Ausubel
A Tree Grows in Brooklyn (Modern Classics),Betty Smith
A Trip to the Stars: A Novel,Nicholas Christopher
"A True History of the Three Brave Indian Spies, John Cherry, Andrew and Adam Poe, Who Wiped Out Big Foot and His Two Brothers, Styled Sons of the Half King (Classic Reprint)",A. W. Poe
A Turn in the Road (A Blossom Street Novel),Debbie Macomber
A Vaquero of the Brush Country: The Life and Times of John D. Young,John D. Young
A View From the Bridge.,Arthur Miller
A View From the Pew: A Collection of Sketches and Monologues About Church Life- From the Potluck to the Bored Meeting,Martha Bolton
A Vintage Affair: A Novel (Random House Reader's Circle),Isabel Wolff
A Vision: The Revised 1937 Edition: The Collected Works of W.B. Yeats Volume XIV,William Butler Yeats
A Voice Full of Cities: The Collected Essays of Robert Kelly,Robert Kelly
A Voyage for Madmen,Peter Nichols
A Waka Anthology - Volume Two: Grasses of Remembrance (Parts A & B) (v. 2),Edwin A. Cranston
A Week in Winter,Maeve Binchy
A White Tea Bowl: 100 Haiku from 100 Years of Life,Mitsu Suzuki
A Wife's Betrayal,Miss KP
A Wild Swan: And Other Tales,Michael Cunningham
A Wilder Rose: A Novel,Susan Wittig Albert
A Window Opens: A Novel,Elisabeth Egan
A Winter Dream: A Novel,Richard Paul Evans
A Woman of Substance (Harte Family Saga),Barbara Taylor Bradford
A World Lit Only by Fire: The Medieval Mind and the Renaissance: Portrait of an Age,William Manchester
A Worthy Pursuit,Karen Witemeyer
"A Writer at War: A Soviet Journalist with the Red Army, 1941-1945",Vasily Grossman
A Writer's Diary,Fyodor Dostoevsky
A Year in the Life of William Shakespeare: 1599,James Shapiro
A Year on Ladybug Farm,Donna Ball
A Year with C. S. Lewis: Daily Readings from His Classic Works,C. S. Lewis
A Year with Hafiz: Daily Contemplations,Hafiz
A Year with Rilke: Daily Readings from the Best of Rainer Maria Rilke,Anita Barrows
A Year with Rumi: Daily Readings,Coleman Barks
A Yellow Raft in Blue Water: A Novel,Michael Dorris
A.C. Swinburne and the Singing Word,Yisrael Levin
A.D. 30: A Novel,Ted Dekker
A.D. 33: A Novel,Ted Dekker
Abandoned Secrets,Toni Larue
Abandoned: Three Short Stories,Jim Heskett
Abelard and Heloise: The Letters and Other Writings (Hackett Classics),Abelard
Abigail Adams: Letters: Library of America #275 (The Library of America),Abigail Adams
Abingdon's Speeches &  Recitations for Young Children,Abingdon Press
About Love and Other Stories (Oxford World's Classics),Anton Chekhov
Above the Waterfall: A Novel,Ron Rash
Abraham and Sarah: History's Most Fascinating Story of Faith and Love,J. SerVaas Williams
Abraham Lincoln (Classic Reprint),Wilbur Fisk Gordy
Abraham Lincoln: Great Speeches (Dover Thrift Editions),Abraham Lincoln
Abraham Lincoln: Vampire Hunter,Seth Grahame-Smith
Absurdistan: A Novel,Gary Shteyngart
Abundant Rain (Urban Christian),Vanessa Miller
Across a Hundred Mountains: A Novel,Reyna Grande
Across Five Aprils,Irene Hunt
Acting in Restoration Comedy (Applause Acting Series),Maria Aitken
Acting Scenes & Monologues For Kids!: Original Scenes and Monologues Combined Into One Very Special Book!,Bo Kane
Acting Scenes And Monologues For Young Teens: Original Scenes and Monologues Combined Into One Book,Bo Kane
Acting Up in Church: Humorous Sketches for Worship Services,M. K. Boyle
Acts of Betrayal (Urban Books),Tracie Loveless-Hill
"Acts of Intervention: Performance, Gay Culture, and AIDS (Unnatural Acts: Theorizing the Performative)",David Roman
"Ada, or Ardor: A Family Chronicle",Vladimir Nabokov
Adaptation and Appropriation (The New Critical Idiom),Julie Sanders
ADAPTED CLASSICS MACBETH SE 96C. (Globe Adapted Classics),GLOBE
Addicted to Drama,Pamala G Briscoe
Addiss: Haiga: Takebe Socho Pa,Stephen Addiss
Adios Hemingway,Leonardo Padura Fuentes
Admiral Hornblower in the West Indies (Hornblower Saga),C. S. Forester
Adolf Hitler and His Airship: An Alternate History: Adolf vs. the Canadians  Part 3 of the Hitler Chronicles,Victor Appleton
Adrienne Rich's Poetry and Prose (Norton Critical Editions),Adrienne Rich
Adulterio (Spanish Edition),Paulo Coelho
Adventures of Huckleberry Finn,Mark Twain
Adventures of Huckleberry Finn,Mark Twain
Adventures of Huckleberry Finn (Third Edition)  (Norton Critical Editions),Mark Twain
Adventures of Huckleberry Finn: and Related Readings (Literature Connections),MCDOUGAL LITTEL
Advice to Little Girls,Mark Twain
Aeneid (Hackett Classics),P. Vergilius Maro
Aeneid (Oxford World's Classics),Virgil
Aeschyli Persae (Latin Edition),Aeschylus
Aeschyli persae ad fidem manusciptorum,Aeschylus
Aeschyli Persae: Ad Fidem Manuscriptorum (1826) (Latin Edition),Aeschylus
Aeschyli Tragoediae Quae Supersunt: Persae,Aeschylus
"Aeschylus I: Oresteia: Agamemnon, The Libation Bearers, The Eumenides (The Complete Greek Tragedies) (Vol 1)",Aeschylus
"Aeschylus I: The Persians, The Seven Against Thebes, The Suppliant Maidens, Prometheus Bound (The Complete Greek Tragedies)",Aeschylus
Aeschylus II: The Oresteia (The Complete Greek Tragedies),Aeschylus
"Aeschylus, I, Persians. Seven against Thebes. Suppliants. Prometheus Bound (Loeb Classical Library)",Aeschylus
"Aeschylus, II, Oresteia: Agamemnon. Libation-Bearers. Eumenides (Loeb Classical Library)",Aeschylus
Aeschylus: Agamemnon (Cambridge Translations from Greek Drama),Aeschylus
Aesop's Fables (Word Cloud Classics),Aesop
Aesop's Fables: A Pop-Up Book of Classic Tales,Kees Moerbeek
Aesthetics: Lectures And Essays,Edward Bullough
"Affairytale, A Memoir",C.J. English
Affirming: Letters 1975-1997,Isaiah Berlin
Affrilachia: Poems by Frank X Walker,Frank X Walker
African Ceremonies: The Concise Edition,Carol Beckwith
"African Narratives of Orishas, Spirits and Other Deities - Stories from West Africa and the African Diaspora: A Journey Into the Realm of Deities, SPI",Alex Cuoco
"Afrodita: Cuentos, Recetas y Otros Afrodisiacos",Isabel Allende
After Abel and Other Stories,Michal Lemberger
"After Antiquity: Greek Language, Myth, and Metaphor (Myth and Poetics)",Margaret Alexiou
After Birth,Elisa Albert
After I Do: A Novel,Taylor Jenkins Reid
After Many a Summer Dies the Swan,Aldous Huxley
After the Circus: A Novel (The Margellos World Republic of Letters),Patrick Modiano
After the Parade: A Novel,Lori Ostlund
Again to Carthage: A Novel,John L. Parker Jr.
Against Interpretation: And Other Essays,Susan Sontag
Age of Arousal,Linda Griffiths
Age of Fools,William A Cook
Agnes of God,John Pielmeier
Ahmed the Philosopher: Thirty-Four Short Plays for Children and Everyone Else,Alain Badiou
Aias (Greek Tragedy in New Translations),Sophocles
Aischylos Agamemnon (1885),Aeschylus
Ajax: With Notes Critical and Explanatory,Sophocles
"Akiane: Her Life, Her Art, Her Poetry",Akiane Kramarik
Ak: The Years of Childhood,Wole Soyinka
Albion's Seed: Four British Folkways in America (America: a cultural history),David Hackett Fischer
Alburquerque: A Novel,Rudolfo Anaya
Alcestis,Euripides Euripides
Alcestis,"John Hampden Haydon, Amy M, Euripides, William Sheldon Hadley"
Alcestis,Richard (trans) EURIPIDES / ALDINGTON
Aleph (Espaol) (Spanish Edition),Paulo Coelho
Aleph (Vintage International),Paulo Coelho
Alex Cross's Trial,James Patterson
Alexandrian Summer,Yitzhak Gormezano Goren
Alibi: A Novel,Joseph Kanon
Alice I Have Been: A Novel (Random House Reader's Circle),Melanie Benjamin
Alice in Plunderland,Steve McCaffery
Alice in Tumblr-land: And Other Fairy Tales for a New Generation,Tim Manley
Alice in Wonderland (Third Edition)  (Norton Critical Editions),Lewis Carroll
Alice's Adventures in Wonderland,Lewis Carroll
Alice's Adventures in Wonderland & Through the Looking-Glass (Bantam Classics),Lewis Carroll
Alice's Adventures in Wonderland and Through the Looking-Glass (Penguin Classics),Lewis Carroll
Alice's Adventures in Wonderland and Through the Looking-Glass: 150th-Anniversary Edition (Penguin Classics Deluxe Edition),Lewis Carroll
Alice's Adventures in Wonderland Decoded: The Full Text of Lewis Carroll's Novel with its Many Hidden Meanings Revealed,David Day
"Alinor, la Reine aux deux couronnes (French Edition)",Eric Brown
All Art Is Propaganda,George Orwell
All Backs Were Turned (Rebel Lit),Marek Hlasko
All Fall Down: A Novel,Jennifer Weiner
All He Ever Desired (The Kowalskis),Shannon Stacey
All I Love and Know: A Novel,Judith Frank
"All in the Timing, Six One-Act Comedies - Acting Edition",David Ives
All in the Timing: Fourteen Plays,David Ives
All Involved: A Novel,Ryan Gattis
All My Friends Are Still Dead,Jory John
All My Puny Sorrows,Miriam Toews
All My Sons (Penguin Classics),Arthur Miller
All Other Nights: A Novel,Dara Horn
All Our Happy Days Are Stupid,Sheila Heti
All Our Names,Dinaw Mengestu
All Quiet on the Western Front,Erich Maria Remarque
All She Ever Wanted,Lynn Austin
All That Followed: A Novel,Gabriel Urza
All That Glitters,Thomas Tryon
All That Is: A Novel (Vintage International),James Salter
"All That You've Seen Here Is God: New Versions of Four Greek Tragedies Sophocles' Ajax, Philoctetes, Women of Trachis; Aeschylus' Prometheus Bound (Vintage Original)",Sophocles
All the Best: My Life in Letters and Other Writings,George H.W. Bush
All the Brave Fellows (Revolution at Sea Saga #5),James L. Nelson
All the Greek Verbs (Greek Language),N. Marinone
"All the Pretty Horses (The Border Trilogy, Book 1)",Cormac McCarthy
All the Rage: A Quest,Martin Moran
All the Single Ladies: A Novel,Dorothea Benton Frank
All the Stars in the Heavens: A Novel,Adriana Trigiani
"All The Wild That Remains: Edward Abbey, Wallace Stegner, and the American West",David Gessner
All the World,Liz Garton Scanlon
All Things New,Lynn Austin
All Year Long!: Funny Readers Theatre for Life's Special Times,Diana R. Jenkins
All's Well that Ends Well: The Oxford Shakespeare (Oxford World's Classics),William Shakespeare
Allen and Greenough's New Latin Grammar (Dover Language Guides),James B Greenough
Allusion and Intertext: Dynamics of Appropriation in Roman Poetry (Roman Literature and its Contexts),Stephen Hinds
Almanac of the Dead,Leslie Marmon Silko
Almost Famous Women: Stories,Megan Mayhew Bergman
"Almost, Maine",John Cariani
Alone with the Horrors: The Great Short Fiction of Ramsey Campbell 1961-1991,Ramsey Campbell
Already Home,Susan Mallery
Altar of Eden,James Rollins
Altneuland: The Old-New-Land,Theodor Herzl
Always Coca-Cola,Alexandra Chreiteh
Ama: A Story of the Atlantic Slave Trade,Manu Herbstein
Ambition and Survival: Becoming a Poet,Christian Wiman
America Deceived III: Department of Homeland Security Warning: Possession of this novel may result in a targeted drone strike,E.A. Blayre III
American Dervish: A Novel,Ayad Akhtar
American Elsewhere,Robert Jackson Bennett
American Hero,Bess Wohl
"American Indian Stories, Legends, and Other Writings (Penguin Classics)",Zitkala-Sa
American Indian Trickster Tales (Myths and Legends),Richard Erdoes
American Literature (EZ-101 Study Keys),Francis E. Skipp
American Primitive,Mary Oliver
American Psycho,Bret Easton Ellis
Americanah,Chimamanda Ngozi Adichie
Americanah,Chimamanda Ngozi Adichie
"Amerika: The Missing Person: A New Translation, Based on the Restored Text (The Schocken Kafka Library)",Franz Kafka
Amistad Mountain & Other Stories,Sigifredo Cavazos
Among the Gods (Chronicles of the Kings #5) (Volume 5),Lynn Austin
Among the Ten Thousand Things: A Novel,Julia Pierpont
Amoroso (Alfonzo) (Volume 16),S.W. Frank
Amulet,Roberto Bolao
Amuse Bouche (Russell Quant Mysteries),Anthony Bidulka
Amy Falls Down: A Novel,Jincy Willett
"Amy Lynn, The Lady Of Castle Dunn (Volume 3)",Jack July
An (Im)possible Life: Poesia y Testimonio in the Borderlands,Elvira Prieto
An Absent Mind,Eric Rill
An Anthropologist On Mars: Seven Paradoxical Tales,Oliver Sacks
An Apple a Day ...,Julian Gough
An Echo in the Bone: A Novel (Outlander),Diana Gabaldon
An Ecology of World Literature: From Antiquity to the Present Day,Alexander Beecroft
An Elephant in the Garden,Michael Morpurgo
An Elm Creek Quilts Sampler: The First Three Novels in the Popular Series (The Elm Creek Quilts),Jennifer Chiaverini
An English Ghost Story,Kim Newman
An Enticing Misconception: A Book of Prose and Poetry,Ruv Burns
An Equal Music: A Novel,Vikram Seth
An Ideal Husband (Dover Thrift Editions),Oscar Wilde
An Iliad,Lisa Peterson
An Iliad,Lisa Peterson
An Improbable Truth: The Paranormal Adventures of Sherlock Holmes,A.C. Thompson
An Inspector Calls (Philip Allan Literature Guide for Gcse),J. B. Priestley
An Introduction to Literature (16th Edition),Sylvan Barnet
An Introduction to Poetry (13th Edition),X. J. Kennedy
An Irish Country Christmas (Irish Country Books),Patrick Taylor
An Irish Country Courtship: A Novel (Irish Country Books),Patrick Taylor
An Irish Country Village (Irish Country Books),Patrick Taylor
An Irish Country Wedding: A Novel (Irish Country Books),Patrick Taylor
An Irish Doctor in Love and at Sea: An Irish Country Novel (Irish Country Books),Patrick Taylor
An Irish Doctor in Peace and at War: An Irish Country Novel (Irish Country Books),Patrick Taylor
An Obedient Father,Akhil Sharma
An Occult Physiology: Eight Lectures By Rudolf Steiner,Rudolf Steiner
An Octoroon,Branden Jacobs-jenkins
An Oresteia: Agamemnon by Aiskhylos; Elektra by Sophokles; Orestes by Euripides,Aeschylus
An Untamed Heart,Lauraine Snelling
An Untamed Land (Red River of the North #1),Lauraine Snelling
Ana of California: A Novel,Andi Teran
Anacalypsis - The Saitic Isis: Languages Nations and Religions (v. 1 & 2),Godfrey Higgins
Anadarko: A Kiowa Country Mystery,Tom Holm
"Analogies at War: Korea, Munich, Dien Bien Phu, and the Vietnam Decisions of 1965",Yuen Foong Khong
Anansi Boys,Neil Gaiman
Ancient Blood: A Navajo Nation Mystery (Volume 3),R. Allen Chappell
Ancient Egyptian Literature: Volume I: The Old and Middle Kingdoms,Miriam Lichtheim
Ancient Near Eastern Thought and the Old Testament: Introducing the Conceptual World of the Hebrew Bible,John H. Walton
Ancient Sorceries and Other Weird Stories (Penguin Classics),Algernon Blackwood
And Eternity (Book Seven of Incarnations of Immortality),Piers Anthony
And Still I Rise,Maya Angelou
And the Birds Rained Down,Jocelyne Saucier
And West Is West,Ron Childress
And Yet...: Essays,Christopher Hitchens
And You Call Yourself a Christian (Urban Books),E.N. Joy
Andersen's Fairy Tales,Hans Christian Andersen
Andrew's Brain: A Novel,E.L. Doctorow
Android Karenina (Quirk Classic),Leo Tolstoy
Anecdotal Shakespeare: A New Performance History,Paul Menzer
Angel of Storms (Millennium's Rule),Trudi Canavan
Angelmaker (Vintage Contemporaries),Nick Harkaway
Angelopolis: A Novel (Angelology Series),Danielle Trussoni
"Angels in America, Part One: Millennium Approaches",Tony Kushner
"Angels in America, Part Two: Perestroika",Tony Kushner
Angels Make Their Hope Here,Breena Clarke
Anglo-Saxon Community in J.R.R. Tolkien's the Lord of the Rings,Deborah a. Higgens
Animal,K'wan
Animal 2: The Omen,K'wan
Animal 3: Revelations,K'wan
Animal Farm,George Orwell
Animal Farm and Related Readings,George Orwell
Animal Farm: Centennial Edition,George Orwell
Animals,Emma Jane Unsworth
Anna Karenina (Modern Library Classics),Leo Tolstoy
Anna Karenina (Oxford World's Classics),Leo Tolstoy
"Anne of Avonlea, Large-Print Edition",Lucy Maud Montgomery
Anne of Avonlea: Anne of Green Gables Part 2,Lucy Maud Montgomery
Annie Freeman's Fabulous Traveling Funeral,Kris Radish
Annie John: A Novel,Jamaica Kincaid
Anno Dracula,Kim Newman
Anno Dracula: Johnny Alucard,Kim Newman
Anno Dracula: The Bloody Red Baron,Kim Newman
Anointed,Patricia Haley
Another,Yukito Ayatsuji
Another Country,James Baldwin
Another Day,David Levithan
Another Man's Moccasins: A Walt Longmire Mystery (A Longmire Mystery),Craig Johnson
Another Piece of My Heart,Jane Green
Another Woman's Daughter (U.S edition),Fiona Sussman
Anselm of Canterbury: The Major Works (Oxford World's Classics),St. Anselm
Anthills of the Savannah,Chinua Achebe
Anthology Of Classical Myth: Primary Sources in Translation,Stephen Trzaskoma
Anthology of Living Theater,Edwin Wilson
Anthroposophy in the Light of Goeth's Faust: Writings and Lectures from Mid-1880s to 1916: of Spiritual-Scientific Commentaries on Goethe's Faust (The Collected Works of Rudolf Steiner),Rudolf Steiner
Antigone,Jean Anouilh
Antigone,Sophocles
Antigone - Activity Pack,Sophocles
Antigone (Dover Thrift Editions),Sophocles
Antigone (Greek Tragedy in New Translations),Sophocles
Antigone (Hackett Classics),Sophocles
"Antigone (Methuen Drama, Methuen Student Edition)",Jean Anouilh
"Antigone, Oedipus the King, Electra (Oxford World's Classics)",Sophocles
Antigone: A New Translation,Sophocles
Antigone: In Plain and Simple English,Sophocles
Antigonick (New Directions Paperbook),Anne Carson
Antologa (Poesia) (Spanish Edition),Len De Greiff
"Anton Chekhov Early Short Stories, 1883-1888 (Modern Library)",Anton Chekhov
"Anton Chekhov Later Short Stories, 1888-1903 (Modern Library)",Anton Chekhov
Anton Chekhov: A Life,Donald Rayfield
Anton Chekhov's Selected Plays (Norton Critical Editions),Anton Chekhov
Anton Chekhov's Short Stories (Norton Critical Editions),Anton Chekhov
Antonio's Will,Yasmin Tirado-Chiodini
Antony & Cleopatra (No Fear Shakespeare),SparkNotes
Antony and Cleopatra (Folger Shakespeare Library),William Shakespeare
Antony and Cleopatra: Oxford School Shakespeare (Oxford School Shakespeare Series),William Shakespeare
Any Other Name: A Longmire Mystery,Craig Johnson
Anything Goes,John Barrowman
Apart at the Seams (Cobbled Court Quilts),Marie Bostwick
Apathy and Other Small Victories,Paul Neilan
Aphrodite: A Memoir of the Senses,Isabel Allende
Apocalypse Cow,Michael Logan
"Apollodorus:  The Library, Volume I: Books 1-3.9 (Loeb Classical Library no. 121)",Apollodorus
"Apollodorus: The Library, Vol. 2: Book 3.10-16 / Epitome (Loeb Classical Library, No. 122) (Volume II)",Apollodorus
"Applause First Folio of Shakespeare in Modern Type: Comedies, Histories & Tragedies (Applause First Folio Editions)",William Shakespeare
Application for Release from the Dream: Poems,Tony Hoagland
Approaching Hysteria,Mark S. Micale
Appropriate and Other Plays,Branden Jacobs-Jenkins
April Morning,Howard Fast
"Apuleius: Metamorphoses (The Golden Ass), Volume II, Books 7-11 (Loeb Classical Library No. 453)",Apuleius
Arabian Love Poems: Full Arabic and English Texts (Three Continents Press),Nizar Qabbani
Aratus: Phaenomena (Cambridge Classical Texts and Commentaries),Aratus
Arcadia: A Play,Tom Stoppard
Argonautika: The Voyage of Jason and the Argonauts,Mary Zimmerman
Arguably: Essays by Christopher Hitchens,Christopher Hitchens
Arguing about Literature: A Guide and Reader,John Schilb
"Aristophanes 1: Clouds, Wasps, Birds (Hackett Classics)",Aristophanes
Aristophanes Clouds,Aristophanes
Aristophanes: Lysistrata (Focus Classical Library),Aristophanes
Aristophanes' Clouds,Aristophanes
Aristotle on Comedy: Towards a Reconstruction of Poetics II,Richard Janko
Aristotle on Life and Death,R.A.H. King
Aristotle On Poetics,Aristotle
Ark: A Dane Maddock Adventure (Dane Maddock Adventures) (Volume 7),David Wood
Armageddon: A Novel of Berlin,Leon Uris
Around the World in 80 Days,Jules Verne
Around the World in Eighty Days (Dover Thrift Editions),Jules Verne
Arrow of God,Chinua Achebe
Arsenic and Old Lace - Acting Edition,Joseph Kesselring
Art and Myth in Ancient Greece (World of Art),Thomas H. Carpenter
Arthur Miller: Collected Plays 1944-1961 (Library of America),Arthur Miller
Arthur Rimbaud: Complete Works,Arthur Rimbaud
Articulos Desarticulados Y Cuentos No Contados (Spanish Edition),Leonardo Guzman
As A Driven Leaf,Milton Steinberg
"As Always, Julia: The Letters of Julia Child and Avis DeVoto",Joan Reardon
"As Bill Sees It (The A.A. Way of Life, Selected writings of AA's co-founder (LARGE PRINT)) (The A.A. Way of Life, Selected writings of AA's co-founder (LARGE PRINT))",Bill Willson
"As Consciousness Is Harnessed to Flesh: Journals and Notebooks, 1964-1980",Susan Sontag
As I Lay Dying: The Corrected Text,William Faulkner
As You Like It (Arden Shakespeare: Third Series),William Shakespeare
As You Like It (Dover Thrift Editions),William Shakespeare
As You Like It (Folger Shakespeare Library),William Shakespeare
As You Like It (No Fear Shakespeare),SparkNotes
As You Like It (Norton Critical Editions),William Shakespeare
As You Like It (Saddleback's Illustrated Classics),William Shakespeare
Asclepius: The Perfect Discourse of Hermes Trismegistus,Clement Salaman
Ash,Malinda Lo
Ashes and Seeds,Michelle Greenblatt
Asimov's Guide to Shakespeare: A Guide to Understanding and Enjoying the Works of Shakespeare,Isaac Asimov
Asking for Trouble: A Novel,Elizabeth Young
Asking the Right Questions (11th Edition),M. Neil Browne
Assassins (Violators: The Coalition) (Volume 2),Nancy Brooks
Associated Press Stylebook 2015 and Briefing on Media Law,The Associated Press
Astonish Me (Vintage Contemporaries),Maggie Shipstead
Astor Place Vintage: A Novel,Stephanie Lehmann
Astray,Emma Donoghue
Astrid and Veronika,Linda Olsson
At Blackwater Pond: Mary Oliver reads Mary Oliver,Mary Oliver
"At Grave's End: Night Huntress, Book 3",Jeaniene Frost
"At Home in Mitford (The Mitford Years, Book 1)",Jan Karon
At Home on Ladybug Farm,Donna Ball
At Home with Jane Austen,Kim Wilson
At Least Once: A Novel (While We Wait) (Volume 1),D. M. Cuffie
At Swim-Two-Birds (Irish Literature),Flann O'Brien
At the Water's Edge: A Novel,Sara Gruen
Atlantis and Lemuria,Rudolf Steiner
Atlas Shrugged,Ayn Rand
Atomic Robo: The Everything Explodes Collection,Brian Clevinger
"Atys, Tragedie En Musique, Ornee D'Entrees de Ballet, de Machines, de Changemens de Theatre (Litterature) (French Edition)",Philippe Quinault
Auden: Poems (Everyman's Library Pocket Poets),W. H. Auden
August Wilson Century Cycle,August Wilson
August: Osage County - Acting Edition,Tracy Letts
Augustus (New York Review Books Classics),John Williams
Aunt Julia and the Scriptwriter: A Novel,Mario Vargas Llosa
Auntie Mame - Acting Edition,Jerome Lawrence and Robert E. Lee
Austerlitz (Modern Library Paperbacks),W.G. Sebald
Autobiography of a Corpse (New York Review Books Classics),Sigizmund Krzhizhanovsky
"Autobiography of Mark Twain, Volume 2: The Complete and Authoritative Edition (Mark Twain Papers)",Mark Twain
"Autobiography of Mark Twain, Volume 3: The Complete and Authoritative Edition (Mark Twain Papers)",Mark Twain
"Autobiography of Mark Twain: The Complete and Authoritative Edition, Vol. 1",Mark Twain
Autobiography of Red,Anne Carson
Autumn Brides: A Year of Weddings Novella Collection,Kathryn Springer
Autumn: Aftermath (Autumn series 5),David Moody
Avenue of Mysteries,John Irving
Averno: Poems,Louise Glck
Awakening Osiris: The Egyptian Book of the Dead,Normandi Ellis
Away: A Novel,Amy Bloom
Aztec and Maya Myths (Legendary Past),Karl Taube
B,Sarah Kay
B-More Careful: A Novel,Shannon Holmes
"Baby, You're the Best (The Crystal Series)",Mary B. Morrison
Bacchae,Euripides
Bacchae,Euripides
Bacchae,Euripides
Bacchae (Dover Thrift Editions),Euripides
Bacchae (Focus Classical Library),Euripides
Bacchae and Other Plays: Iphigenia among the Taurians; Bacchae; Iphigenia at Aulis; Rhesus (Oxford World's Classics),Euripides
Back Channel,Stephen L. Carter
Back Roads to Far Towns: Basho's Oku-No-Hosomichi (Ecco Travels),Basho Matsuo
"Backpack Literature: An Introduction to Fiction, Poetry, Drama, and Writing (5th Edition)",X. J. Kennedy
Bad Apple (The Baddest Chick) Part 1,Nisa Santiago
Bad Bitch (Bitch Series),Joy Deja King
Bad Blood,Mary Monroe
Bad Boys Do (The Donovan Family),Victoria Dahl
Bad Jews,Joshua Harmon
Bad Men: A Thriller,John Connolly
Bad Romeo (The Starcrossed Series),Leisa Rayven
Badenheim 1939,Aharon Appelfeld
Baking Cakes in Kigali: A Novel,Gaile Parkin
Bakkhai (Greek Tragedy in New Translations),Euripides
Balls,Julian Tepper
Balm: A Novel,Dolen Perkins-Valdez
Balseros (Spanish Edition),Ernesto Ochoa
Balzac and the Little Chinese Seamstress: A Novel,Dai Sijie
Band of Sisters,Cathy Gohlke
Bang the Drum Slowly (Second Edition),Mark Harris
Banker,Dick Francis
Banquet for the Damned,Adam Nevill
Barbara the Slut and Other People,Lauren Holmes
Barefoot Heart: Stories of a Migrant Child,Elva Trevino Hart
Barefoot: A Novel,Elin Hilderbrand
Barely a Lady (The Drake's Rakes series),Eileen Dreyer
Barney's Version (Vintage International),Mordecai Richler
Barometer Rising,Hugh Maclennan
"Bartleby, The Scrivener A Story of Wall-Street",Herman Melville
Baseball Dads,Matthew S. Hiley
Basho and His Interpreters: Selected Hokku with Commentary,Basho Matsuo
Basho: The Complete Haiku,Matsuo Basho
Basho's Narrow Road: Spring and Autumn Passages (Rock Spring Collection of Japanese Literature),Matsuo Basho
Bastard Out of Carolina: A Novel,Dorothy Allison
Bastards of the Reagan Era (Stahlecker Selections),Reginald Dwayne Betts
Bathed in Blood (Rogue Angel),Alex Archer
Battle Cry,Leon Uris
Battleborn: Stories,Claire Vaye Watkins
Battlemage (Age of Darkness),Stephen Aryan
Be Careful What You Pray For: A Novel (The Reverend Curtis Black Series),Kimberla Lawson Roby
Be Careful What You Wish For (The Clifton Chronicles),Jeffrey Archer
"Be Still My Vampire Heart (Love at Stake, Book 3)",Kerrelyn Sparks
Beasts of No Nation: A Novel,Uzodinma Iweala
Beasts of No Nation: A Novel (P.S.),Uzodinma Iweala
Beatrix Potter the Complete Tales (Peter Rabbit),Beatrix Potter
Beatrix Potter's Gardening Life: The Plants and Places That Inspired the Classic Children's Tales,Marta McDowell
Beautiful Chaos,Robert M. Drake
Beautiful Chaos,Robert M. Drake
Beautiful Day: A Novel,Elin Hilderbrand
Beautiful Oblivion,Addison Moore
Beautiful Thing - Acting Edition,Jonathan Harvey
Bed of Roses: Bride Quartet,Nora Roberts
"Bedford Introduction to Literature: Reading, Thinking, Writing",Michael Meyer
Before The Muses: An Anthology Of Akkadian Literature,Benjamin R. Foster
Before We Set Sail,Chika Ezeanya
Behind the Beauty 3,YaYa Grant
Behold the Man (The Jerusalem Chronicles),Bodie and Brock Thoene
"Bell, Book and Candle: A Comedy in Three Acts",John van Druten
Bella Fortuna,Rosanna Chiofalo
Bella Poldark,Winston Graham
Belleville,Amy Herzog
Bellman & Black: A Novel,Diane Setterfield
Bellocq's Ophelia: Poems,Natasha Trethewey
Beloved,Toni Morrison
Below Zero (A Joe Pickett Novel),C. J. Box
"Ben-Hur: A Tale of the Christ, Complete and Unabridged",Lew Wallace
Bendigo Shafter: A Novel,Louis L'Amour
Beneath Still Waters (Rogue Angel),Alex Archer
Beneath the Dark Ice,Greig Beck
BENITA: prey for him,Virginia Tranel
Beowulf - Autotypes Of The Unique Cotton Manuscript Vitellius A XV In The British Museum,Julius Zupitza
Beowulf (Broadview Literary Texts) (Broadview Literary Texts Series),Roy Liuzza
Beowulf (Signet Classics),Anonymous
Beowulf: A Dual-Language Edition,Howell D. Chickering
Beowulf: A New Telling,Robert Nye
Beowulf: A New Translation,Seamus Heaney
Beowulf: A Prose Translation (Penguin Classics)paperback,Anonymous
Beowulf: A Translation and Commentary,J.R.R. Tolkien
Berlin Diary,William L. Shirer
Bertie's Guide to Life and Mothers (44 Scotland Street Series),Alexander McCall Smith
Beside a Burning Sea,John Shors
Best Food Writing 2015,Holly Hughes
Best Friends Forever,Kimberla Lawson Roby
"Best Ghost Stories of Algernon Blackwood (Dover Mystery, Detective, & Other Fiction)",Algernon Blackwood
Best Ghost Stories of J. S. LeFanu,J. Sheridan Le Fanu
Best Loved Poems of the American People,Hazel Felleman
Best Russian Short Stories,Various
Best Served Cold,Joe Abercrombie
Best Tales of the Yukon,Robert Service
Best-Loved Folktales of the World (The Anchor folktale library),Joanna Cole
"Bestiary: Being an English Version of the Bodleian Library, Oxford, MS Bodley 764",Richard Barber
Betrayal,Harold Pinter
Betrayal (Exposed Series),Naomi Chase
Betsey Brown: A Novel,Ntozake Shange
Better Off Without Him,Dee Ernst
Better Than Chocolate (Life in Icicle Falls),Sheila Roberts
Between Distant Modernities: Performing Exceptionality in Francoist Spain and the Jim Crow South,Brittany Powell Kennedy
Between Friends,Amos Oz
"Between Heaven and Texas (A Too Much, Texas Novel)",Marie Bostwick
"Between Levinas and Lacan: Self, Other, Ethics",Mari Ruti
"Between Parentheses: Essays, Articles and Speeches, 1998-2003",Roberto Bolano
Between Riverside and Crazy,Stephen Adly Guirgis
Between Sisters,Kristin Hannah
Between Sisters: A Novel (Random House Reader's Circle),Kristin Hannah
Between the Bridge and the River,Craig Ferguson
Between the World and Me,Ta-Nehisi Coates
Beyond Black: A Novel,Hilary Mantel
Beyond Bolao: The Global Latin American Novel (Literature Now),Hctor Hoyos
Beyond the Great Snow Mountains: Stories,Louis L'Amour
Big Cherry Holler: A Novel (Ballantine Reader's Circle),Adriana Trigiani
Big Game Fishing Journal,Speedy Publishing LLC
Big Girls Don't Cry,Brenda Novak
Big Sky Wedding,Linda Lael Miller
Big Stone Gap (Movie Tie-in Edition): A Novel,Adriana Trigiani
Billy Lynn's Long Halftime Walk,Ben Fountain
Biloxi Blues,Neil Simon
Binocular Vision: New & Selected Stories,Edith Pearlman
Birds and Other Plays (Oxford World's Classics),Aristophanes
Birds of Paradise Lost,Andrew Lam
Birthday Letters: Poems,Ted Hughes
Bitch A New Beginning (Bitch Series),Joy Deja King
Bitter Creek (The Montana Mysteries Featuring Gabriel Du Pr),Peter Bowen
Bitter Eden: A Novel,Tatamkhulu Afrika
Bittersweet Dreams (Forbidden),V.C. Andrews
Bittersweet: A Novel,Colleen McCullough
Black ButterFly,Robert M. Drake
Black Cat Journal: 160 Page Lined Journal/Notebook,Mahtava Journals
Black Culture and Black Consciousness: Afro-American Folk Thought from Slavery to Freedom,Lawrence W. Levine
Black Diamond,Brittani Williams
Black Diamond 2: Nicety,Brittani Williams
Black Gangster,Donald Goines
Black Girl Lost,Donald Goines
"Black Men, Obsolete, Single, Dangerous?: The Afrikan American Family in Transition",Haki R Madhubuti
Black Ops (Presidential Agent),W.E.B. Griffin
Black Scarface (Volume 1),Jimmy DaSaint
Black Scarface 3 The Wrath of Face,Jimmy DaSaint
Black Scarface II The Rise of an American Kingpin,Jimmy DaSaint
Black Scarface IV: Live A King...Die A Legend (Volume 4),Jimmy DaSaint
Black Swan Green,David Mitchell
Black Tuesday (Area 51: Time Patrol),Bob Mayer
Blackberry Days of Summer: A Novel (Zane Presents),Ruth P. Watson
Blake and Tradition,Kathleen Raine
Blake or The Huts of America,Martin R. Delany
Bleachers: A Novel,John Grisham
Bleeding Edge: A Novel,Thomas Pynchon
"Bless Me, Ultima",Rudolfo Anaya
Blessing the Boats: New and Selected Poems 1988-2000 (American Poets Continuum),Lucille Clifton
Blind Descent (Anna Pigeon),Nevada Barr
Blind Faith,Ben Elton
Blind Your Ponies,Stanley Gordon West
Blindsided (Sisterhood),Fern Michaels
Bliss,Olivier Choinire
Bliss House: A Novel,Laura Benedict
Bliss: A Novel,Shay Mitchell
"Blithe Spirit, Hay Fever, Private Lives: Three Plays",Noel Coward
Block Party 5k1: Diplomatic Immunity (Volume 1),Al-Saadiq Banks
Block Party 5k1: Diplomatic Immunity (Volume 2),Al-Saadiq Banks
Blockade Billy,Stephen King
Blonde Eskimo: A Novel,Kristen Hunt
"Blonde Hair, Blue Eyes",Karin Slaughter
Blood and Beauty: The Borgias; A Novel,Sarah Dunant
"Blood and Steel 2: The Wehrmacht Archive - Retreat to the Reich, September to December 1944",Donald E. Graves
Blood Girls (Nunatak Fiction),Meira Cook
Blood Harvest,Sharon Bolton
Blood Knot and Other Plays,Athol Fugard
Blood Meridian: Or the Evening Redness in the West,Cormac McCarthy
Blood of Elves,Andrzej Sapkowski
Blood of My Brother II: The Face Off,Zoe Woods
Blood on the church house steps,Avery Bond
Blood on the Forge (New York Review Books Classics),William Attaway
Blood Orchid (Holly Barker),Stuart Woods
Blood Red Tide (Deathlands),James Axler
"Blood Shadows: Blackthorn, Book 1",Lindsay J. Pryor
Bloodletting and Miraculous Cures: Stories,Vincent Lam
Bloodshed of the Mountain Man,William W. Johnstone
Bloodstream,Tess Gerritsen
Blow-Up: And Other Stories,Julio Cortazar
"Blowhard: A Steampunk Fairy Tale: The Clockwork Republic Series, Volume 1",Katina French
Blue at the Mizzen (Vol. Book 20)  (Aubrey/Maturin Novels),Patrick O'Brian
"Blue Gold: A Kurt Austin Adventure (A Novel from the NUMA Files, Book 2)",Paul Kemprecos
Blue Horses: Poems,Mary Oliver
Blue Iris: Poems and Essays,Mary Oliver
Blue Jeans and Coffee Beans,Joanne DeMaio
Blue Lines Up In Arms,James Craig Atchison
"Blue Shoes and Happiness (No. 1 Ladies Detective Agency, Book 7)",Alexander McCall Smith
Bluebeard: A Novel (Delta Fiction),Kurt Vonnegut
Bluets,Maggie Nelson
Bodega Dreams,Ernesto Quinonez
Body Awareness,Annie Baker
Body Surfing: A Novel,Anita Shreve
"Bolder Than Bus: Original Poems, Art, Photography, and Music from the Furthur 50th Anniversary Tour to the Undiscovered Planet",Jessica Charnley
Bone,Fae Myenne Ng
Bone Black: Memories of Girlhood,bell hooks
Bones of Home and Other Plays,Charlene A. Donaghy
Boo (Vintage Contemporaries),Neil Smith
"Book Lust To Go: Recommended Reading for Travelers, Vagabonds, and Dreamers",Nancy Pearl
"Book Lust: Recommended Reading for Every Mood, Moment, and Reason",Nancy Pearl
"Book of Haikus (Poets, Penguin)",Jack Kerouac
Book of Hours,Kevin Young
Book of the Hopi,Frank Waters
BOOK REVIEW: All the Light We Cannot See,J.T. Salrich
Books I've Read: Books I Want to Read,Annabel Fraser
"Books of Blood, Vols. 1-3",Clive Barker
"Books of Blood, Vols. 4-6 (v. 2)",Clive Barker
Bootycandy,Robert O'Hara
"Booze, Broads, and Blackjack: A Deadly Combination",Carl Nicita
Borderlands / La Frontera: The New Mestiza,Gloria Anzalda
Borges: Selected Poems,Jorge Luis Borges
Boricuas: Influential Puerto Rican Writings - An Anthology,Roberto Santiago
Born in Fire: Irish Born Trilogy,Nora Roberts
Boss Bitch (Bitch Series),Joy Deja King
Boss Divas,De'nesha Diamond
Boy Who Fell into a Book,Alan Ayckbourn
Boy with Thorn (Pitt Poetry Series),Rickey Laurentiis
"Boy, Snow, Bird: A Novel",Helen Oyeyemi
"Boy, Snow, Bird: A Novel",Helen Oyeyemi
Boy's Life,Robert McCammon
Boyhood: Scenes From Provincial Life,J. M. Coetzee
Boys in the Band,Mart Crowley
Bradbury Stories: 100 of His Most Celebrated Tales,Ray Bradbury
Braided Creek: A Conversation in Poetry,Jim Harrison
Bram Stoker - Dracula (Readers' Guides to Essential Criticism),William Hughes
Bratya Karamazovy - EEEE EE (Russian Edition),Fyodor Dostoevsky
Brave Enough,Cheryl Strayed
Brave New World and Brave New World Revisited,Aldous Huxley
Bread Alone: A Novel,Judith Ryan Hendricks
Breakdown on Bethlehem Street: A Christmas Play,Frank Ramirez
Breakfast at Tiffany's and Three Stories,Truman Capote
Breaking into Japanese Literature: Seven Modern Classics in Parallel Text,Giles Murray
"Breaking News (Godmothers, Book 5) (The Godmothers)",Fern Michaels
Bream Gives Me Hiccups,Jesse Eisenberg
Breath: A Novel,Tim Winton
"Brewer's Dictionary of Phrase and Fable, Seventeenth Edition",John Ayto
"Brian Friel: Plays 2: Dancing at Lughnasa, Fathers and Sons, Making History, Wonderful Tennessee and Molly Sweeney (Contemporary Classics (Faber & Faber)) (v. 2)",Brian Friel
Brick Shakespeare: Four Tragedies & Four Comedies,John McCann
Breathers: A Zombie's Lament,S.G. Browne
"Brick Shakespeare: The ComediesEEA Midsummer NightEEs Dream, The Tempest, Much Ado About Nothing, and The Taming of the Shrew",John McCann
Brickstone,Sim Ciarlo
Brida: A Novel (P.S.),Paulo Coelho
Bride of Pendorric,Victoria Holt
Bridge to Haven,Francine Rivers
Bridget Jones: Mad About the Boy,Helen Fielding
Bridget Jones: Mad About the Boy (Vintage Contemporaries),Helen Fielding
Bridget Jones: the Edge of Reason,Helen Fielding
Brief Interviews with Hideous Men,David Foster Wallace
Bright Dead Things: Poems,Ada Limn
"Bright Lights, Big City",Jay McInerney
Bright Lines: A Novel,Tanwi Nandini Islam
Brightest Heaven of Invention: A Christian Guide To Six Shakespeare Plays,Peter J. Leithart
"Bring Up the Bodies (Wolf Hall, Book 2)",Hilary Mantel
British Library Desk Diary 2016,British Library
British Literature of World War I,Angela K Smith
Broadway Babies Say Goodnight: Musicals Then and Now,Mark Steyn
Broadway Bound,Neil Simon
Broken Hierarchies: Poems 1952-2012,Geoffrey Hill
Brooklyn,Colm Toibin
Brooklyn Fictions: The Contemporary Urban Community in a Global Age (Bloomsbury Studies in the City),James Peacock
Brotherhood of Evil (The Family Jensen),William W. Johnstone
"Brown Girl, Brownstones",Paule Marshall
"Browsings: A Year of Reading, Collecting, and Living with Books",Michael Dirda
Brush Talks from Dream Brook,Shen Kuo
Buddenbrooks: The Decline of a Family,Thomas Mann
Buddhist Animal Wisdom Stories,Mark W. McGinnis
"Buddhist Tales of India, China, and Japan: Indian Section",Yoshiko Dykstra
Buffalo Bird Woman's Garden: Agriculture of the Hidatsa Indians (Borealis Books),Gilbert Wilson
Buffalo Trail: A Novel of the American West,Jeff Guinn
Building: Letters 1960-1975,Isaiah Berlin
Bulfinch's Mythology (Leather-bound Classics),Thomas Bulfinch
Bullets in the Washing Machine,Melissa Littles
Burger's Daughter,Nadine Gordimer
Burning Desire (Dark Kings),Donna Grant
Burning Down George Orwell's House,Andrew Ervin
Burro Genius: A Memoir,Victor Villasenor
Busker: Avazkhan-e Doregard (Persian Edition),Moniro Ravanipour
But Beautiful: A Book About Jazz,Geoff Dyer
"Butch Queens Up in Pumps: Gender, Performance, and Ballroom Culture in Detroit (Triangulations: Lesbian/Gay/Queer Theater/Drama/Performance)",Marlon  M. Bailey
Butcher's Crossing (New York Review Books Classics),John Williams
Butterball (Hesperus Classics),Guy de Maupassant
By Blood We Live,Glen Duncan
By Night in Chile,Roberto Bolao
By The Book: Stories and Pictures,Diane Schoemperlen
"By the Way, Meet Vera Stark (TCG Edition)",Lynn Nottage
Byron's Poetry and Prose (Norton Critical Edition),George Gordon Byron
C. S. Lewis at War: The Dramatic Story Behind Mere Christianity (Radio Theatre),C. S. Lewis
"C. S. Lewis Signature Classics: Mere Christianity, The Screwtape Letters, A Grief Observed, The Problem of Pain, Miracles, and The Great Divorce (Boxed Set)",C. S. Lewis
"C.S. Lewis: The Signature Classics Audio Collection: The Problem of Pain, The Screwtape Letters, The Great Divorce, Mere Christianity",C. S. Lewis
Caballo de Troya 1. Jerusaln (NE) (Caballo De Troya / Trojan Horse) (Spanish Edition),Juan Jos Bentez
Caffe Cino: The Birthplace of Off-Off-Broadway (Theater in the Americas),Wendell C. Stone
Calculating God,Robert J. Sawyer
Caleb,Charles Alverson
"Calico Joe by Grisham, John [Hardcover]",John.. Grisham
Calico Joe: A Novel,John Grisham
California's Wild Edge,Tom Killion
Call It Sleep: A Novel,Henry Roth
Calligraphy Lesson: The Collected Stories,Mikhail Shishkin
Calloustown,George Singleton
Calming The Anxiety Within (The Healing Journal Series),Kaitlyn Storm
Camera Lucida: Reflections on Photography,Roland Barthes
Can I Taste It?,David Weaver
Can such things be?: A Collection of Supernatural Fiction,Ambrose Bierce
Canadian Fiction: A Guide to Reading Interests (Genreflecting Advisory Series),Sharron Smith
Candace Reign (Zane Presents),Sharai Robbin
Candide,Voltaire
Candide (A Norton Critical Edition),Voltaire
Candide (Dover Thrift Editions),Voltaire
Candies: A Comedy Composite,Basil H. Johnston
Candy: A Novel of Love and Addiction,Luke Davies
Cane (New Edition),Jean Toomer
Cannibal (A Jack Sigler Thriller Book 7) (Volume 7),Jeremy Robinson
Canterbury Tales by Chaucer,Geoffrey Chaucer
Canyons: A Novel,Samuel Western
Cape Horn and Other Stories from the End of the World (Discoveries),Francisco Coloane
Capitalism: The Unknown Ideal,Ayn Rand
Capone Bloodline: A T-Bone Capone Adventure,Tom Belton
Captain Blood (Penguin Classics),Rafael Sabatini
Captains Courageous (Dramatized),Rudyard Kipling
Capturing the Moon: Classic and Modern Jewish Tales,Edward Feinstein
Cara Massimina: A Novel (Duckworth and the Italian Girls),Tim Parks
Caramelo,Sandra Cisneros
Caramelo: En Espanol (Spanish Edition),Sandra Cisneros
Carey's Trade,Gregory Ast
Caribbee: A Kydd Sea Adventure (Kydd Sea Adventures),Julian Stockwin
Cariboo Magi,Lucia Frangione
Carmilla,J. Sheridan LeFanu
Carmilla: A Critical Edition (Irish Studies),Joseph Le Fanu
Carol (Movie Tie-In),Patricia Highsmith
"Carolina Israelite: How Harry Golden Made Us Care about Jews, the South, and Civil Rights",Kimberly Marlowe Hartnett
Carribbean Discourse: Selected Essays (Caribbean and African Literature),Edouard Glissant
Carried Forward By Hope (# 6 in the Bregdan Chronicles Historical Fiction Romance Series) (Volume 6),Ginny Dye
"Carry On, Jeeves (A Jeeves and Bertie Novel)",P. G. Wodehouse
"Carrying Albert Home: The Somewhat True Story of A Man, His Wife, and Her Alligator",Homer Hickam
Cartel 3: The Last Chapter (The Cartel),Ashley and JaQuavis
Casca #01: Eternal Mercenary,Barry Sadler
Casting the Runes and Other Ghost Stories (Oxford World's Classics),M. R. James
Cat on a Hot Tin Roof,Tennessee Williams
Cat Out of Hell,Lynne Truss
Catalog of Unabashed Gratitude (Pitt Poetry Series),Ross Gay
Catch-22,Joseph Heller
Cathedral of the Black Madonna: The Druids and the Mysteries of Chartres,Jean Markale
Catherine de Valois:  A Play in Three Acts (The Legendary Women of World History) (Volume 2),Laurel A. Rockefeller
Cattle King for a Day (Western Short Stories Collection),L. Ron Hubbard
Catullus,John Ferguson
Catullus and the Poetics of Roman Manhood,David Wray
"Catullus, Tibullus, Pervigilium Veneris (Loeb Classical Library No. 6)",Gaius Valerius Catullus
Caucasia: A Novel,Danzy Senna
"Celebrate Christmas: Easy Dramas, Speeches, and Recitations for Children",Peggy Augustine
Celia's House,D.E. Stevenson
Censorship and the Limits of the Literary: A Global View,Nicole Moore
Centennial: A Novel,James A. Michener
Ceremonies in Dark Old Men: A Play,Lonne Elder III
Ceremony: (Penguin Classics Deluxe Edition),Leslie Marmon Silko
Certain Dark Things: Stories,M.J. Pack
Certain Prey,John Sandford
Changing the Subject: Art and Attention in the Internet Age,Sven Birkerts
Chango's Fire: A Novel,Ernesto Quinonez
Charlie and the Chocolate Factory: a Play,Roald Dahl
Charlotte Bront: A Fiery Heart,Claire Harman
Charlotte's Web (Trophy Newbery),E. B. White
"Chased: A Novella: Titan, Book 3.5",Cristin Harber
Chasing Utopia: A Hybrid,Nikki Giovanni
Chekhov: The Essential Plays (Modern Library Classics),Anton Chekhov
Chekhov's Three Sisters and Woolf's Orlando: Two Renderings for the Stage,Virginia Woolf
Chelsea Girls: A Novel,Eileen Myles
Chemical Theatre,Charles Nicholl
Chemistry,Steven S. Zumdahl
"Chemistry, 11th Edition",Raymond Chang
Cherry,Mary Karr
Chesapeake: A Novel,James A. Michener
Chester Creek Ravine: Haiku,Bart Sutter
"CHEW Omnivore Edition, Vol. 2",John Layman
Chicago Stories (Prairie State Books),James T. Farrell
Childe Harold's Pilgrimage,Lord Byron
Children of Paradise: A Novel (P.S.),Fred D'Aguiar
Children of the Days: A Calendar of Human History,Eduardo Galeano
Chilly Scenes of Winter,Ann Beattie
China Dolls: A Novel,Lisa See
China Men,Maxine Hong Kingston
China Rich Girlfriend: A Novel,Kevin Kwan
"Chinese Link: Beginning Chinese, Simplified Character Version, Level 1/Part 1 (2nd Edition)",Sue-mei Wu
"Chinese Link: Beginning Chinese, Simplified Character Version, Level 1/Part 2 (2nd Edition)",Sue-mei Wu
"Chinese Link: Beginning Chinese, Traditional Character Version, Level 1/Part 2 (2nd Edition)",Sue-mei Wu
Chinese Mythology: An Introduction,Anne M. Birrell
Chinglish (TCG Edition),David Henry Hwang
Chippewa Customs (Publications of the Minnesota Historical Society),Frances Densmore
Chiyo-ni: Woman Haiku Master,Patricia Donegan
Choices (Cole),Noah Gordon
Choir Boy,Tarell Alvin Mccraney
Choir Boy,Tarell Alvin McCraney
Choose Your Own Misery: The Office Adventure,Mike MacDonald
Christ the Lord: Out of Egypt: A Novel,Anne Rice
Christmas at Lilac Cottage: A perfect romance to curl up by the fire with (White Cliff Bay) (Volume 1),Holly Martin
Christmas at Thompson Hall: And Other Christmas Stories (Penguin Christmas Classics),Anthony Trollope
Christmas at Tiffany's: A Novel,Karen Swan
Christmas Bells: A Novel,Jennifer Chiaverini
Christmas Bliss: A Novel,Mary Kay Andrews
Christmas Gifts: A Children's Christmas Play,Valerie Howard
Christmas Lights,Valerie Howard
Christmas on Stage: An Anthology of Royalty-Free Christmas Plays for All Ages,Theodore O. Zapel
Christmas Program Builder No. 50,Paul M. Miller
Christmas Program Builder No. 64: Creative Resources for Program Directors (Lillenas Drama),Heidi Petak
Christopher Durang Volume I: 27 Short Plays,Christopher Durang
Christopher Marlowe: The Complete Plays,Christopher Marlowe
Chronicle of a Death Foretold,Gabriel Garca Mrquez
Chronicle of the Abbey of Bury St. Edmunds (Oxford World's Classics),Jocelin of Brakelond
Cicero and the Jurists,Jill Harries
Cicero: A Portrait (Bristol Classical Paperbacks),Elizabeth Rawson
Cicero: Catilinarians (Cambridge Greek and Latin Classics),Marcus Tullius Cicero
Cicero: On the Orator: Book 3. On Fate. Stoic Paradoxes. On the Divisions of Oratory: A. Rhetorical Treatises (Loeb Classical Library No. 349) (English and Latin Edition),Cicero
Cinnamon and Gunpowder: A Novel,Eli Brown
Cicero: Rhetorica ad Herennium (Loeb Classical Library No. 403) (English and Latin Edition),Cicero
Cien aos de soledad (Spanish Edition),Gabriel Garca Mrquez
Cinderella (Dramatized),Brothers Grimm
Cinderella Skeleton,Robert D. San Souci
Circle of Friends,Maeve Binchy
Circling the Sun: A Novel,Paula McLain
Cities of Salt,Abdelrahman Munif
Cities of the Plain: Border Trilogy (3),Cormac McCarthy
Citizen: An American Lyric,Claudia Rankine
City of Clowns,Daniel Alarcn
City of God,Gil Cuadros
City of God (Penguin Classics),Augustine of Hippo
City of Lost Dreams: A Novel,Magnus Flyte
City of Thieves: A Novel,David Benioff
City on Fire: A novel,Garth Risk Hallberg
City On Fire: A Novelette,Mandy De Sandra
Civil Disobedience and Other Essays (Dover Thrift Editions),Henry David Thoreau
Civil War Stories (Dover Thrift Editions),Ambrose Bierce
CivilWarLand in Bad Decline,George Saunders
Claire of the Sea Light (Vintage Contemporaries),Edwidge Danticat
Clara and Mr. Tiffany: A Novel,Susan Vreeland
Clarissa Pinkola Estes Live: Theatre of the Imagination,Clarissa Pinkola Ests
"Clases de literatura.  Berkeley, 1980 (Spanish Edition)",Julio Cortzar
Clash of Eagles: The Clash of Eagles Trilogy Book I,Alan Smale
Classic Crews: A Harry Crews Reader,Harry Crews
Classic Goosebumps #7: Be Careful What You Wish For,R.L. Stine
"Classic Myths to Read Aloud: The Great Stories of Greek and Roman Mythology, Specially Arranged for Children Five and Up by an Educational Expert",William F. Russell
Classical Greek Prose: A Basic Vocabulary,Malcolm Campbell
Classical Monologues For Men (Audition Speeches),Chrys Salt
Classical Mythology,Mark Morford
Classical Mythology: A Very Short Introduction,Helen Morales
Classical Tragedy - Greek and Roman: Eight Plays in Authoritative Modern Translations,Aeschylus
"Classics Reimagined, Edgar Allan Poe: Stories & Poems",Edgar Allan Poe
"Classics Reimagined, The Wonderful Wizard of Oz",L. Frank Baum
Classics: A Very Short Introduction,Mary Beard
Claudian and the Roman Epic Tradition,Catherine Ware
Cleaning Nabokov's House: A Novel,Leslie Daniels
Clear Light of Day,Anita Desai
Cleopatra's Shadows,Emily Holleman
CliffsComplete Macbeth,William Shakespeare
CliffsComplete Romeo and Juliet,William Shakespeare
CliffsComplete Shakespeare's Hamlet,William Shakespeare
CliffsNotes on Bradbury's Fahrenheit 451,Kristi Hiner
CliffsNotes on Dickens' A Tale of Two Cities (Cliffsnotes Literature Guides),Marie Kalil
CliffsNotes on Fitzgerald's The Great Gatsby (Cliffsnotes Literature Guides),Kate Maurer
CliffsNotes on Golding's Lord of the Flies (Cliffsnotes Literature),Maureen Kelly
CliffsNotes on Homer's Odyssey (Cliffsnotes Literature Guides),Stanley P Baldwin
CliffsNotes on Rand's Anthem (Cliffsnotes Literature Guides),Andrew Bernstein
CliffsNotes on Rand's Atlas Shrugged (Cliffsnotes Literature Guides),Andrew Bernstein
CliffsNotes on Salinger's The Catcher in the Rye (Cliffsnotes Literature Guides),Stanley P. Baldwin
CliffsNotes on Shakespeare's Hamlet (Cliffsnotes Literature Guides),Carla Lynn Stockton
CliffsNotes on Shakespeare's Julius Caesar (Cliffsnotes Literature Guides),James E Vickers
CliffsNotes on Shakespeare's Macbeth (Cliffsnotes Literature),Alex Went
CliffsNotes on Shakespeare's Romeo and Juliet (Cliffsnotes Literature),Annaliese F Connolly
CliffsNotes on Shelley's Frankenstein (Cliffsnotes Literature Guides),Jeff Coghill
Climbing Parnassus: A New Apologia for Greek and Latin,Tracy Lee Simmons
Clipboard Christmas Skits,Tom Spence
"Close Your Eyes, Hold Hands (Vintage Contemporaries)",Chris Bohjalian
Closing Time: The Sequel to Catch-22,Joseph Heller
"Clotel: Or, The President's Daughter: A Narrative of Slave Life in the United States (Bedford Cultural Editions Series)",William Wells Wells Brown
Cloud 9,Caryl Churchill
Cloud Atlas: A Novel (Modern Library),David Mitchell
Cloudstreet: A Novel,Tim Winton
Clybourne Park,Bruce Norris
Coastal Disturbances: Four Plays,Tina Howe
Cobra Trap (Modesty Blaise series),Peter O'Donnell
Coca Kola (The Baddest Chick) Part 2,Nisa Santiago
Cocaine and Champagne: Road To My Recovery,Sherrie Lueder
Coeur de Lion,Ariana Reines
Cold Comfort Farm,Stella Gibbons
Cold Comfort Farm (Penguin Classics Deluxe Edition),Stella Gibbons
Cold-Cocked: On Hockey,Lorna Jackson
Coldheart Canyon: A Hollywood Ghost Story,Clive Barker
Coleridge's Poetry and Prose (Norton Critical Editions),Samuel Taylor Coleridge
"Colic Solved: The Essential Guide to Infant Reflux and the Care of Your Crying, Difficult-to- Soothe Baby",Bryan Vartabedian
Collateral: A Novel,Ellen Hopkins
Collected Fictions,Jorge Luis Borges
Collected Ghost Stories (Oxford World's Classics),M. R. James
Collected Haiku of Yosa Buson,Yosa Buson
Collected Poems,Jack Gilbert
Collected Poems,Philip Larkin
Collected Poems,Robert Hayden
Collected Short Stories: of Percival Christopher Wren (Volume 2),P. C. Wren
Collected Shorter Fiction: Volume 1 (Everyman's Library),Leo Tolstoy
Collected Stories,Gabriel Garcia Marquez
"College Essays That Made a Difference, 6th Edition (College Admissions Guides)",Princeton Review
Color My Fro: A Natural Hair Coloring Book for Big Hair Lovers of All Ages,Crystal Swain-Bates
"Comanche Moon (Lonesome Dove Story, Book 2)",Larry McMurtry
"Combat Ops (Tom Clancy's Ghost Recon, Book 2)",David Michaels
Combined and Uneven Development: Towards a New Theory of World-Literature (Postcolonialism Across the Disciplines LUP),Sharae Deckard
"Come On, Rain",Karen Hesse
Come Rain or Come Shine (A Mitford Novel),Jan Karon
Comedia & Drama (Dionisios) (Volume 1) (Spanish Edition),Carmen Resino
Comedy Scenes for Student Actors: Short Sketches for Young Performers,Laurie Allen
Cometh the Hour (The Clifton Chronicles),Jeffrey Archer
Comfort: A Novel of the Reverse Underground Railroad,H. A. Maxson
Coming to Rosemont,Barbara Hinske
Commodore Hornblower (Hornblower Saga),C. S. Forester
"Common Liar: Essay on ""Antony and Cleopatra"" (Study in English)",Janet Adelman
Como agua para chocolate (Spanish Edition),Laura Esquivel
"Compact Literature: Reading, Reacting, Writing",Laurie G. Kirszner
Comparative Literature 108: Myths and Mythologies Package for Pennsylvania State University,Sidney Aboul-Hosn
Comparative Religion For Dummies,William P. Lazarus
"Compendium of Roman History / Res Gestae Divi Augusti (Loeb Classical Library, No. 152)",Velleius Paterculus
Complete Greek Tragedies Euripides,Euripedes
Complete Greek Tragedies: Aeschylus I,trans Aeschylus / Richmond Lattimore
"Complete Harley 2253 Manuscript, Volume 1 (Middle English Texts)",Susanna Fein
Complete Letters (Oxford World's Classics),Pliny the Younger
Complete Plays of Aristophanes (Bantam Classics),Aristophanes
Complete Poems and Selected Letters of John Keats (Modern Library Classics),John Keats
Complete Poems and Songs of Robert Burns,Robert Burns
Complete Poems of Whitman (Wordsworth Poetry) (Wordsworth Poetry Library),Walt Whitman
"Complete Poems, 1904-1962 (Liveright Classics)",E. E. Cummings
Complete Sonnets and Poems: The Oxford Shakespeare The Complete Sonnets and Poems (Oxford World's Classics),William Shakespeare
Complete Stories (Penguin Classics),Dorothy Parker
Complete Stories and Poems of Edgar Allan Poe,Edgar Allan Poe
Complete Works of Oscar Wilde (Collins Classics),Oscar Wilde
Complete Works of William Shakespeare (Leather-bound Classics),William Shakespeare
Comprehensive Chess Endings Volume 4 Pawn Endings,Yuri Averbakh
Concepts In Solids: Lectures On The Theory Of Solids,Philip Warren Anderson
Concerning the Book that is the Body of the Beloved,Gregory Orr
Concrete Situations (Situations Series) (Volume 1),Crystal Darks
Conde de Montecristo (Coleccion los Inmortales) (Spanish Edition),Alejandro Dumas
Conduit: [A Collection of Poems and Short Stories by Jon Goode],Jon Goode
Confession of the Lioness: A Novel,Mia Couto
Confessions of a Crap Artist,Philip K. Dick
Confessions of a First Lady,Denora M Boone
Confessions of a First Lady 2,Denora M Boone
Confessions of a Mask,Yukio Mishima
Confessions of a Preachers Wife (Urban Christian),Mikasenoja
Confessions of a Wild Child (Lucky: the Early Years),Jackie Collins
Conflict Resolution for Holy Beings: Poems,Joy Harjo
Connemara: Listening to the Wind,Tim Robinson
Conqueror: A Novel of Kublai Khan (The Khan Dynasty),Conn Iggulden
Consequences Of A SideChick: SideChicks,Vladimir Dubois
Consider the Lobster and Other Essays,David Foster Wallace
Consorts of the Caliphs: Women and the Court of Baghdad (Library of Arabic Literature),Ibn al-Sai
Constantinople and the West in Medieval French Literature: Renewal and Utopia (Gallica),Rima Devereaux
Constellation Myths: with Aratus's Phaenomena (Oxford World's Classics),Eratosthenes
Constellations: A Play,Nick Payne
Consumed by Fire (The Fire Series),Anne Stuart
Contact Harvest (Halo),Joseph Staten
Contemporary Chicana Literature: (Re)Writing the Maternal Script,Cristina Herrera
Contested Will: Who Wrote Shakespeare?,James Shapiro
"Continuum: New And Selected Poems, Revised Edition",Mari Evans
Contrition,Maura Weiler
Copenhagen,Michael Frayn
Copper and Stone: stories,Bethany Snyder
Copper Sun,Sharon M. Draper
Coram Boy (Nick Hern Books),Jamila Gavin
Corduroy Mansions (Corduroy Mansions Series),Alexander McCall Smith
Coriolanus (The New Cambridge Shakespeare),William Shakespeare
Coriolanus: Oxford School Shakespeare (Oxford School Shakespeare Series),William Shakespeare
Coronado's Children: Tales of Lost Mines and Buried Treasures of the Southwest (Barker Texas History Center Series),J. Frank Dobie
Corregidora,Gayl Jones
Corrupt City (Urban Books),Tra Verdejo
Corsair (The Oregon Files),Clive Cussler
Cougar Club,Dark Chocolate
Could You Ever Live Without?,David Jones
"Count Magnus and Other Ghost Stories (The Complete Ghost Stories of M. R. James, Vol. 1)",M. R. James
Count the Waves: Poems,Sandra Beasley
"Countdown City: The Last Policeman, Book 2",Ben H. Winters
Countdown: M Day,Tom Kratman
Counternarratives,John Keene
Courage: The Backbone of Leadership,Gus Lee
Coven Thirteen Motorcycle Club: Volume One: Phoenix Fire (Volume 1),C Leigh Addison
Covenant of War (Lion of War Series),Cliff Graham
Coyote Wisdom: The Power of Story in Healing,Lewis Mehl-Madrona
Cracking India: A Novel,Bapsi Sidhwa
Cracks in Her Foundation,Shani Mixon
Cracks In The Sidewalk,Bette Lee Crosby
Cranford (Hardcover Classics),Elizabeth Gaskell
Crash Course: Essays From Where Writing and Life Collide,Robin Black
Crave Radiance: New and Selected Poems 1990-2010,Elizabeth Alexander
Crazy Horse's Girlfriend,Erika T. Wurth
Crazy In Luv 2: Blood Don't Make You Family,La'Tonya West
Crazy Rich Asians,Kevin Kwan
Creating a Scene in Corinth: A Simulation,Reta Halteman Finger
Creation Myths,Marie-Louise Von Franz
Creatures of a Day: And Other Tales of Psychotherapy,Irvin D. Yalom
"Cries for Help, Various: Stories",Padgett Powell
Crime and Punishment,Fyodor Dostoyevsky
Crime and Punishment: Pevear & Volokhonsky Translation (Vintage Classics),Fyodor Dostoevsky
Crimes of the Heart.,Beth Henley
Critique of Pure Reason (Penguin Classics),Immanuel Kant
Cromwell's Place in History: Founded on Six Lectures Delivered in the University of Oxford (Classic Reprint),Samuel Rawson Gardiner
Crooked,Catherine Trieschmann
Crooked Heart: A Novel,Lissa Evans
Crossing Delancey: A Romantic Comedy,Susan Sandler
Crossing to Safety (Modern Library Classics),Wallace Earle Stegner
Crossings: Nietzsche and the Space of Tragedy,John Sallis
Crossroads (Urban Books),Skyy
Crow: From the Life and Songs of the Crow,Ted Hughes
Crowned Heads,Thomas Tryon
"Crowned: Becoming the Woman of my Dreams: The Missing Things Were Goddess Wings: Poems, Prayers, and Love Letters",Sherry Sharp
Crowning Glory (Urban Christian),Pat Simmons
Crumbs from the Table of Joy and Other Plays,Lynn Nottage
Crusade (Destroyermen),Taylor Anderson
Crush (Yale Series of Younger Poets),Richard Siken
"Cry, the Beloved Country",Alan Paton
Cryptos,James R Wylder
Crnica de una muerte anunciada (Spanish Edition),Gabriel Garca Mrquez
Cuatro dias de enero (Best Seller (Debolsillo)) (Spanish Edition),Jordi Sierra
Cuatro ruedas compartidas (Spanish Edition),Mauro Hernandez
Cuentos de Amor de Locura y de Muerte,Horacio Quiroga
Cuentos de Amor de Locura y de Muerte (Spanish Edition),Horacio Quiroga
Cuentos De La Alhambra (1888) (Spanish Edition),Washington Irving
Cultural Amnesia: Necessary Memories from History and the Arts,Clive James
Cultural Intelligence: A Guide to Working with People from Other Cultures,Brooks Peterson
Cupid and Psyche: An Adaptation from The Golden Ass of Apuleius (Latin Edition),Apuleius
Cure for the Common Breakup (Black Dog Bay Novel),Beth Kendrick
Curiosity,Alberto Manguel
Curious Lives: Adventures from the Ferret Chronicles,Richard Bach
Cutting for Stone,Abraham Verghese
Cymbeline (The New Cambridge Shakespeare),William Shakespeare
D'aulaire's Book of Greek Myths,Ingri d'Aulaire
Damaged Goods,Nikki Urban
Damascus Nights,Rafik Schami
Damned,Chuck Palahniuk
Damsels in Distress (Urban Books),Nikita Lynnette Nichols
Dance Hall of the Dead,Tony Hillerman
Dance Me to the End of Love (Art & Poetry),Leonard Cohen
Dancer from the Dance: A Novel,Andrew Holleran
Dancer: A Novel (Picador Modern Classics),Colum McCann
Dancing at Lughnasa,Brian Friel
"Dancing at the Edge of the World: Thoughts on Words, Women, Places",Ursula K. Le Guin
Dancing at the Harvest Moon,K.C. McKinnon
Dancing Dogs: Stories,Jon Katz
Dancing with Butterflies: A Novel,Reyna Grande
Dandelion Through the Crack,Kiyo Sato
Dangerous to Go Alone!: an anthology of gamer poetry,CB Droege
Dangerous Work: Diary of an Arctic Adventure,Arthur Conan Doyle
"Dangerously In Love: ""Blame It on the Streets"" (Volume 1)",Aletta H.
Daniel X: Watch the Skies,James Patterson
"Dante, Poet of the Desert: History and Allegory in the DIVINE COMEDY",Giuseppe Mazzotta
"Dark Celebration: A Carpathian Reunion (The Carpathians (Dark) Series, Book 14)",Christine Feehan
Dark Chaos (# 4 in the Bregdan Chronicles Historical Fiction Romance Series) (Volume 4),Ginny Dye
Dark Infidelity,Shawn Starr
Dark Night of the Soul (Dover Thrift Editions),St. John of the Cross
Dark Sparkler,Amber Tamblyn
Dark Sweet: A Collection of Poetry,Keishi Ando
Dark Watch (The Oregon Files),Clive Cussler
Darkest Flame (Dark Kings),Donna Grant
Das Zeichen der Vier: Ein Sherlock Holmes Roman (German Edition),Arthur Conan Doyle
Dashing Through the Snow - Acting Edition,"Nicholas Hope, and Jamie Wooten Jessie Jones"
Dashing Through the Snow: A Christmas Novel (Random House Large Print),Debbie Macomber
Daughter of Fortune: A Novel,Isabel Allende
Daughters of Copper Woman,Anne Cameron
David Copperfield (Penguin Classics),Charles Dickens
"David Foster Wallace's Infinite Jest: A Reader's Guide, 2nd Edition",Stephen J. Burn
Dawn,Elie Wiesel
Day After Night: A Novel,Anita Diamant
Day: A Novel,Elie Wiesel
Days Of Poetry: My writing,Genna Beth Strachan
De Nerval: Selected Writings (Penguin Classics),Gerard de Nerval
De Profundis,Oscar Wilde
De Profundis and Other Prison Writings (Penguin Classics),Oscar Wilde
De Shootinest Gent'man & Other Tales,Nash Buckingham
Dead Cert,Dick Francis
Dead Man's Cell Phone (TCG Edition),Sarah Ruhl
Dead Man's Hand: An Anthology of the Weird West,John Joseph Adams
Dead Man's Walk (Lonesome Dove),Larry McMurtry
Dead Men's Boots,Mike Carey
Dead Shot: A Sniper Novel,Jack Coughlin
Dead Six,Larry Correia
Dead Solid Perfect,Dan Jenkins
Deadlight Hall: A haunted house mystery (A Nell West and Michael Flint Haunted House Story),Sarah Rayne
Deadline,Sandra Brown
Deadly Deals (Sisterhood),Fern Michaels
Deadtown Abbey: An Undead Homage,Sean Hoade
Deaf Sentence: A Novel,David Lodge
Deal Breaker: The First Myron Bolitar Novel,Harlan Coben
Dear and Glorious Physician: A Novel about Saint Luke,Taylor Caldwell
Dear Emily,Fern Michaels
Dear Miss Breed: True Stories of the Japanese American Incarceration During World War II and a Librarian Who Made a Difference,Joanne Oppenheim
"Dear Santa: Children's Christmas Letters and Wish Lists, 1870 - 1920",Chronicle Books
Death and Taxes: Hydriotaphia and Other Plays,Tony Kushner
Death and the King's Horseman: A Play,Wole Soyinka
"Death and the King's Horseman: Authoritative Text, Backgrounds and Contexts, Criticism, Norton",Wole Soyinka
Death and the Maiden,Ariel Dorfman
Death at Tammany Hall (A Gilded Age Mystery),Charles O'Brien
Death Before Compline: Short Stories,Sharan Newman
Death Before Wicket: A Phryne Fisher Mystery,Kerry Greenwood
Death by Facebook,Everett Peacock
Death by Water,Kenzaburo Oe
Death Comes for the Archbishop (Vintage Classics),Willa Cather
Death Defying Acts,Woody Allen
Death du Jour (Temperance Brennan Novels),Kathy Reichs
Death in the Afternoon,Ernest Hemingway
Death in the Andes: A Novel,Mario Vargas Llosa
Death in Venice (Dover Thrift Editions),Thomas Mann
Death of a Salesman (Penguin Plays),Arthur Miller
Death of Kings (Saxon Tales),Bernard Cornwell
Death or Liberty: African Americans and Revolutionary America,Douglas R. Egerton
Death Rides the River: A Joshua Miller Adventure (Joshua Miller Series) (Volume 2),Wayne Lincourt
Death Traps: The Survival of an American Armored Division in World War II,Belton Y. Cooper
Debbie Doesn't Do It Anymore (Vintage Crime/Black Lizard),Walter Mosley
December (Seagull Books - The German List),Alexander Kluge
Decolonising the Mind (Studies in African Literature),Ngugi Wa Thiong'O
Decolonising the Mind: The Politics of Language in African Literature,Ngugi wa Thiong'o
"Decreation: Poetry, Essays, Opera",Anne Carson
Deep Down True: A Novel,Juliette Fay
Defence Speeches (Oxford World's Classics),Cicero
Del amor y otros demonios (Spanish Edition),Gabriel Garca Mrquez
Delicious!: A Novel,Ruth Reichl
Deliverance (Modern Library 100 Best Novels),James Dickey
"Demelza: A Novel of Cornwall, 1788-1790 (Poldark)",Winston Graham
"Demons (Everyman's Library, 182)",Fyodor Dostoevsky
Demons (Penguin Classics),Fyodor Dostoevsky
Demons: A Novel in Three Parts (Vintage Classics),Fyodor Dostoevsky
Der Kleine Prinz (German),Antoine de Saint-Exupry
Der Lehnsmann und das Hexenweib (German Edition),Annika Stinner
Der Traum ein Leben: Dramatisches Mrchen in vier Aufzgen (German Edition),Franz Grillparzer
Der Wildtdter (TREDITION CLASSICS) (German Edition),James Fenimore Cooper
Derailed,Dave Jackson
Descent to the Goddess: A Way of Initiation for Women,Sylvia Brinton Perera
Desdemona (Oberon Modern Plays),Toni Morrison
"Desert Sun, Red Blood",E. W. Farnsworth
"Desire and Anxiety: Circulations of Sexuality in Shakespearean Drama (Gender, Culture, Difference)",Valerie Traub
Desire and the Female Therapist: Engendered Gazes in Psychotherapy and Art Therapy,Joy Schaverien
"Desolation Island  (The Aubrey/Maturin Novels, Book 5)",Patrick O'Brian
Desperately Seeking Exclusivity (Volume 1),Christopher Markland
Detroit '67,Dominique Morisseau
Developing Minds: An American Ghost Story,Jonathan LaPoma
Deviants,Peter Kline
Devices and Desires (Engineer Trilogy),K. J. Parker
Devil Knows: A Tale of Murder and Madness in America's First Century,David Joseph Kolb
Devil on the Cross (Heinemann African Writers Series),Ngugi wa Thiong'o
"Devil's Gate (Numa Files, Book 9)",Clive Cussler
Devils (Oxford World's Classics),Fyodor Dostoevsky
Dewdrops on a Lotus Leaf: Zen Poems of Ryokan,Ryokan
Dia's Story Cloth: The Hmong People's Journey of Freedom,Dia Cha
Diamond Head: A Novel,Cecily Wong
Diaries 1969-1979: The Python Years (Michael Palin Diaries),Michael Palin
"Diaries: Diary and Autobiographical Writings of Louisa Catherine Adams, Volumes 1 and 2: 1778-1849 (Adams Papers)",Louisa Catherine Adams
Diary (The Margellos World Republic of Letters),Witold Gombrowicz
Diary of a Mad Bride,Laura Wolf
Diary of a Mad Diva,Joan Rivers
Diary Of An Oxygen Thief,Anonymous
"Diasporic Dis(locations): Indo-Caribbean Women Writers Negotiate the ""Kala Pani""",Brinda J. Mehta
Dick Francis's Damage,Felix Francis
Dick Francis's Gamble,Felix Francis
Dickens at Christmas (Vintage Classics),Charles Dickens
Dickinson: Poems (Everyman's Library Pocket Poets),Emily Dickinson
Dickinson's Misery: A Theory of Lyric Reading,Virginia Jackson
Dictee,Theresa Hak Kyung Cha
Did You Ever Have A Family,Bill Clegg
Dien Cai Dau (Wesleyan Poetry Series),Yusef Komunyakaa
Different Seasons (Signet),Stephen King
Digest (Stahlecker Selections),Gregory Pardlo
Digging into Literature,Joanna Wolfe
Digital Mammals,Luiz Mauricio Azevedo
Dilemma of a Ghost and Anowa,Ama Ata Aidoo
Dime quin soy (Spanish Edition),Julia Navarro
Dinner with Buddha,Roland Merullo
Din Bahane': The Navajo Creation Story,Paul G. Zolbrod
"Dionysius of Halicarnassus: Roman Antiquities, Volume VI. Books 9.25-10 (Loeb Classical Library No. 378)",Dionysius of Halicarnassus
Directed by Desire: The Collected Poems of June Jordan,June Jordan
Dirty Divorce part 4 (The Dirty Divorce) (Volume 4),Miss KP
"Dirty Little Secrets: A J.J. Graves Mystery, Book 1",Liliana Hart
Dirty Money (Urban Books),Ashley and JaQuavis
Dirty Pretty Things,Michael Faudet
Dirty Rush,Taylor Bell
"Disappearing Acts: Spectacles of Gender and Nationalism in Argentina's ""Dirty War""",Diana Taylor
Disappearing Man,Doug Peterson
"Disarming: Reign of Blood, Book 2",Alexia Purdy
Disco for the Departed (A Dr. Siri Paiboun Mystery),Colin Cotterill
Disgraced: A Play,Ayad Akhtar
Disgruntled: A Novel,Asali Solomon
Disney Fairies Storybook Collection Special Edition,Disney Book Group
"Dispara, yo ya estoy muerto (Spanish Edition)",Julia Navarro
Distant Neighbors: The Selected Letters of Wendell Berry and Gary Snyder,Gary Snyder
Divine Misdemeanors: A Novel (Merry Gentry),Laurell K. Hamilton
Divine Secrets of the Ya-Ya Sisterhood: A Novel (The Ya-Ya Series),Rebecca Wells
Diving into the Wreck: Poems 1971-1972,Adrienne Rich
Divinity School,Alicia Jo Rabins
Divisadero,Michael Ondaatje
Divorce Turkish Style (Kati Hirschel Murder Mystery),Esmahan Aykol
Doctor Thorne (Oxford World's Classics),Anthony Trollope
Doctor Who: City of Death,Douglas Adams
Doctor Zhivago,Boris Pasternak
Doctor Zhivago (Vintage International),Boris Pasternak
Doctors: A Novel,Erich Segal
Dog Sees God: Confessions of a Teenage Blockhead - Acting Edition,Bert V. Royal
Dog Songs: Poems,Mary Oliver
Dollbaby: A Novel,Laura Lane McNeal
Dollhouse: A Novel,Kim Kardashian
Dominoes,Susan Emshwiller
Don Juan Tenorio (Clasicos de la literatura series),Jose Zorrilla
Don Quijote de la Mancha (Spanish Edition),Miguel Cervantes
Don Quijote de la Mancha (Spanish Edition),Miguel de Cervantes
"Don Quijote: Legacy Edition (Cervantes & Co.) (Spanish Edition) (European Masterpieces, Cervantes & Co. Spanish Classics)",Miguel de Cervantes Saavedra
Don Quixote,Miguel de Cervantes
Don Quixote of La Mancha (Restless Classics),Miguel de Cervantes
Don't Ever Get Old (Buck Schatz Series),Daniel Friedman
"Don't Jump: Sex, Drugs, Rock 'N Roll... And My Fucking Mother",Vicki Abelson
Don't Let Me Go,Catherine Ryan Hyde
Don't Make Me Wait (Urban Books),Shana Burton
Don't: A Manual Of Mistakes And Improprieties More Or Less Prevalent In Conduct And Speech (1884),Censor
Donald Duk,Frank Chin
Doomed,Chuck Palahniuk
Dopefiend,Donald Goines
Dopeman: Memoirs of a Snitch:: Part 3 of Dopeman's Trilogy,JaQuavis Coleman
Down For Him: A Hood Love Story,Jada Pullen
Down These Mean Streets,Piri Thomas
"Dr Seuss Collection 20 Books Set Pack (The Cat in the Hat, Green Eggs and Ham, Fox in Socks, One Fish Two Fish Red Fish Blue Fish, How the Grinch Stole Christmas!, Oh the Places You'll Go!, the Cat in the Hat Comes Back, Dr. Seuss' Abc, Dr. Seuss ..)",Dr. Seuss
Dr. Faustus (Dover Thrift Editions),Christopher Marlowe
Dr. Seuss Pops Up,Dr. Seuss
Dracula,Bram Stoker
Dracula: A Play in Two Acts,Bram Stoker
Dracula: Writer's Digest Annotated Classics,Bram Stoker
Dracula's Guest and Other Weird Tales (Penguin Classics),Bram Stoker
Dragon (Dirk Pitt Adventure),Clive Cussler
Dragon Warrior (Midnight Bay),Janet Chapman
Dragonfish: A Novel,Vu Tran
Dragonvein: Book Two,Brian D. Anderson
Drama Essentials: An Anthology of Plays,Matthew Roudane
Drama Ministry,Steve Pederson
"Drama, Drinks and Double Faults: The Skinny about Tennis Fanatics That No One Has Had the Balls to Say . . . 'Til Now!",Mary Moses
"Drama, Skits, & Sketches 3",Youth Specialties
Drama: A Pocket Anthology (Penguin Academics Series) (5th Edition),R. S. Gwynn
Dreams in a Time of War: A Childhood Memoir,Ngugi wa Thiong'o
Dreams Of My Mothers: A Story Of Love Transcendent,Joel L. A. Peterson
Dreams of the Red Phoenix,Virginia Pye
Dreamtigers (Texas Pan American Series),Jorge Luis Borges
Drifting House,Krys Lee
Drink Cultura: Chicanismo,Jos Antonio Burciaga
Driven To Be Loved (Carmen Sisters),Pat Simmons
Driving the King: A Novel,Ravi Howard
Drone Command (A Troy Pearce Novel),Mike Maden
Drone String: Poems,Sherry Cook Stanforth
Drones (The Maliviziati Series.) (Volume 1),Johnny Ray
Drown,Junot Diaz
Drowned Boy: Stories (Mary McCarthy Prize in Short Fiction),Jerry Gabriel
Drowning Ruth: A Novel (Oprah's Book Club),Christina Schwarz
Drunk Enough to Say I Love You?,Caryl Churchill
Dry Bones: A Walt Longmire Mystery (Walt Longmire Mysteries),Craig Johnson
Dryland,Sara Jaffe
Dubliners,James Joyce
Dubliners (Dover Thrift Editions),James Joyce
Duende: Poems,Tracy K. Smith
Dumpling Field: Haiku Of Issa,Lucien Stryk
Dust,Yvonne Adhiambo Owuor
Dusty Locks and the Three Bears,Susan Lowell
Dysfluencies: On Speech Disorders in Modern Literature,Chris Eagle
Each Shining Hour: A Novel of Watervalley,Jeff High
Each Thing Unblurred is Broken,Andrea Baker
Early One Morning,Virginia Baily
Early Warning: A novel,Jane Smiley
"Earth Is My Mother, Sky Is My Father: Space, Time, and Astronomy in Navajo Sandpainting",Trudy Griffin-Pierce
Earth Medicine: Ancestor's Ways of Harmony for Many Moons,Jamie Sams
East of Acre Lane,Alex Wheatle
East Of Eden - John Steinbeck Centennial Edition (1902-2002),"John; With an Introduction by Wyatt, David Steinbeck"
East of Eden (Penguin Twentieth Century Classics),John Steinbeck
Ebenezer Scrooge: Ghost Hunter,Jaqueline Kyle
Echoes of a Distant Summer,Guy Johnson
"Echoes: Tired, Worn Out and Over It. Ignoring the Echoes and Listening to God's Voice.",Stephanie DeLores Moore
Ecocriticism on the Edge: The Anthropocene as a Threshold Concept,Timothy Clark
Edda (Everyman's Library),Snorri Sturluson
Edgar Allan Poe: Complete Tales and Poems,Edgar Allan Poe
EDGE OF WONDER: Notes From The Wildness Of Being,Victoria Erickson
Edipo Rey / Oedipus the King (Catedra Base / Base Cathedra) (Spanish Edition),Sophocles
Edith and Winnifred Eaton: CHINATOWN MISSIONS AND JAPANESE ROMANCES (Asian American Experience),Dominika Ferens
"Edith Stein: Letters to Roman Ingarden (Stein, Edith//the Collected Works of Edith Stein)",Edith Stein
Edmund Burke: Selected Writings and Speeches,Edmund Burke
Egeria's Travels,John Wilkinson
"Egyptian Mythology: A Guide to the Gods, Goddesses, and Traditions of Ancient Egypt",Geraldine Pinch
Egyptian Proverbs (Tem T Tchaas),Muata Ashby
Eichmann in Jerusalem: A Report on the Banality of Evil (Penguin Classics),Hannah Arendt
Eight Months on Ghazzah Street: A Novel,Hilary Mantel
El Alquimista: Una Fabula Para Seguir Tus Suenos,Paulo Coelho
El amor en los tiempos del clera (Oprah #59) (Spanish Edition),Gabriel Garca Mrquez
El Arroyo de la Llorona y otros cuentos,Sandra Cisneros
El Borak and Other Desert Adventures,Robert E. Howard
El Caballero De La Armadura Oxidada / the Knight in Rusty Armor (Spanish Edition),Robert Fisher
El caballero de los Siete Reinos [Knight of the Seven Kingdoms-Spanish] (A Vintage Espaol Original) (Spanish Edition),George R. R. Martin
El Conde Lucanor (Spanish Edition),Don Juan Manuel
El Corazon de un Artista (Spanish Edition),Rory Noland
El coronel no tiene quien le escriba (Spanish Edition),Gabriel Garca Mrquez
El Diario de Ana Frank (Anne Frank: The Diary of a Young Girl) (Spanish Edition),Ana Frank
El hombre que amaba a los perros (Coleccion Andanzas) (Spanish Edition),Leonardo Padura
El laberinto de la soledad,Octavio Paz
El Leon Bruja y el Ropero (Narnia) (Spanish Edition),C. S. Lewis
El maestro y Margarita (Spanish Edition),Mijal Bulgkov
El murmullo de las abejas (Spanish Edition),Sofa Segovia
El Presagio: El misterio ancestral que guarda el secreto del futuro del mundo (Spanish Edition),Jonathan Cahn
El profeta rojo (Spanish Edition),Orson Scott Card
El secreto del Bamb: Una fbula (Spanish Edition),Ismael Cala
El senor de las moscas / Lord of the Flies (Spanish Edition),William Golding
El Senor Presidente,Miguel Angel Asturias
El tiempo entre costuras: Una novela (Atria Espanol) (Spanish Edition),Mara Dueas
El Tunel / The Tunnel (Spanish Edition),Ernesto Sabato
El Viaje de Su Vida (Nivel 1 / Libro D) (Spanish Edition),Lisa Ray Turner
Eldorado Red,Donald Goines
Electra (Greek Tragedy in New Translations),Sophocles
Electra and Other Plays (Penguin Classics),Sophocles
Elegy for a Broken Machine: Poems,Patrick Phillips
Elephant Prince: The Story of Ganesh,Amy Novesky
Eleven Minutes: A Novel (P.S.),Paulo Coelho
Elias' Proverbs,Daniel Molyneux
Elijah in Jerusalem,Michael D. O'Brien
"Elizabeth Bishop: Poems, Prose and Letters (Library of America)",Elizabeth Bishop
Elizabeth Street,Laurie Fabiano
"Elliot, A Soldier's Fugue",Quiara Alegra Hudes
"Elmer Rice: Three Plays: The Adding Machine, Street Scene and Dream Girl",Elmer Rice
Emerson: Essays and Lectures: Nature: Addresses and Lectures / Essays: First and Second Series / Representative Men / English Traits / The Conduct of Life (Library of America),Ralph Waldo Emerson
Emerson: The Mind on Fire (Centennial Books),Robert D. Richardson
Emily Dickinson:  A Biography,Connie Ann Kirk
Emily Dickinson: Selected Letters,Emily Dickinson
Emily's Hope,Ellen Gable
Emma (Dover Thrift Editions),Jane Austen
Emma (Fourth Edition)  (Norton Critical Editions),Jane Austen
Emma (Penguin Classics),Jane Austen
"Emma, la cautiva (Spanish Edition)",Csar Aira
Emotionally Weird: A Novel,Kate Atkinson
Empire and Honor (Honor Bound),W.E.B. Griffin
Empire and Memory: The Representation of the Roman Republic in Imperial Culture (Roman Literature and its Contexts),Alain M. Gowing
Empire of Chance: The Napoleonic Wars and the Disorder of Things,Anders Engberg-Pedersen
Empire of Gold: A Novel (Nina Wilde and Eddie Chase),Andy McDermott
Empire of Kalman the Cripple (Library of Modern Jewish Literature),Yehuda Elberg
Empire of Self: A Life of Gore Vidal,Jay Parini
"En busca de la verdad / In search of truth: Discursos, Cartas De Lector, Entrevistas, Artculos / Speeches, Reader Letters, Interviews, Articles (Spanish Edition)",Thomas Bernhard
En el tiempo de las mariposas (Spanish Edition),Julia Alvarez
Enacting the Word: Using Drama in Preaching,James O. Chatham
Enchantress: A Novel of Rav Hisda's Daughter,Maggie Anton
End Of The Rainbow (Modern Plays),Peter Quilter
End Zone,Don DeLillo
Ends of Assimilation: The Formation of Chicano Literature,John Alba Cutler
Enemy In The Ashes,William W. Johnstone
Enemy Women,Paulette Jiles
English Romantic Poetry: An Anthology (Dover Thrift Editions),William Blake
Enigma of China: An Inspector Chen Novel (Inspector Chen Cao),Qiu Xiaolong
Envy (New York Review Books Classics),Yuri Olesha
Epigrams: With parallel Latin text (Oxford World's Classics),Martial
Epistemology of the Closet,Eve Kosofsky Sedgwick
Epitaph: A Novel of the O.K. Corral,Mary Doria Russell
Erasure: A Novel,Percival Everett
Eric Carle's Animals Animals,Eric Carle
Eros the Bittersweet (Canadian Literature),Anne Carson
Erratic Facts,Kay Ryan
Escape From The Ashes,William W. Johnstone
Escape Velocity,Charles Portis
Espresso Tales,Alexander McCall Smith
Essays (Everyman's Library Classics & Contemporary Classics),George Orwell
Essential Literary Terms: A Brief Norton Guide with Exercises,Sharon Hamilton
Essential Shakespeare Handbook,Leslie Dunton-Downer
Essie's Roses,Michelle Muriel
Estrategias Tematicas Y Narrativas En La Novela Feminizada De Maria De Zayas: Spa,Pilar Alcalde
Eternity's Sunrise: The Imaginative World of William Blake,Leo Damrosch
"Eudora Welty : Stories, Essays & Memoir (Library of America, 102)",Eudora Welty
Eugene O'Neill : Complete Plays 1913-1920 (Library of America),Eugene O'Neill
Eugene O'Neill : Complete Plays 1932-1943 (Library of America),Eugene O'Neill
Eugene Onegin (Russian Edition),Alexander Pushkin
Euphoria,Lily King
"Euripides IV: Helen, The Phoenician Women, Orestes (The Complete Greek Tragedies)",Euripides
"Euripides V: Bacchae, Iphigenia in Aulis, The Cyclops, Rhesus (The Complete Greek Tragedies)",Euripides
"Euripides V: Electra, The Phoenician Women, The Bacchae (The Complete Greek Tragedies) (Vol 5)",Euripides
"Euripides I: Alcestis, Medea, The Children of Heracles, Hippolytus (The Complete Greek Tragedies)",Euripides
"Euripides I: Alcestis, The Medea, The Heracleidae, Hippolytus (The Complete Greek Tragedies) (Vol 3)",Euripides
"Euripides III: Heracles, The Trojan Women, Iphigenia among the Taurians, Ion (The Complete Greek Tragedies)",Euripides
"Euripides, Volume III. Suppliant Women. Electra. Heracles (Loeb Classical Library No. 9)",Euripides
"Euripides, Volume IV. Trojan Women. Iphigenia among the Taurians. Ion (Loeb Classical Library No. 10)",Euripides
Euripides: Bacchae (Duckworth Companions to Greek & Roman Tragedy),Sophie Mills
Euripides: Hippolytus (Duckworth Companions to Greek & Roman Tragedy),Sophie Mills
Euripides: Medea (Cambridge Greek and Latin Classics) (Greek and English Edition),Euripides
Euripides: Medea (Cambridge Translations from Greek Drama),Euripides
Euripides: Medea (Duckworth Companions to Greek & Roman Tragedy),William Allan
"Euripides: Medea, Hippolytus, Heracles, Bacchae",Euripides
Euripides: Suppliant Women (Classical Texts) (Ancient Greek Edition),James Morwood
Euripides: Suppliant Women (Companions to Greek and Roman Tragedy),Ian C. Storey
Euripides: Trojan Women (Duckworth Companions to Greek & Roman Tragedy),Barbara Goff
Euripides' Hippolytus,Euripides
"European Proverbs in 55 Languages with Equivalents in Arabic, Persian, Sanskrit, Chinese and Japanese",Gyula Paczolay
Eurydice,Sarah Ruhl
Eve: A Novel,WM. Paul Young
Eve's Hollywood (New Yorkreview Books Classics),Eve Babitz
Even in Darkness: A Novel,Barbara Stark-Nemon
Evening Stars (Blackberry Island),Susan Mallery
Ever Yours: The Essential Letters,Vincent van Gogh
Evergreen Falls: A Novel,Kimberley Freeman
Every Closed Eye Ain't 'Sleep,MaRita Teague
Every Day,David Levithan
Every Day Is for the Thief: Fiction,Teju Cole
Every Thug Needs A Lady,Wahida Clark
Everybody Rise: A Novel,Stephanie Clifford
Everyman and Other Miracle and Morality Plays (Dover Thrift Editions),Anonymous
Everyone I Love is a Stranger to Someone,Annelyse Gelman
Everything and Nothing (New Directions Pearls),Jorge Luis Borges
Everything Begins and Ends at the Kentucky Club,Benjamin Alire Senz
Everything Flows (New York Review Books Classics),Vasily Grossman
Everything I Never Told You,Celeste Ng
Everything Is Illuminated: A Novel,Jonathan Safran Foer
Everything's Eventual: 14 Dark Tales,Stephen King
Evidence: Poems,Mary Oliver
Evolution of an Unorthodox Rabbi,John Moscowitz
Ex Libris: Confessions of a Common Reader,Anne Fadiman
Exclusive,Sandra Brown
Exclusive (The Godmothers),Fern Michaels
"Excursions with Thoreau: Philosophy, Poetry, Religion",Edward F. Mooney
Exemplary Stories (Oxford World's Classics),Miguel de Cervantes
Exemplary Traits: Reading Characterization in Roman Poetry,J. Mira Seo
Existentialism Is a Humanism,Jean-Paul Sartre
"Exploring Literature: Writing and Arguing about Fiction, Poetry, Drama, and the Essay, 5th Edition",Frank Madden
Exploring the Northern Tradition,Galina Krasskova
Explosion in a Cathedral,Alejo Carpentier
Extracting the Stone of Madness: Poems 1962 - 1972,Alejandra Pizarnik
"Extravagant Postcolonialism: Modernism and Modernity in Anglophone Fiction, 1958-1988",Brian T. May
Extreme Metaphors,J.G Ballard
Eyes Like Mine,Lauren Cecile
Eyes Only,Fern Michaels
Eyes: Novellas and Stories,William H. Gass
Eyewitness Christmas,Shell Isenhoff
Ezra Pound: Poet: Volume III: The Tragic Years 1939-1972,A. David Moody
F for Effort: More of the Very Best Totally Wrong Test Answers,Richard Benson
Face Off - The Baddest Chick Part 4,Nisa Santiago
Faces of the Game,Mandi Mac
Faces of the Game 2 (Volume 2),Mandi Mac
Facing Unpleasant Facts,George Orwell
Faggots,Larry Kramer
Fahrenheit 451,Ray Bradbury
"Fair Share Divorce for Women, Second Edition: The Definitive Guide to Creating a Winning Solution",Kathleen A. Miller
Fairy Godmothers Inc.,Jenniffer Wardell
Fairy-Faith in Celtic Countries (Library of the Mystic Arts),Walter Evans-Wentz
Faith and Fat Chances: A Novel,Carla Trujillo
Faith Healer,Brian Friel
Faithful and Virtuous Night: Poems,Louise Glck
Fake Fruit Factory,Patrick Wensink
Fala,Dana Kittendorf
Fall Leaves,Loretta Holland
Fall of Giants: Book One of the Century Trilogy,Ken Follett
Fall of Giants: Book One of the Century Trilogy,Ken Follett
Fall of Poppies: Stories of Love and the Great War,Heather Webb
Falling For a Drug Dealer (Volume 1),Melikia Gaino
Falling for You,Jill Mansell
Fallout,Ellen Hopkins
Family Affair LP,Debbie Macomber
"Family Furnishings: Selected Stories, 1995-2014 (Vintage International)",Alice Munro
Family Life: A Novel,Akhil Sharma
Family of Lies,Mary Monroe
Fans of the Impossible Life,Kate Scelsa
Fantasmagoria,Rick Wayne
Far Away (Nick Hern Books Drama Classics),Caryl Churchill
Far from the Madding Crowd (Penguin Classics),Thomas Hardy
Fast Animal,Tim Seibles
Fat City (New York Review Books Classics),Leonard Gardner
Fat Pig: A Play,Neil LaBute
"Fatal Decision: Edith Cavell, World War I Nurse",Terri Arthur
Fate Is the Hunter: A Pilot's Memoir,Ernest K. Gann
Fatelessness,Imre Kertesz
Fates and Furies: A Novel,Lauren Groff
"Father Comes Home From the Wars (Parts 1, 2 & 3)",Suzan-Lori Parks
Father Marquette and the Great Rivers (Vision Book),August Derleth
Fathers and Sons (Oxford World's Classics),Ivan Turgenev
Fathers and Sons (Penguin Classics),Ivan Turgenev
"Faust, Part One (Oxford World's Classics) (Pt. 1)",J. W. von Goethe
Faust: Part Two (Oxford World's Classics) (Pt. 2),J. W. von Goethe
Favorite Folktales from Around the World (The Pantheon Fairy Tale and Folklore Library),Jane Yolen
Favorite Novels and Stories: Four-Book Collection (Dover Thrift Editions),Jack London
Favorite Poems (Dover Thrift Editions),William Wordsworth
Fear and Loathing at Rolling Stone: The Essential Writing of Hunter S. Thompson,Hunter S. Thompson
Fear of Dying: A Novel,Erica Jong
Federal Agent (Violators: The Coalition) (Volume 3),Nancy Brooks
"Feed Your Vow, Poems for Falling into Fullness",Brooke McNamara
Feet of Clay: A Novel of Discworld,Terry Pratchett
Fefu and Her Friends,Maria Irene Fornes
Felicity: Poems,Mary Oliver
Female Hustler,Joy Deja King
Female Hustler Part 2,Joy Deja King
Fences,August Wilson
Fences (August Wilson Century Cycle),August Wilson
Fever 1793,Laurie Halse Anderson
Fever at Dawn,Pter Grdos
Fevre Dream,George R. R. Martin
"Feydeau Plays: 1: Heart's Desire Hotel, Sauce for the Goose, The One That Got Away, Now You See it, Pig in a Poke (World Classics) (Vol 1)",Georges Feydeau
Ficciones (Spanish Edition),Jorge Luis Borges
Fields of Fire,James Webb
Fierce and True: Plays for Teen Audiences,ChildrenEEs Theatre Company
Fierce Day,Rose Styron
Fiercombe Manor,Kate Riordan
Fifteen One-Act Plays,Sam Shepard
Fifteen Years,Kendra Norman-Bellamy
Fifty Letters of Pliny,Pliny the Younger
Fifty Shades of Dorian Gray (Classic),Nicole Audrey Spector
"Fighting for Rome: Poets and Caesars, History and Civil War",John Henderson
Figures of Speech Used in the Bible,E. W. Bullinger
Filth,Irvine Welsh
Finale: A Novel of the Reagan Years,Thomas Mallon
Finding Amos,J.D. Mason
Finding Emma (Finding Emma Series),Steena Holmes
Finding Freedom: Writings from Death Row,Jarvis Jay Masters
Finding Out: A Novel,Sheryn MacMunn
Finding Soul on the Path of Orisa: A West African Spiritual Tradition,Tobe Melora Correal
Finding the Dream: Dream Trilogy,Nora Roberts
Finding Them Gone: Visiting China's Poets of the Past,Red Pine
Fingersmith,Sarah Waters
Finish This Book,Keri Smith
Finishing Forty,Sean Patrick Brennan
Fire,Sebastian Junger
Fire Ice (The NUMA Files),Clive Cussler
Fire in Beulah,Rilla Askew
Fire in the Head: Shamanism and the Celtic Spirit,Tom Cowan
Fire in the Hole: Stories,Elmore Leonard
Fire in the Treetops: Celebrating Twenty-Five Years  of Haiku North America,"Michael Dylan Welch, Editor"
Fire Rising (Dark Kings),Donna Grant
Firefly Lane,Kristin Hannah
Firestorm (Anna Pigeon),Nevada Barr
First Frost,Sarah Addison Allen
First Love and Other Stories (Oxford World's Classics),Ivan Turgenev
Fishing Stories (Everyman's Pocket Classics),Henry Hughes
Five Ghosts Volume 1: The Haunting of Fabian Gray TP,Frank J. Barbiere
Five Great Greek Tragedies (Dover Thrift Editions),Sophocles
Five Great Science Fiction Novels (Dover Thrift Editions),H. G. Wells
Five Smooth Stones: A Novel (Rediscovered Classics),Ann Fairbairn
Five Women Wearing the Same Dress,Alan Ball
"Five Years of My Life, 1894-1899 (Classic Reprint)",Alfred Dreyfus
Flaming Iguanas: An Illustrated All-Girl Road Novel Thing,Erika Lopez
Flannery O'Connor : Collected Works : Wise Blood / A Good Man Is Hard to Find / The Violent Bear It Away / Everything that Rises Must Converge / Essays & Letters (Library of America),Flannery O'Connor
Flat Water Tuesday: A Novel,Ron Irwin
Flight Behavior: A Novel,Barbara Kingsolver
Flight of the Sparrow: A Novel of Early America,Amy Belding Brown
Flight: A Novel,Sherman Alexie
Flint: A Novel,Louis L'Amour
Flirt: The Interviews,Lorna Jackson
Flood of Fire: A Novel (The Ibis Trilogy),Amitav Ghosh
Flower Fairies of the Autumn,Cicely Mary Barker
Flowers and Stone,Jan Sikes
Flowers for Algernon,Daniel Keyes
Flowers in the Attic (Dollanganger),V.C. Andrews
Flowers in the Attic /  Petals on the Wind / If There Be Thorns / Seeds of Yesterday / Garden of Shadows,Virginia Andrews
Fludd: A Novel,Hilary Mantel
Fly Away,Kristin Hannah
Flyin' West and Other Plays,Pearl Cleage
Flying Changes: A Novel (Riding Lessons),Sara Gruen
Flying Colours (Hornblower Saga),C. S. Forester
Flying Too High : a Phryne Fisher Mystery,Kerry Greenwood
Folk Medicine in Southern Appalachia,Anthony Cavender
Folk Medicine: A Vermont Doctor's Guide to Good Health,D. C. Jarvis
"Folktales and Fairy Tales [4 volumes]: Traditions and Texts from around the World, 2nd Edition",Donald Haase Ph.D.
"Folktales on Stage: Children's Plays for Readers Theater, with 16 Reader's Theatre Play Scripts from World Folk and Fairy Tales and Legends, Including Asian, African, Middle Eastern, & Native American",Aaron Shepard
Follies of God: Tennessee Williams and the Women of the Fog,James Grissom
Fool for Love - Acting Edition,Sam Shepard
Fool for Love and Other Plays,Sam Shepard
Fools Crow (Penguin Classics),James Welch
For All My Walking,Santoka Taneda
For Love of the Game,Michael Shaara
For One More Day Large Print Edition,Mitch Albom
For the Love of 2am: Poetry For Insomniacs,Zena A. White
For the Sake of Love (Urban Books),Dwan Abrams
For Today I Am a Boy,Kim Fu
For Whom the Bell Tolls,Ernest Hemingway
For Whom The Bell Tolls,Ernest Hemingway
For Your Love: A Blessings Novel,Beverly Jenkins
Forbidden Acts: Pioneering Gay & Lesbian Plays of the 20th Century,Ben Hodges
Force Majeure: A Novel,Bruce Wagner
Fore!: The Best of Wodehouse on Golf (P.G. Wodehouse Collection),P. G. Wodehouse
"Foreign Gods, Inc.",Okey Ndibe
Foreign Influence: A Thriller (The Scot Harvath Series),Brad Thor
Forensics Duo Series Volume 4: Duo Practice and Competition Thirty-five 8-10 Minute Original Dramatic Plays for Two Females,Ira Brodsky
Forest Primeval: Poems,Vievee Francis
Forever a Hustler's Wife: A Novel (Nikki Turner Original),Nikki Turner
Forever an Ex: A Novel,Victoria Christopher Murray
Forever Human,Tom Conyers
"Forever, Interrupted: A Novel",Taylor Jenkins Reid
Forgiven (Urban Christian),Vanessa Miller
Forgotten (Forsaken) (Volume 3),Vanessa Miller
Forgotten Country,Catherine Chung
Form Line of Battle! (The Bolitho Novels) (Volume 9),Alexander Kent
Forrest Gump,Winston Groom
Forsaken (Urban Books),Vanessa Miller
Fortune Smiles: Stories,Adam Johnson
Forty Stories (Vintage Classics),Anton Chekhov
Foundations Of The Republic: Speeches And Addresses,Calvin Coolidge
Four Comedies,Aristophanes
Four Comedies: The Braggart Soldier; The Brothers Menaechmus; The Haunted House; The Pot of Gold (Oxford World's Classics),Plautus
"Four Greek Plays: The Agamemnon, The Oedipus Rex, The Alcestis, The Birds",Aeschylus and Sophocles
"Four Major Plays: Lysistrata, The Acharnians, The Birds, The Clouds",Aristophanes
Four Plays,Samuel D. Hunter
Four Plays by Aristophanes: The Birds; The Clouds; The Frogs; Lysistrata (Meridian Classics),Aristophanes
"Four Tragedies: Ajax, Women of Trachis, Electra, Philoctetes",Sophocles
Four-Legged Girl: Poems,Diane Seuss
Fourplay: A Novel,Jane Moore
Fourth of July Creek: A Novel,Smith Henderson
Foxfire 10,Inc. Foxfire Fund
Foxfire 9,Inc. Foxfire Fund
Fragile Things: Short Fictions and Wonders,Neil Gaiman
"Fragments: Poems, Intimate Notes, Letters",Marilyn Monroe
Francesco: Una vida entre el cielo y la tierra (Spanish Edition),Yohana Garca
Francis Bacon: The Major Works (Oxford World's Classics),Francis Bacon
Frank N' Goat: A Tale of Freakish Friendship,Jessica Watts
Frankenstein,Mary Shelley
Frankenstein (Hardcover Classics),Mary Shelley
Frankenstein (Second Edition)  (Norton Critical Editions),Mary Shelley
Frankenstein Makes a Sandwich,Adam Rex
Frankenstein: (Penguin Classics Deluxe Edition),Mary Shelley
Frankenstein: IT Support,James Livingood
Frankie and Johnny in the Claire de Lune,Terrence McNally
Franny and Zooey,J. D. Salinger
Franz Kafka: The Complete Stories,Franz Kafka
Freaks I've Met,Donald Jans
"Frederick Law Olmsted: Writings on Landscape, Culture, and Society: (Library of America #270)",Frederick Law Olmsted
Free to Be...You and Me (The 35th Anniversary Edition),Marlo Thomas and Friends
Freedom from Fear: And Other Writings,Aung San Suu Kyi
Freedom Time: The Poetics and Politics of Black Experimental Writing (The  Callaloo African Diaspora Series),Anthony Reed
Freedom: A Novel (Oprah's Book Club),Jonathan Franzen
Freedom's Battle Being A Comprehensive Collection Of Writings And Speeches On The Present Situation,Mahatma Gandhi
Freeman,Leonard Jr. Pitts
Freezer Burn,Joe R Lansdale
French Grammar (Quickstudy: Academic),Inc. BarCharts
Freud's Rome: Psychoanalysis and Latin Poetry (Roman Literature and its Contexts),Ellen Oliensis
Friday Night Love (Days Of Grace V1),Tia McCollors
Fried Green Tomatoes at the Whistle Stop Cafe: A Novel,Fannie Flagg
Friends & Foes,ReShonda Tate Billingsley
Friends with Full Benefits,Luke Young
Friends With Partial Benefits (Friends With... Benefits Series (Book 1)),Luke Young
Frog: A Novel,Mo Yan
Frogs and Other Plays (Penguin Classics),Aristophanes
"From Distant Days: Myths, Tales, and Poetry of Ancient Mesopotamia",Benjamin R. Foster
"From Hitler's Doorstep: The Wartime Intelligence Reports of Allen Dulles, 1942-1945",Neal H. Petersen
From Olympus to Camelot: The World of European Mythology,David Leeming
From Russia with Love (James Bond Series),Ian Fleming
From Slave to Governor: the Unlikely Life of Lott Cary,Perry Thomas
From the Cincinnati Reds to the Moscow Reds: The Memoirs of Irwin Weil (Jews of Russia & Eastern Europe and Their Legacy),Irwin Weil
From the Dissident Right II: Essays 2013,John Derbyshire
From the Listening Hills: Stories,Louis L'Amour
From the New World: Poems 1976-2014,Jorie Graham
From the Soul: My Haiku and My Senryu,Opal F. Caleb
Frostgrave: Thaw of the Lich Lord,Joseph A. McCullough
Frozen Socks: New & Selected Short Poems,Alan Pizzarelli
Fucked Up Shit: A Mixtape Anthology,Berti Walker
Fugitive Colors: A Novel,Lisa Barr
Full Black: A Thriller (The Scot Harvath Series),Brad Thor
Full Circle (Urban Books),Skyy
"Full Dark, No Stars",Stephen King
Full Force and Effect (Jack Ryan),Mark Greaney
Fun Home,Jeanine Tesori
Funny 4 God: A Variety of Christian Comedy Skits,Rick Eichorn
Funny Girl: A Novel,Nick Hornby
"Gabriel Marcel's Perspectives on the Broken World: The Broken World, a Four-Act Play : Followed by Concrete Approaches to Investigating the Ontological Mystery (Marquette Studies in Philosophy)",Gabriel Marcel
Gabriel: A Poem,Edward Hirsch
Gaia Codex,Sarah Drew
"Galateo: Or, The Rules of Polite Behavior",Giovanni Della Casa
Galore,Michael Crummey
"Gambled: A Novella: Titan, Book 3.25",Cristin Harber
Game Seven,Tom Rock
Ganesha Goes to Lunch: Classics from Mystic India (Mandala Classics),Kamla K. Kapur
Gangsta (Urban Books),K'wan
Garden of Shadows (Dollanganger),V.C. Andrews
Garden Spells (Bantam Discovery),Sarah Addison Allen
Gardenias: A Novel,Faith Sullivan
GARDENS IN THE DUNES: A Novel,Leslie Marmon Silko
Gate of the Sun,Elias Khoury
Gates of Fire,Steven Pressfield
Gates of Fire: An Epic Novel of the Battle of Thermopylae,Steven Pressfield
Gates of Gold,Frank McGuinness
Gathering of Waters,Bernice L. McFadden
Gaunt's Ghosts: The Founding,Dan Abnett
Gem of the Ocean,August Wilson
Gemini: A Novel,Carol Cassella
Gems of Gemvron: Five Onyx Stones,Mr Michael J Murtuagh
Gender Trouble: Feminism and the Subversion of Identity (Routledge Classics),Judith Butler
Genealogical Fictions: Cultural Periphery and Historical Change in the Modern Novel,Jobst Welge
Genealogy of the Tragic: Greek Tragedy and German Philosophy,Joshua Billings
General A. P. Stewart: His Life And Letters,Marshall Wingfield
Genesis Revisited (Earth Chronicles),Zecharia Sitchin
George,Alex Gino
George MacDonald: An Anthology 365 Readings,George MacDonald
"George R. R. Martin's A Game of Thrones Leather-Cloth Boxed Set (Song of Ice and Fire Series): A Game of Thrones, A Clash of Kings, A Storm of Swords, A Feast for Crows, and A Dance with Dragons",George R. R. Martin
George Washington Gomez: A Mexicotexan Novel,Americo Paredes
Georgette Heyer,Jennifer Kloester
Georgette Heyer's Regency World,Jennifer Kloester
Gerard Manley Hopkins: The Major Works (Oxford World's Classics),Gerard Manley Hopkins
Geronimo Rex,Barry Hannah
Getting Back,Kelly Sinclair
Getting Over Kyle: Second Chances Series Book II (Volume 2),A. W. Myrie
Getting to Happy,Terry McMillan
Gettysburg Address and Other Writings,Abraham Lincoln
Ghetto Bastard (Animal),K'wan
Ghost Ship (The NUMA Files),Clive Cussler
Ghostly: A Collection of Ghost Stories,Audrey Niffenegger
Ghetto Love 4 (Volume 4),Sonovia Alexander
Ghost Medicine: An Ella Clah Novel,Aime Thurlo
Ghost of the Machine (ShatterGlass and WinterHeld) (Volume 1),Diana N. Logg
Ghosts of Bungo Suido,P. T. Deutermann
Ghosts of Manitowish Waters,G. M. Moore
"Giambattista Basile's The Tale of Tales, or Entertainment for Little Ones (Series in Fairy-Tale Studies)",Giambattista Basile
Gilead: A Novel,Marilynne Robinson
Gilgamesh: A Verse Narrative,Herbert Mason
Gillian's Island,Natalie Vivien
Giovanni's Room (Vintage International),James Baldwin
Girl at War: A Novel,Sara Novic
Girl in Translation,Jean Kwok
Girl Singer,Mick Carlon
Girlchild: A Novel,Tupelo Hassman
Glengarry Glen Ross: A Play,David Mamet
Glew I: Maneater (GLEW: the horse that eats people) (Volume 1),Michael R Peterson MP
Glimmer in the Darkness (Forgiveness),Nicole Hampton
Glitterwolf Magazine: Halloween Special,Matt Cresswell
Glorious,Bernice L. McFadden
Glorious: A Novel of the American West,Jeff Guinn
Go Set a Watchman: A Novel,Harper Lee
Go Tell It on the Mountain (Vintage International),James Baldwin
Go the F**k to Sleep,Adam Mansbach
God Don't Like Ugly,Mary Monroe
God Says No,James Hannaham
God's Country,Percival Everett
God's Formula,James Lepore
God's Kingdom: A Novel,Howard Frank Mosher
Goddess,Kelly Gardiner
Gods and Generals: A Novel of the Civil War (Civil War Trilogy),Jeff Shaara
Gods and Heroes of Ancient Greece (The Pantheon Fairy Tale and Folklore Library),Gustav Schwab
Gods and Kings (Chronicles of the Kings #1) (Volume 1),Lynn Austin
Gods and Myths of Northern Europe,H.R. Ellis Davidson
Gods Behaving Badly: A Novel,Marie Phillips
Going Home: A Novel of the Civil War,James D. Shipman
Going to Meet the Man: Stories,James Baldwin
Gold: (Poiema Poetry),Barbara Crooker
Gold: A Novel,Chris Cleave
Golden Age: A novel (Last Hundred Years Trilogy),Jane Smiley
Golden State: A Novel,Michelle Richmond
Goldilocks and the Three Bears: Special Edition,Robert Southey
Golf in the Kingdom,Michael J. Murphy
GOLF MAGAZINE How To Hit Every Shot,Editors of Golf Magazine
GOLF MAGAZINE'S BIG BOOK OF BASICS: Your step-by-step guide to building a complete and reliable game from the ground up WITH THE TOP 100 TEACHERS IN AMERICA,GOLF Magazine
GOLF Magazine's The Par Plan: A Revolutionary System to Shoot Your Best Score in 30 Days,GOLF Magazine
Golf: The Best Short Game Instruction Book Ever!,Editors of Golf Magazine
Golfing with God: A Novel of Heaven and Earth,Roland Merullo
Gone (Hannah Smith Novels),Randy Wayne White
Good Bones and Simple Murders,Margaret Atwood
"Good Dog: True Stories of Love, Loss, and Loyalty",Editors of Garden and Gun
Good Faeries/Bad Faeries,Brian Froud
Good People,David Lindsay-Abaire
Goth,Otsuichi
Gothic Tales (Penguin Classics),Elizabeth Gaskell
Grace - Acting Edition,Craig Wright
Grace: A Novel,Richard Paul Evans
Graceland (Today Show Pick January 2005),Chris Abani
Graceland and Asleep on the Wind,Ellen Byron
Graffiti and the Literary Landscape in Roman Pompeii,Kristina Milnor
"Graham R.: Rosamund Marriott Watson, Woman of Letters",Linda K. Hughes
Grand Central: Original Stories of Postwar Love and Reunion,Karen White
Grand Opening: A Family Business Novel (Family Business Novels),Carl Weber
Graphic Women: Life Narrative and Contemporary Comics (Gender and Culture Series),Hillary L. Chute
Grass Crown (Masters of Rome),Colleen McCullough
Gratitude,Oliver Sacks
Grave on Grand Avenue (An Officer Ellie Rush Mystery),Naomi Hirahara
Gravity,Tess Gerritsen
Gray,Pete Wentz
Great Books of the Western World,Mortimer J. Adler
Great Classic Stories: 22 Unabridged Classics,Derek Jacobi
Great Expectations (Dover Thrift Editions),Charles Dickens
Great Hunting Stories: Inspiring Adventures for Every Hunter,Steve Chapman
Great Short Works of Fyodor Dostoevsky (Harper Perennial Modern Classics),Fyodor Dostoevsky
Great Speeches (Dover Thrift Editions),Franklin Delano Roosevelt
Great Speeches of our Time,Hywel Williams
"Great Wilderness, A",Samuel D Hunter
Great with Child: Letters to a Young Mother,Beth Ann Fennelly
"Greece, in 1823 and 1824: Being a Series of Letters and Other Documents on the Greek Revolution, Written during a Visit to that Country (Cambridge Library Collection - European History)",Leicester Stanhope
Greek Drama (Bantam Classics),Moses Hadas
Greek Lexicon of the Roman and Byzantine Periods from B.C. 146 to A.D. 1100 V1,E. A. Sophocles
Greek Lexicon of the Roman and Byzantine Periods from B.C. 146 to A.D. 1100 V2,E. A. Sophocles
Greek Lyric Poetry (Bcp Greek Texts),David A. Campbell
"Greek Lyric, Volume III, Stesichorus, Ibycus, Simonides, and Others (Loeb Classical Library No. 476)",Stesichorus
Greek Lyric: Sappho and Alcaeus (Loeb Classical Library No. 142) (Volume I),Sappho
"Greek Tragedies, Volume 1",Aeschylus
"Greek Tragedies, Volume 2 The Libation Bearers (Aeschylus), Electra (Sophocles), Iphigenia in Tauris, Electra, & The Trojan Women (Euripides)",Aeschylus
"Greek Tragedy on the American Stage: Ancient Drama in the Commercial Theater, 1882-1994 (Contributions in Drama and Theatre Studies)",Karelisa Hartigan
Green Calder Grass,Janet Dailey
Green Eyes,Ari Eastman
Green Girl: A Novel (P.S.),Kate Zambreno
"Green Grass, Running Water",Thomas King
Green Hills of Africa,Ernest Hemingway
Green Skull and Crossbones Journal: 160 Page Lined Journal/Notebook,Mahtava Journals
Greenbeard,Richard James Bentley
Grey: Fifty Shades of Grey as Told By Christian By E L James | Summary & Analysis,Quick Read
Grief Lessons: Four Plays by Euripides (New York Review Books Classics),Euripides
Grimm's Complete Fairy Tales,Jacob Grimm
Grimm's Fairy Tales - Illustrated by Charles Folkard,Grimm Brothers
Growing Older with Jane Austen,Maggie Lane
Grown Folks Business: A Novel,Victoria Christopher Murray
Gruesome Playground Injuries,Rajiv Joseph
Gruesome Playground Injuries; Animals Out of Paper; Bengal Tiger at the Baghdad Zoo: Three Plays,Rajiv Joseph
Guerra y Paz (Spanish Edition),Len Tolsti
Gulliver's Travels,Jonathan Swift
Gulliver's Travels,Martin Rowson
Gulliver's Travels (Calla Editions),Jonathan Swift
Gulliver's Travels: A Signature Performance by David Hyde Pierce,Jonathan Swift
Gutenberg's Apprentice: A Novel,Alix Christie
Gutter: A Novel,K'wan
Gtz and Meyer: G?tz and Meyer (Serbian Literature),David Albahari