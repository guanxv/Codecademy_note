

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------PYTHON RECURSION PYTHON RECURSION PYTHON RECURSION PYTHON RECURSION PYTHON RECURSION---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Call Stacks and Execution Frames
A recursive approach requires the function invoking itself with different arguments. How does the computer keep track of the various arguments and different function invocations if it’s the same function definition?

Repeatedly invoking functions may be familiar when it occurs sequentially, but it can be jarring to see this invocation occur within a function definition.

Languages make this possible with call stacks and execution contexts.

Stacks, a data structure, follow a strict protocol for the order data enters and exits the structure: the last thing to enter is the first thing to leave.

Your programming language often manages the call stack, which exists outside of any specific function. This call stack tracks the ordering of the different function invocations, so the last function to enter the call stack is the first function to exit the call stack

we can think of execution contexts as the specific values we plug into a function call.

A function which adds two numbers:

Invoking the function with 3 and 4 as arguments...
execution context:
X = 3
Y = 4

Invoking the function with 6 and 2 as arguments...
execution context:
X = 6
Y = 2
Consider a pseudo-code function which sums the integers in an array:

 function, sum_list 
   if list has a single element
     return that single element
   otherwise...
     add first element to value of sum_list called with every element minus the first
This function will be invoked as many times as there are elements within the list! Let’s step through:

psudo-code

Sum_list(arr):
    if arr empty ?:
        return 0
    return first + sum_list(arr-1)


CALL STACK EMPTY
___________________

Our first function call...
sum_list([5, 6, 7])

CALL STACK CONTAINS
___________________
sum_list([5, 6, 7])
with the execution context of a list being [5, 6, 7]
___________________

Base case, a list of one element not met.
We invoke sum_list with the list of [6, 7]...

CALL STACK CONTAINS
___________________
sum_list([6, 7])
with the execution context of a list being [6, 7]
___________________
sum_list([5, 6, 7])
with the execution context of a list being [5, 6, 7]
___________________

Base case, a list of one element not met.
We invoke sum_list with the list of [7]...

CALL STACK CONTAINS
___________________
sum_list([7])
with the execution context of a list being [7]
___________________
sum_list([6, 7])
with the execution context of a list being [6, 7]
___________________
sum_list([5, 6, 7])
with the execution context of a list being [5, 6, 7]
___________________

We've reached our base case! List is one element. 
We return that one element.
This return value does two things:

1) "pops" sum_list([7]) from CALL STACK.
2) provides a return value for sum_list([6, 7])

----------------
CALL STACK CONTAINS
___________________
sum_list([6, 7])
with the execution context of a list being [6, 7]
RETURN VALUE = 7
___________________
sum_list([5, 6, 7])
with the execution context of a list being [5, 6, 7]
___________________

sum_list([6, 7]) waits for the return value of sum_list([7]), which it just received. 

sum_list([6, 7]) has resolved and "popped" from the call stack...


----------------
CALL STACK contains
___________________
sum_list([5, 6, 7])
with the execution context of a list being [5, 6, 7]
RETURN VALUE = 6 + 7
___________________

sum_list([5, 6, 7]) waits for the return value of sum_list([6, 7]), which it just received. 
sum_list([5, 6, 7]) has resolved and "popped" from the call stack.


----------------
CALL STACK is empty
___________________
RETURN VALUE = (5 + 6 + 7) = 18





RECURSION: CONCEPTUAL
Base Case and Recursive Step
Recursion has two fundamental aspects: the base case and the recursive step.

When using iteration, we rely on a counting variable and a boolean condition. For example, when iterating through the values in a list, we would increment the counting variable until it exceeded the length of the dataset.

Recursive functions have a similar concept, which we call the base case. The base case dictates whether the function will recurse, or call itself. Without a base case, it’s the iterative equivalent to writing an infinite loop.

Because we’re using a call stack to track the function calls, your computer will throw an error due to a stack overflow if the base case is not sufficient.

The other fundamental aspect of a recursive function is the recursive step. This portion of the function is the step that moves us closer to the base case.

In an iterative function, this is handled by a loop construct that decrements or increments the counting variable which moves the counter closer to a boolean condition, terminating the loop.

In a recursive function, the “counting variable” equivalent is the argument to the recursive call. If we’re counting down to 0, for example, our base case would be the function call that receives 0 as an argument. We might design a recursive step that takes the argument passed in, decrements it by one, and calls the function again with the decremented argument. In this way, we would be moving towards 0 as our base case.

Analyzing the Big O runtime of a recursive function is very similar to analyzing an iterative function. Substitute iterations of a loop with recursive calls.

For example, if we loop through once for each element printing the value, we have a O(N) or linear runtime. Similarly, if we have a single recursive call for each element in the original function call, we have a O(N) or linear runtime.'''


'''RECURSION: PYTHON
Building Our Own Call Stack
The best way to understand recursion is with lots of practice! At first, this method of solving a problem can seem unfamiliar but by the end of this lesson, we’ll have implemented a variety of algorithms using a recursive approach.

Before we dive into recursion, let’s replicate what’s happening in the call stack with an iterative function.

The call stack is abstracted away from us in Python, but we can recreate it to understand how recursive calls build up and resolve.

Let’s write a function that sums every number from 1 to the given input.

sum_to_one(4)
# 10
sum_to_one(11)
# 66
To depict the steps of a recursive function, we’ll use a call stack and execution contexts for each function call.

The call stack stores each function (with its internal variables) until those functions resolve in a last in, first out order.

call_stack = []
recursive_func()
call_stack = [recursive_func_1]

# within the body of recursive_func, another call to recursive_func()
call_stack = [recursive_func_1, recursive_func_2]
# the body of the second call to recursive_func resolves...
call_stack = [recursive_func_1]
# the body of the original call to recursive_func resolves...
call_stack = [] 
Execution contexts are a mapping between variable names and their values within the function during execution. We can use a list for our call stack and a dictionary for each execution context.

Let’s get started!

Instructions
1.
Define a sum_to_one() function that has n as the sole parameter.

Inside the function body:

declare the variable result and set it to 1.
declare the variable call_stack and set it to an empty list.
Use multiple return to return both of these values: result, call_stack

You can return multiple values like so:

def two_things(a, b):
  return a, b

first, second = two_things("apple", "pie")
first # "apple"
second # "pie"
2.
Fill in the sum_to_one() function body by writing a while loop after the variable call_stack.

This loop represents the recursive calls which lead to a base case.

We’ll want to loop until the input n reaches 1.

Inside the loop, create a variable execution_context and assign it to a dictionary with the key of "n_value" pointing to n.

Use a list method to add execution_context to the end of call_stack.

This is our way of simulating the recursive function calls being “pushed” onto the system’s call stack.

Decrement n after its value has been stored.

End the loop by printing call_stack.

We’re using a Python dictionary to represent the execution context of each recursive call.

Here’s how it would look if the execution context captured a value foo:

call_stack = []
while foo != 1:
  execution_context = {"foo_value": foo}
  call_stack.append(execution_context)
  foo -= 1
3.
After the while loop concludes, we’ve reached our “base case”, where n == 1.

At this point we haven’t summed any values, but we have all the information we’ll need stored in our call_stack.

In the next exercise, we’ll handle the summation of values from the execution contexts captured in call_stack.

For now, print out “BASE CASE REACHED” outside of the loop block before our multiple return statement.
'''
# define your sum_to_one() function above the function call
'''
def sum_to_one(num):

  if num == 0:
    return 0
  return num + sum_to_one(num-1)
'''

def sum_to_one(n):
  result = 1
  call_stack = []
  while n > 1:
    execution_context = {'n_value':n}
    call_stack.append(execution_context)
    n -= 1
    print(call_stack)  
  print('BASE CASE REACHED')
  return result, call_stack



print(sum_to_one(4))

'''

RECURSION: PYTHON
Building Our Own Call Stack, Part II
In the previous exercise, we used an iterative function to implement how a call stack accumulates execution contexts during recursive function calls.

We’ll now address the conclusion of this function, where the separate values stored in the call stack are accumulated into a single return value.

Instructions
1.
This is the point in a recursive function when we would begin returning values as the function calls are “popped” off the call stack.

We’ll use another while loop to simulate this process. Write the while loop below the “Base Case Reached” print statement.

This loop will run as long as there are execution contexts stored in call_stack.

Inside this second loop:

declare the variable return_value
assign the last element in call_stack to return_value.
Remove that value from call_stack otherwise you’ll have an infinite loop!
Print call_stack to see how the execution contexts are removed from call_stack.

This loop runs as long as there are elements within call_stack.

We’ll also want to remove values from call_stack in each iteration.

You can set this as a condition on a loop like so:

cats = ['buffy', 'wampus', 'felix']

while len(cats) != 0:
  cat = cats.pop()
  print(cat)

# 'felix'
# 'wampus'
# 'buffy'

2.
Print that you’re adding return_value["n_value"] to result and their respective values.

Finish the loop by retrieving "n_value" from return_value and add it to result.

Each element we .pop() from call_stack is a dictionary which represents a single recursive function call. We can access the value stored in the dictionary with the key like so:

execution_context = {"my_value": 42}
execution_context["my_value"]
# 42'''

def sum_to_one(n):
  result = 1
  call_stack = []
  
  while n != 1:
    execution_context = {"n_value": n}
    call_stack.append(execution_context)
    n -= 1
    print(call_stack)
  print("BASE CASE REACHED")

  return_value = 0
  while call_stack != []:
    
    return_value += call_stack[-1]["n_value"]
    call_stack.pop(-1)
    print(return_value)

  result += return_value

  return result, call_stack

sum_to_one(4)

'''
RECURSION: PYTHON
Sum to One with Recursion
Now that we’ve built a mental model for how recursion is handled by Python, let’s implement the same function and make it truly recursive.

To recap: We want a function that takes an integer as an input and returns the sum of all numbers from the input down to 1.

sum_to_one(4)
# 4 + 3 + 2 + 1
# 10
Here’s how this function would look if we were to write it iteratively:

def sum_to_one(n):
  result = 0
  for num in range(n, 0, -1):
    result += num
  return result

sum_to_one(4)
# num is set to 4, 3, 2, and 1
# 10
We can think of each recursive call as an iteration of the loop above. In other words, we want a recursive function that will produce the following function calls:

recursive_sum_to_one(4)
recursive_sum_to_one(3)
recursive_sum_to_one(2)
recursive_sum_to_one(1)
Every recursive function needs a base case when the function does not recurse, and a recursive step, when the recursing function moves towards the base case.

Base case:

The integer given as input is 1.
Recursive step:

The recursive function call is passed an argument 1 less than the last function call.
Instructions
1.
Define the sum_to_one() function.

It takes n as the sole parameter.

We’ll start by setting up our base case.

This function should NOT recurse if the given input, n is 1.

In the base case, we return n.

2.
Now, we’ll consider the recursive step.

We want our return value to be the current input added to the return value of sum_to_one().

We also need to invoke sum_to_one() with an argument that will get us closer to the base case.

# return {recursive call} + {current input}
This should be a single line solution:

return argument + recursive_call(argument - 1)
3.
Each recursive call is responsible for adding one of those integers to the ultimate total.

To help us visualize the different function calls, add a print statement before the recursive call that tells us the current value of n.

Use the following string for the print statement: print("Recursing with input: {0}".format(n))

Let’s test out our function. Call sum_to_one() with 7 as input and print out the result. Nice work!

If we try to print the value before the return statement, we’ll never see it!


def sum_to_one(n):
  if n == 1:
    return n
  else:
    return n + sum_to_one(n - 1)
    print("You will never see meeeeeee!")'''
    
# Define sum_to_one() below...
def sum_to_one(n):
  print("Recursing with input: {0}".format(n))
  if n == 1:
    return n
  return sum_to_one(n-1) + n

# uncomment when you're ready to test
print(sum_to_one(7))

'''RECURSION: PYTHON
Recursion and Big O
Excellent job writing your first recursive function. Our next task may seem familiar so there won’t be as much guidance.

We’d like a function factorial that, given a positive integer as input, returns the product of every integer from 1 up to the input. If the input is less than 2, return 1.

For example:

factorial(4)
# 4 * 3 * 2 * 1
# 24
Since this function is similar to the previous problem, we’ll add an additional wrinkle. You’ll need to evaluate the big O runtime of the function.

With an iterative function, we would consider how the number of iterations grows in relation to the size of the input.

For example you may ask yourself, are we looping once more for each new element in the list?

That’s linear or O(N).

Are we looping an additional number of elements in the list for each new element in the list?

That’s quadratic or O(N^2).

With recursive functions, the thought process is similar but we’re replacing loop iterations with recursive function calls.

In other words, are we recursing once more for each new element in the list?

That’s linear or O(N).

Let’s analyze our previous function, sum_to_one().

sum_to_one(4)
# recursive call to sum_to_one(3)
# recursive call to sum_to_one(2)
# recursive call to sum_to_one(1)

# Let's increase the input...

sum_to_one(5)
# recursive call to sum_to_one(4)
# recursive call to sum_to_one(3)
# recursive call to sum_to_one(2)
# recursive call to sum_to_one(1)
What do you think? We added one to the input, how many more recursive calls were necessary?

Talk through a few more inputs and then start coding when you’re ready to move on.

Instructions
1.
Define the factorial function with one parameter: n.

Set up a base case.

Think about the input(s) that wouldn’t need a recursive call for your function.

Return the appropriate value.

Factorial numbers are the total product of every number from 1 to a given input.

With 0 or 1, we don’t need any other number to compute the factorial.

if n <= 1:
  return 1
2.
Now let’s consider the recursive step for factorial().

If we’re in the recursive step that means factorial() has been invoked with an integer of at least 2.

We need to return the current input value multiplied by the return value of the recursive call.

Structure the recursive call so it invokes factorial() with an argument one less than the current input.

To compute factorial(3), we’d need factorial(2) * factorial(1). Each recursive call decrements the argument by one.

Your recursive step should look like the following:

return # {current input} * factorial(current input minus one)
3.
Nice work, test out your function by printing the result of calling factorial() with 12 as an input.

Now, change the input to a really large number, think big, and run the code.

If you chose an input large enough, you should see a RecursionError.'''

# Define factorial() below:
def factorial(n):
  print("factorial Called Once.")
  if n == 1:
    return n
  
  return n * factorial(n-1)

print(factorial(6))

'''RECURSION: PYTHON
Stack Over-Whoa!
The previous exercise ended with a stack overflow, which is a reminder that recursion has costs that iteration doesn’t. We saw in the first exercise that every recursive call spends time on the call stack.

Put enough function calls on the call stack, and eventually there’s no room left.

Even when there is room for any reasonable input, recursive functions tend to be at least a little less efficient than comparable iterative solutions because of the call stack.

The beauty of recursion is how it can reduce complex problems into an elegant solution of only a few lines of code. Recursion forces us to distill a task into its smallest piece, the base case, and the smallest step to get there, the recursive step.

Let’s compare two solutions to a single problem: producing a power set. A power set is a list of all subsets of the values in a list.

This is a really tough algorithm. Don’t be discouraged!

power_set(['a', 'b', 'c'])
# [
#   ['a', 'b', 'c'], 
#   ['a', 'b'], 
#   ['a', 'c'], 
#   ['a'], 
#   ['b', 'c'], 
#   ['b'], 
#   ['c'], 
#   []
# ]
Phew! That’s a lot of lists! Our input length was 3, and the list returned had a length of 8.

Producing subsets requires a runtime of at least O(2^N), we’ll never do better than that because a set of N elements creates a power set of 2^N elements.

Binary, a number system of base 2, can represent 2^N numbers for N binary digits. For example:

# 1 binary digit, 2 numbers
# 0 in binary
0
# 1 in binary
1

# 2 binary digits, 4 numbers
# 00 => 0
# 01 => 1
# 10 => 2
# 11 => 3
The iterative approach uses this insight for a very clever solution by including an element in the subset if its “binary digit” is 1.

set = ['a', 'b', 'c']
binary_number = "101"
# produces the subset ['a', 'c']
# 'b' is left out because its binary digit is 0
That process is repeated for all O(2^N) numbers!

Here is the complete solution. You’re not expected to understand every line, just take in the level of complexity.

def power_set(set):
  power_set_size = 2**len(set)
  result = []

  for bit in range(0, power_set_size):
    sub_set = []
    for binary_digit in range(0, len(set)):
      if((bit & (1 << binary_digit)) > 0):
        sub_set.append(set[binary_digit])
    result.append(sub_set)
  return result
Very clever but not very intuitive! Let’s try recursion.

Consider the base case, where the problem has become so simple we can solve it without doing any work.

What’s the simplest power set possible? An empty list!

power_set([])
# [[]]
Now the recursive step. We need to progress towards our base case, an empty list, so we’ll be removing an element from the input.

Examine the simplest powerset that isn’t the base case:

power_set(['a'])
# [[], ['a']]
A power set in the recursive step requires:

all subsets which contain the element
in this case "a"
all subsets which don’t contain the element
in this case [].
With the recursive approach, we’re able to articulate the problem in terms of itself. No need to bring in a whole number system to find the solution!

Here’s the recursive solution in its entirety:

def power_set(my_list):
  if len(my_list) == 0:
    return [[]]
  power_set_without_first = power_set(my_list[1:])
  with_first = [ [my_list[0]] + rest for rest in power_set_without_first ]
  return with_first + power_set_without_first
Neither of these solutions is simple, this is a complicated algorithm, but the recursive solution is almost half the code and more directly conveys what this algorithm does.

Give yourself a pat on the back for making it through a tough exercise!

Instructions
Run the code to see subsets of universities.

Try adding your own school.

See how large you can make the input list before these computations become impossibly slow…

O(2^N) runtime is no joke!'''

def power_set(my_list):
    # base case: an empty list
    if len(my_list) == 0:
        return [[]]
    # recursive step: subsets without first element
    power_set_without_first = power_set(my_list[1:])
    # subsets with first element
    with_first = [ [my_list[0]] + rest for rest in power_set_without_first ]
    # return combination of the two
    return with_first + power_set_without_first

#Power_set Function made with irterate method.
def ipower_set(set):
  power_set_size = 2**len(set)
  result = []

  for bit in range(0, power_set_size):
    sub_set = []
    for binary_digit in range(0, len(set)):
      if((bit & (1 << binary_digit)) > 0):
        sub_set.append(set[binary_digit])
    result.append(sub_set)
  return result
  
universities = ['MIT', 'UCLA', 'Stanford', 'NYU', 'BIM','DU']
power_set_of_universities = power_set(universities)
#print(ipower_set(universities))

for set in power_set_of_universities:
  print(set)

#note for shifts operation

a = 5
print( a << 10) # = 5 * pow(2, 10) = 5 * 2**10
'''These operators accept integers as arguments. They shift the first argument to the left or right by the number of bits given by the second argument.

A right shift by n bits is defined as floor division by pow(2,n). A left shift by n bits is defined as multiplication with pow(2,n).
Shift Operation'''


'''
RECURSION: PYTHON
No Nested Lists Anymore, I Want Them to Turn Flat
Let’s use recursion to solve another problem involving lists: flatten().

We want to write a function that removes nested lists within a list but keeps the values contained.

nested_planets = ['mercury', 'venus', ['earth'], 'mars', [['jupiter', 'saturn']], 'uranus', ['neptune', 'pluto']]

flatten(nested_planets)
# ['mercury', 
#  'venus', 
#  'earth', 
#  'mars', 
#  'jupiter', 
#  'saturn', 
#  'uranus', 
#  'neptune', 
#  'pluto']
Remember our tools for recursive functions. We want to identify a base case, and we need to think about a recursive step that takes us closer to achieving the base case.

For this problem, we have two scenarios as we move through the list.

The element in the list is a list itself.
We have more work to do!
The element in the list is not a list.
All set!
Which is the base case and which is the recursive step?

Instructions
1.
Define flatten() which has a single parameter named my_list.

We’ll start by declaring a variable, result and setting it to an empty list.

result is our intermediary variable that houses elements from my_list.

Return result.

2.
Returning an empty list isn’t much good to us, it should be filled with the values contained in my_list.

Use a for loop to iterate through my_list.

Inside the loop, we need a conditional for our recursive step. Check if the element in the current iteration is a list.

We can use Python’s isinstance() like so:

a_list = ['listing it up!']
not_a_list = 'string here'

isinstance(a_list, list)
# True
isinstance(not_a_list, list)
# False
For now, print "List found!" in the conditional.

Outside of the method definition, call flatten() and pass planets as an argument.

Use isinstance(iteration_element, list).

Here’s an example:

my_list = ['apples', ['cherries'], 'bananas']

for element in my_list:
  if isinstance(element, list):
    print("this element is a list!")
  else:
    print(element)

# apples
# this element is a list!
# bananas
3.
We need to make the recursive step draw us closer to the base case, where every element is not a list.

After your print statement, declare the variable flat_list, and assign it to a recursive call to flatten() passing in your iterating variable as the argument.

flatten() will return a list, update result so it now includes every element contained in flat_list.

Test flatten() by calling it on the planets and printing the result.

We can combine two lists like so:

first_list = ['a', 'b', 'c']
second_list = ['d', 'e', 'f']
first_list + second_list
# ['a', 'b', 'c', 'd', 'e', 'f']
We can use this to update the result list like so:

result = ['a']
flat_list = ['b', 'c']
result += flat_list

result # ['a', 'b', 'c']
4.
Nice work! Now the base case.

If the iterating variable is not a list, we can update result, so it includes this element at the end of the list.

flatten() should now return the complete result.

Print the result!

Why is it important that the element is added at the end?

Let’s think through how these recursive calls will work in a simple case:

nested = ['green', 'red', ['blue', 'yellow'], 'purple']

flatten(nested)
# inside flatten()...
# result = ['green']
# result = ['green', 'red']
# recursive call! 
'''
    
# define flatten() below...
def xu_flatten(list):
  result = []
  if list == []:
    return
  
  if type(list[0]) == type(list):
    flatten(list)
  else:
    result.append(list[0])
    list.pop(0)
    flatten(list)
    
  return result

def flatten(my_list):
  result = []
  
  for item in my_list:
    if isinstance(item, list):
      print("List found!")
      result +=  flatten(item)
    else:
      result.append(item)
  return result


### reserve for testing...
planets = ['mercury', 'venus', ['earth'], 'mars', [['jupiter', 'saturn']], 'uranus', ['neptune', 'pluto']]

print(flatten(planets))

'''RECURSION: PYTHON
Fibonacci? Fibonaccu!
So far our recursive functions have all included a single recursive call within the function definition.

Let’s explore a problem which pushes us to use multiple recursive calls within the function definition.

Fibonacci numbers are integers which follow a specific sequence: the next Fibonacci number is the sum of the previous two Fibonacci numbers.

We have a self-referential definition which means this problem is a great candidate for a recursive solution!

We’ll start by considering the base case. The Fibonacci Sequence starts with 0 and 1 respectively. If our function receives an input in that range, we don’t need to do any work.

If we receive an input greater than 1, things get a bit trickier. This recursive step requires two previous Fibonacci numbers to calculate the current Fibonacci number.

That means we need two recursive calls in our recursive step. Expressed in code:

fibonacci(3) == fibonacci(1) + fibonacci(2) 
Let’s walk through how the recursive calls will accumulate in the call stack:

call_stack = []
fibonacci(3)
call_stack = [fibonacci(3)]
To calculate the 3rd Fibonacci number we need the previous two Fibonacci numbers. We start with the previous Fibonacci number.

fibbonacci(2)
call_stack = [fibbonacci(3), fibbonacci(2)]
fibonacci(2) is a base case, the value of 1 is returned…

call_stack = [fibbonacci(3)]
The return value of fibonacci(2) is stored within the execution context of fibonacci(3) while ANOTHER recursive call is made to retrieve the second most previous Fibonacci number…

fibonacci(1)
call_stack = [fibonacci(3), fibonacci(1)]
Finally, fibonacci(1) resolves because it meets the base case and the value of 1 is returned.

call_stack = [fibonacci(3)]
The return values of fibonacci(2) and fibonacci(1) are contained within the execution context of fibonacci(3), which can now return the sum of the previous two Fibonacci numbers.

As you can see, those recursive calls add up fast when we have multiple recursive invocations within a function definition!

Can you reason out the big O runtime of this Fibonacci function?

Instructions
1.
Define our fibonacci() function that takes n as an argument.

Let’s address our base cases:

if the input is 1, we return 1
if the input is 0, we return 0
While we should guard against faulty values, such as numbers below 0, we’ll keep our function simple.

The base case can be expressed a number of different ways, so get creative!

Here’s an example of a base case that checks for two conditions:

def is_odd_or_negative(num):
  if num % 2 != 0:
    return "It's odd!"
  if num < 0:
    return "It's negative!"
2.
Now take care of the recursive step.

This step involves summing two recursive calls to fibonacci().

We need to retrieve the second to last and last Fibonacci values and return their sum.

We can get the second to last Fibonacci by decrementing the input by 2 and the last by decrementing the input by 1.

Again, there are a few different ways we can do this, but a single line solution is what we’re looking for.

Something similar to this:

return recursive_call(second_to_last_num) + recursive_call(last_num)
3.
Add print statements within fibonacci() to explore the different recursive calls.

Set fibonacci_runtime to the appropriate big O runtime.

You’ll notice there are quite a bit of repeated function calls with the same input. This contributes to the expensive runtime…

Can you think of a way to make this function more efficient?

Here’s a hint. https://en.wikipedia.org/wiki/Memoization'''


# define the fibonacci() function below...
def fibonacci(n):
  print('run once')
  if n == 1:
    return 1
  if n == 0:
    return 0
  
  num = fibonacci(n-1) + fibonacci(n-2)

  return num



print(fibonacci(100))
# set the appropriate runtime:
# 1, logN, N, N^2, 2^N, N!
fibonacci_runtime = "2^N"

'''
RECURSION: PYTHON
Recursive Data Structures
Data structures can also be recursive.

Trees are a recursive data structure because their definition is self-referential. A tree is a data structure which contains a piece of data and references to other trees!

Trees which are referenced by other trees are known as children. Trees which hold references to other trees are known as the parents.

A tree can be both parent and child. We’re going to write a recursive function that builds a special type of tree: a binary search tree.

Binary search trees:

Reference two children at most per tree node.
The “left” child of the tree must contain a value lesser than its parent
The “right” child of the tree must contain a value greater than its parent.
Trees are an abstract data type, meaning we can implement our version in a number of ways as long as we follow the rules above.

For the purposes of this exercise, we’ll use the humble Python dictionary:

bst_tree_node = {"data": 42}
bst_tree_node["left_child"] = {"data": 36}
bst_tree_node["right_child"] = {"data": 73}

bst_tree_node["data"] > bst_tree_node["left_child"]["data"]
# True
bst_tree_node["data"] < bst_tree_node["right_child"["data"]
# True
We can also assume our function will receive a sorted list of values as input.

This is necessary to construct the binary search tree because we’ll be relying on the ordering of the list input.

Our high-level strategy before moving through the checkpoints.

base case: the input list is empty
Return "No Child" to represent the lack of node
recursive step: the input list must be divided into two halves
Find the middle index of the list
Store the value located at the middle index
Make a tree node with a "data" key set to the value
Assign tree node’s "left child" to a recursive call using the left half of the list
Assign tree node’s "right child" to a recursive call using the right half of the list
Return the tree node
Instructions
1.
Define the build_bst() function with my_list as the sole parameter.

If my_list has no elements, return “No Child” to represent the lack of a child tree node.

This is the base case of our function.

The recursive step will need to remove an element from the input to eventually reach an empty list.

2.
We’ll be building this tree by dividing the list in half and feeding those halves to the left and right sides of the tree.

This dividing step will eventually produce empty lists to satisfy the base case of the function.

Outside of the conditional you just wrote, declare middle_idx and set it to the middle index of my_list.

Then, declare middle_value and set it to the value in my_list located at middle_idx.

Print “Middle index: “ + middle_idx.

Then, print “Middle value: “ + middle_value

You can use .format() or addition for the print the statement. Addition will require you to use str() on the variables since they are integers!

You can reach the mid-point of a list like so:

colors = ['brown', 'red', 'olive']

mid_idx = len(colors) // 2
# 1
mid_color_value = colors[mid_idx]
# 'red'
and format a string like so:

color = "blue"
print("My favorite color is: {0}".format(color))
# "My favorite color is: blue"
3.
After the print statements, declare the variable tree_node that points to a Python dictionary with a key of "data" pointing to middle_value.

tree_node represents the tree being created in this function call. We want a tree_node created for each element in the list, so we’ll repeat this process on the left and right sub-trees using the appropriate half of the input list.

Now for the recursive calls!

Set the key of "left_child" in tree_node to be a recursive call to build_bst() with the left half of the list not including the middle value as an argument.

Set the key of "right_child" in tree_node to be a recursive call to build_bst() with the right half of the list not including the middle value as an argument.

It’s very important we don’t include the middle_value in the lists we’re passing as arguments, or else we’ll create duplicate nodes!

Finally, return tree_node. As the recursive calls resolve and pop off the call stack, the final return value will be the root or “top” tree_node which contains a reference to every other tree_node.

Our recursive calls will look like the following:

tree_node["left_child"] = build_bst(left_half_of_list)
tree_node["right_child"] = build_bst(right_half_of_list)
We can copy half of a list like so:

pets = ["dogs", "cats", "lizards", "parrots", "giraffes"]
middle_idx = len(pets) // 2
# 2
first_half_pets = pets[:middle_idx + 1]
# ["dogs", "cats", "lizards"]
last_half_pets = pets[middle_idx + 1:]
# ["parrots", "giraffes"]
4.
Congratulations! You’ve built up a recursive data structure with a recursive function!

This data structure can be used to find values in an efficient O(logN) time.

Fill in the variable runtime with the runtime of your build_bst() function.

This runtime is a tricky one so don’t be afraid to use that hint!

N is the length of our input list.

Our tree will be logN levels deep, meaning there will logN times where a new parent-child relationship is created.

If we have an 8 element list, the tree is 3 levels deep: 2**3 == 8.

Each recursive call is going to copy approximately N elements when the left and right halves of the list are passed to the recursive calls. We’re reducing by 1 each time (the middle_value), but that’s a constant factor.

Putting that together, we have N elements being copied logN levels for a big O of N*logN.

'''
# Define build_bst() below...
def build_bst(my_list):
  if my_list == []:
    return "No Child"
  middle_idx = len(my_list) // 2
  middle_value = my_list[middle_idx]

  print('Middle Index: {}'.format(middle_idx))
  print('Middle Value: {}'.format(middle_value))

  tree_node = {}

  tree_node['data'] = middle_value

  tree_node['left_child'] = build_bst(my_list[:middle_idx])
  tree_node['right_child'] = build_bst(my_list[middle_idx+1:])

  return tree_node

# For testing
sorted_list = [12, 13, 14, 15, 16]
binary_search_tree = build_bst(sorted_list)
print(binary_search_tree)

# fill in the runtime as a string
# 1, logN, N, N*logN, N^2, 2^N, N!
runtime = 'N*logN'


'''RECURSION VS. ITERATION - CODING THROWDOWN
Rules of the Throwdown
This lesson will provide a series of algorithms and an iterative or recursive implementation.

Anything we write iteratively, we can also write recursively, and vice versa. Often, the difference is substituting a loop for recursive calls.

Your mission is to recreate the algorithm using the alternative strategy. If the example is recursive, write the algorithm using iteration. If the algorithm uses iteration, solve the problem using recursion.

By the end of this lesson, you’ll have gained a better understanding of the different strategies to implement an algorithm, and along the way, we’ll discuss the big O runtimes of each algorithm.

Each exercise will have a single checkpoint. You can implement the algorithm however you like as long as you’re following the prescribed approach (iterative or recursive).

If you’re feeling stuck, the hint will give a detailed breakdown of how to implement the algorithm.

We’ll start with a classic recursive example, factorial(). This function returns the product of every number from 1 to the given input.

'''
# runtime: Linear - O(N)
def factorial(n):  
  if n < 0:    
    ValueError("Inputs 0 or greater only") 
  if n <= 1:    
    return 1  
  return n * factorial(n - 1)

factorial(3)
# 6
factorial(4)
# 24
factorial(0)
# 1
factorial(-1)
# ValueError "Input must be 0 or greater"

'''
This is a linear implementation, or O(N), where N is the number given as input.

Instructions
1.
Implement your version of factorial() which has the same functionality without using any recursive calls!


Here’s the step by step strategy:

'''
# initialize a result variable set to 1
# loop while the input does not equal 0
  # reassign result to be result * input
  # decrement input by 1
# return result
'''

'''
# recursive function for factoria
# runtime: Linear - O(N)
def recu_factorial(n):  
  if n < 0:    
    ValueError("Inputs 0 or greater only") 
  if n <= 1:    
    return 1  
  return n * factorial(n - 1)

def factorial(n):
  if n < 0:
    ValueError("Inputs 0 or greater only")
  if n == 0：
    return 1

  result = 1
  for i in range(1,n):
    result *= i
  return i  


# test cases
print(factorial(3) == 6)
print(factorial(0) == 1)
print(factorial(5) == 120)

'''
RECURSION VS. ITERATION - CODING THROWDOWN
When Fibs Are Good
Nice work! We’ll demonstrate another classic recursive function: fibonacci().

fibonacci() should return the Nth Fibonacci number, where N is the number given as input. The first two numbers of a Fibonacci Sequence start with 0 and 1. Every subsequent number is the sum of the previous two.

Our recursive implementation:
'''

# runtime: Exponential - O(2^N)

def fibonacci(n):
  if n < 0:
    ValueError("Input 0 or greater only!")
  if n <= 1:
    return n
  return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(3)
# 2
fibonacci(7)
# 13
fibonacci(0)
# 0

'''
Instructions
1.
Implement your version of fibonacci() which has the same functionality without using any recursive calls!


Here’s our step-by-step strategy:
'''

# define a fibs list of [0, 1]
# if the input is <= to len() of fibs - 1
  # return value at index of input
# else:
  # while input is > len() of fibs - 1
    # next_fib will be fibs[-1] + fibs[-2]
    # append next_fib to fibs
# return value at index of input
'''

'''

# runtime: Exponential - O(2^N)
# fibonacci in Recusion method

def recu_fibonacci(n):
  if n < 0:
    ValueError("Input 0 or greater only!")
  if n <= 1:
    return n
  return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci(n):
  if n < 0:
    FibonacciValueError('Input 0 or greater Only!')
  if n <= 1:
    return n
  
  a = 0 
  b = 1

  for i in range(n-1):
    c = a + b
    a = b
    b = c    
  return c

# test cases
print(fibonacci(3) == 2)
print(fibonacci(7) == 13)
print(fibonacci(0) == 0)

print(recu_fibonacci(100))


'''
RECURSION VS. ITERATION - CODING THROWDOWN
Let's Give'em Sum Digits To Talk About
Fantastic! Now we’ll switch gears and show you an iterative algorithm to sum the digits of a number.

This function, sum_digits(), produces the sum of all the digits in a positive number as if they were each a single number:
'''

# Linear - O(N), where "N" is the number of digits in the number
def sum_digits(n):
  if n < 0:
    ValueError("Inputs 0 or greater only!")
  result = 0
  while n is not 0:
    result += n % 10
    n = n // 10
  return result + n

sum_digits(12)
# 1 + 2
# 3
sum_digits(552)
# 5 + 5 + 2
# 12
sum_digits(123456789)
# 1 + 2 + 3 + 4...
# 45

'''
Instructions
1.
Implement your version of sum_digits() which has the same functionality using recursive calls!


Here’s the outline of our strategy:

'''

# base case: if input <= 9
  # return input
# recursive step
# last_digit set to input % 10
# return recursive call with (input // 10) added to last_digit
'''

'''
def sum_digits(n):
  if n < 0:
    SumDigitsError('Input number Greater than 0!')
  if n < 10:
    return n
  
  result = 0

  result += n % 10 + sum_digits(n // 10)

  return result

# test cases
print(sum_digits(12) == 3)
print(sum_digits(552) == 12)
print(sum_digits(123456789) == 45)

'''
RECURSION VS. ITERATION - CODING THROWDOWN
It Was Min To Be
We’ll use an iterative solution to the following problem: find the minimum value in a list.
'''

def find_min(my_list):
  min = None
  for element in my_list:
    if not min or (element < min):
      min = element
  return min

find_min([42, 17, 2, -1, 67])
# -1
find_mind([])
# None
find_min([13, 72, 19, 5, 86])
# 5

'''
This solution has a linear runtime, or O(N), where N is the number of elements in the list.

Instructions
1.
Implement your version of find_min() which has the same functionality using recursive calls!


Here’s our strategy:
'''

# function definition with two inputs: 
# a list and a min that defaults to None
  # BASE CASE
  # if input is an empty list
    # return min
  # else
    # RECURSIVE STEP
    # if min is None
    # OR
    # first element of list is < min
      # set min to be first element
  # return recursive call with list[1:] and the min
'''

'''

''' this code not working
def find_min(my_list):
  min = None
  
  if len(my_list) == 0:
    return min
  
  if len(my_list) == 1:
    return my_list[0]

  min = my_list[0]

  if min < find_min(my_list[1:]):
    
    return min
'''

def find_min(my_list): # this code doesn't either
  min = None
  
  if len(my_list) == 0:
    return None
  
  if min == None or my_list[0] < min:
    min = my_list[0]
  
  return find_min(my_list[1:])
  
  
def find_min(my_list, min = None):
  if not my_list:
    return min

  if not min or my_list[0] < min:
    min = my_list[0]
  return find_min(my_list[1:], min)


# test cases

print(find_min([42, 17, 2, -1, 67]))
print(find_min([]) == None)
print(find_min([13, 72, 19, 5, 86]))

#print(find_min([42, 17, 2, -1, 67]) == -1)
#print(find_min([]) == None)
#print(find_min([13, 72, 19, 5, 86]) == 5)

'''
RECURSION VS. ITERATION - CODING THROWDOWN
Taco Cat
Palindromes are words which read the same forward and backward. Here’s an iterative function that checks whether a given string is a palindrome:
'''

def is_palindrome(my_string):
  while len(my_string) > 1:
    if my_string[0] != my_string[-1]:
      return False
    my_string = my_string[1:-1]
  return True 

palindrome("abba")
# True
palindrome("abcba")
# True
palindrome("")
# True
palindrome("abcd")
# False

'''
Take a moment to think about the runtime of this solution.

In each iteration of the loop that doesn’t return False, we make a copy of the string with two fewer characters.

Copying a list of N elements requires N amount of work in big O.

This implementation is a quadratic solution: we’re looping based on N and making a linear operation for each loop!

Here’s a more efficient version:
'''

# Linear - O(N)
def is_palindrome(my_string):
  string_length = len(my_string)
  middle_index = string_length // 2
  for index in range(0, middle_index):
    opposite_character_index = string_length - index - 1
    if my_string[index] != my_string[opposite_character_index]:
      return False  
  return True
  
'''  
Note these solutions do not account for spaces or capitalization in the input!

Instructions
1.
Implement your version of is_palindrome() which has the same functionality using recursive calls!


Here’s the outline of our strategy:
'''

# BASE CASE
# the input string is less than 2 characters
  # return True
# RECURSIVE STEP
# str[0] does not match str[-1]
  # return False
# return recursive call with str[1:-1]

'''

'''
def is_palindrome(my_string):
  if len(my_string) < 2:
    return True
  
  if my_string[0] != my_string[-1]:
    return False
  
  return is_palindrome(my_string[1:-1])


# test cases
print(is_palindrome("abba") == True)
print(is_palindrome("abcba") == True)
print(is_palindrome("") == True)
print(is_palindrome("abcd") == False)


'''
RECURSION VS. ITERATION - CODING THROWDOWN
Multiplication? Schmultiplication!
All programming languages you’re likely to use will include arithmetic operators like +, -, and *.

Let’s pretend Python left out the multiplication, *, operator.

How could we implement it ourselves? Well, multiplication is repeated addition. We can use a loop!

Here’s an iterative approach:
'''

def multiplication(num_1, num_2):
  result = 0
  for count in range(0, num_2):
    result += num_1
  return result

multiplication(3, 7)
# 21
multiplication(5, 5)
# 25
multiplication(0, 4)
# 0

'''
This implementation isn’t quite as robust as the built-in operator. It won’t work with negative numbers, for example. We don’t expect your implementation to handle negative numbers either!

What is the big O runtime of our implementation?

Instructions
1.
Implement your version of multiplication() which has the same functionality using recursive calls!


Here’s the outline of our strategy:
'''

# BASE CASE
# if either input is 0, return 0
# RECURSIVE STEP
# return one input added to a recursive call with the OTHER input decremented by 1
'''

'''
def multiplication(num1, num2):
  result = 0
  if num2 == 0:
    return result

  result += num1 + multiplication(num1, num2 - 1)
  return result



# test cases
print(multiplication(3, 7) == 21)
print(multiplication(5, 5) == 25)
print(multiplication(0, 4) == 0)

'''
RECURSION VS. ITERATION - CODING THROWDOWN
How Deep Is Your Tree?
Binary trees, trees which have at most two children per node, are a useful data structure for organizing hierarchical data.

It’s helpful to know the depth of a tree, or how many levels make up the tree.
'''

# first level
root_of_tree = {"data": 42}
# adding a child - second level
root_of_tree["left_child"] = {"data": 34}
root_of_tree["right_child"] = {"data": 56}
# adding a child to a child - third level
first_child = root_of_tree["left_child"]
first_child["left_child"] = {"data": 27}

'''
Here’s an iterative algorithm for counting the depth of a given tree.

We’re using Python dictionaries to represent each tree node, with the key of "left_child" or "right_child" referencing another tree node, or None if no child exists.
'''

def depth(tree):
  result = 0
  # our "queue" will store nodes at each level
  queue = [tree]
  # loop as long as there are nodes to explore
  while queue:
    # count the number of child nodes
    level_count = len(queue)
    for child_count in range(0, level_count):
      # loop through each child
      child = queue.pop(0)
     # add its children if they exist
      if child["left_child"]:
        queue.append(child["left_child"])
      if child["right_child"]:
        queue.append(child["right_child"])
    # count the level
    result += 1
  return result

two_level_tree = {
"data": 6, 
"left_child":
  {"data": 2}
}

four_level_tree = {
"data": 54,
"right_child":
  {"data": 93,
   "left_child":
     {"data": 63,
      "left_child":
        {"data": 59}
      }
   }
}


depth(two_level_tree)
# 2
depth(four_level_tree)
# 4
'''

This algorithm will visit each node in the tree once, which makes it a linear runtime, O(N), where N is the number of nodes in the tree.

Implement your version of depth() which has the same functionality using recursive calls!


Here’s our strategy:
'''

# function takes "tree_node" as input
  # BASE CASE
  # if tree_node is None
    # return 0
  # RECURSIVE STEP
  # set left_depth to recursive call passing tree_node's left child
  # set right_depth to recursive call passing tree_node's right child

  # if left_depth is greater than right depth:
    # return left_depth + 1
  # else
    # return right_depth + 1
'''



'''


def iterate_depth(tree):
  result = 0
  # our "queue" will store nodes at each level
  queue = [tree]
  #print(queue)
  #print('\n')
  #print(len(queue))
  # loop as long as there are nodes to explore
  while queue:
    # count the number of child nodes
    level_count = len(queue)
    for child_count in range(0, level_count):
      # loop through each child
      child = queue.pop(0)
      #print(child)
     # add its children if they exist
      if child["left_child"]:
        queue.append(child["left_child"])
      if child["right_child"]:
        queue.append(child["right_child"])
    # count the level
    result += 1
  return result

  
def depth(tree):  #Gux version
  result = 0

  if tree['left_child'] == None:
    return 1
   
  else:
    result += 1 + depth(tree['left_child'])
    return result


# HELPER FUNCTION TO BUILD TREES
def build_bst(my_list):
  if len(my_list) == 0:
    return None

  mid_idx = len(my_list) // 2
  mid_val = my_list[mid_idx]

  tree_node = {"data": mid_val}
  tree_node["left_child"] = build_bst(my_list[ : mid_idx])
  tree_node["right_child"] = build_bst(my_list[mid_idx + 1 : ])

  return tree_node

# HELPER VARIABLES
tree_level_1 = build_bst([1])
tree_level_2 = build_bst([1, 2, 3])
tree_level_4 = build_bst([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) 


# test cases
print(depth(tree_level_1) == 1)
print(depth(tree_level_2) == 2)
print(depth(tree_level_4) == 4)

#print(depth(tree_level_1))
#print(depth(tree_level_2))
#print(depth(tree_level_4))


# first level
#root_of_tree = {"data": 42}
# adding a child - second level
#root_of_tree["left_child"] = {"data": 34}
#root_of_tree["right_child"] = {"data": 56}
# adding a child to a child - third level
#first_child = root_of_tree["left_child"]
#first_child["left_child"] = None
#first_child["right_child"] = None
#second_child = root_of_tree["right_child"]
#second_child["left_child"] = None
#second_child["right_child"] = None
#first_child["right_child"] = {"data": 67}

#print(root_of_tree)
#print(depth(root_of_tree))



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-RECURSION CHEATSHEETS #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Cheatsheets / Learn Recursion: Python

Recursion: Python
Print PDF icon
Print Cheatsheet

TOPICS
Recursion: Conceptual
Recursion: Python
Stack Overflow Error in Recursive Function
A recursive function that is called with an input that requires too many iterations will cause the call stack to get too large, resulting in a stack overflow error. In these cases, it is more appropriate to use an iterative solution. A recursive solution is only suited for a problem that does not exceed a certain number of recursive calls.

For example, myfunction() below throws a stack overflow error when an input of 1000 is used.'''

def myfunction(n):
  if n == 0:
    return n
  else:
    return myfunction(n-1)
    
'''
myfunction(1000)  #results in stack overflow error
Fibonacci Sequence
A Fibonacci sequence is a mathematical series of numbers such that each number is the sum of the two preceding numbers, starting from 0 and 1.

Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
Call Stack Construction in While Loop
A call stack with execution contexts can be constructed using a while loop, a list to represent the call stack and a dictionary to represent the execution contexts. This is useful to mimic the role of a call stack inside a recursive function.

Binary Search Tree
In Python, a binary search tree is a recursive data structure that makes sorted lists easier to search. Binary search trees:

Reference two children at most per tree node.
The “left” child of the tree must contain a value lesser than its parent.
The “right” child of the tree must contain a value greater than it’s parent.
     5
    / \
   /   \
  3     8
 / \   / \
2   4 7   9
Recursion and Nested Lists
A nested list can be traversed and flattened using a recursive function. The base case evaluates an element in the list. If it is not another list, the single element is appended to a flat list. The recursive step calls the recursive function with the nested list element as input.'''

def flatten(mylist):
  flatlist = []
  for element in mylist:
    if type(element) == list:
      flatlist += flatten(element)
    else:
      flatlist += element
  return flatlist

print(flatten(['a', ['b', ['c', ['d']], 'e'], 'f']))
# returns ['a', 'b', 'c', 'd', 'e', 'f']

'''
Fibonacci Recursion
Computing the value of a Fibonacci number can be implemented using recursion. Given an input of index N, the recursive function has two base cases – when the index is zero or 1. The recursive function returns the sum of the index minus 1 and the index minus 2.

The Big-O runtime of the Fibonacci function is O(2^N).
'''
def fibonacci(n):
  if n <= 1:
    return n
  else:
    return fibonacci(n-1) + fibonacci(n-2)
'''
Modeling Recursion as Call Stack
One can model recursion as a call stack with execution contexts using a while loop and a Python list. When the base case is reached, print out the call stack list in a LIFO (last in first out) manner until the call stack is empty.

Using another while loop, iterate through the call stack list. Pop the last item off the list and add it to a variable to store the accumulative result.

Print the result.
'''
def countdown(value):
  call_stack = []
  while value > 0 : 
    call_stack.append({"input":value})
    print("Call Stack:",call_stack)
    value -= 1
  print("Base Case Reached")
  while len(call_stack) != 0:
    print("Popping {} from call stack".format(call_stack.pop()))
    print("Call Stack:",call_stack)
countdown(4)
'''
Call Stack: [{'input': 4}]             
Call Stack: [{'input': 4}, {'input': 3}]         
Call Stack: [{'input': 4}, {'input': 3}, {'input': 2}]     
Call Stack: [{'input': 4}, {'input': 3}, {'input': 2}, {'input': 1}]                                
Base Case Reached                                  
Popping {'input': 1} from call stack                       
Call Stack: [{'input': 4}, {'input': 3}, {'input': 2}]  
Popping {'input': 2} from call stack                   
Call Stack: [{'input': 4}, {'input': 3}]       
Popping {'input': 3} from call stack            
Call Stack: [{'input': 4}]                                 
Popping {'input': 4} from call stack              
Call Stack: []
'''
'''


Recursion in Python
In Python, a recursive function accepts an argument and includes a condition to check whether it matches the base case. A recursive function has:

Base Case - a condition that evaluates the current input to stop the recursion from continuing.
Recursive Step - one or more calls to the recursive function to bring the input closer to the base case.'''
def countdown(value):
  if value <= 0:   #base case  
    print("done")
  else:
    print(value)
    countdown(value-1)  #recursive case 

'''    
Build a Binary Search Tree
To build a binary search tree as a recursive algorithm do the following:

BASE CASE: 
If the list is empty, return "No Child" to show that there is no node. 

RECURSIVE STEP:
1. Find the middle index of the list.
2. Create a tree node with the value of the middle index.
3. Assign the tree node's left child to a recursive call with the left half of list as input.
4. Assign the tree node's right child to a recursive call with the right half of list as input.
5. Return the tree node.'''

def build_bst(my_list):
  if len(my_list) == 0:
    return "No Child"

  middle_index = len(my_list) // 2
  middle_value = my_list[middle_index]
  
  print("Middle index: {0}".format(middle_index))
  print("Middle value: {0}".format(middle_value))
  
  tree_node = {"data": middle_value}
  tree_node["left_child"] = build_bst(my_list[ : middle_index])
  tree_node["right_child"] = build_bst(my_list[middle_index + 1 : ])

  return tree_node
  
sorted_list = [12, 13, 14, 15, 16]
binary_search_tree = build_bst(sorted_list)
print(binary_search_tree)
'''
Recursive Depth of Binary Search Tree
A binary search tree is a data structure that builds a sorted input list into two subtrees. The left child of the subtree contains a value that is less than the root of the tree. The right child of the subtree contains a value that is greater than the root of the tree.

A recursive function can be written to determine the depth of this tree.'''

def depth(tree):
  if not tree:
    return 0
  left_depth = depth(tree["left_child"])
  right_depth = depth(tree["right_child"])
  return max(left_depth, right_depth) + 1
  
'''
Sum Digits with Recursion
Summing the digits of a number can be done recursively. For example:

552 = 5 + 5 + 2 = 12
'''
def sum_digits(n): 
  if n <= 9: 
    return n 
  last_digit = n % 10 
  return sum_digits(n // 10) + last_digit

sum_digits(552) #returns 12
'''

Palindrome in Recursion
A palindrome is a word that can be read the same both ways - forward and backward. For example, abba is a palindrome and abc is not.

The solution to determine if a word is a palindrome can be implemented as a recursive function.'''

def is_palindrome(str):
  if len(str) < 2:
    return True
  if str[0] != str[-1]:
    return False
  return is_palindrome(str[1:-1])
'''  
Fibonacci Iterative Function
A Fibonacci sequence is made up adding two previous numbers beginning with 0 and 1. For example:

0, 1, 1, 2, 3, 5, 8, 13, ...
A function to compute the value of an index in the Fibonacci sequence, fibonacci(index) can be written as an iterative function.
'''
def fibonacci(n):
  if n < 0:
    raise ValueError("Input 0 or greater only!")
  fiblist = [0, 1]
  for i in range(2,n+1):
    fiblist.append(fiblist[i-1] + fiblist[i-2])
  return fiblist[n]
'''
Recursive Multiplication
The multiplication of two numbers can be solved recursively as follows:

Base case: Check for any number that is equal to zero.
Recursive step: Return the first number plus a recursive call of the first number and the second number minus one.

'''
def multiplication(num1, num2):
  if num1 == 0 or num2 == 0:
    return 0
  return num1 + multiplication(num1, num2 - 1)
  
'''
Iterative Function for Factorials
To compute the factorial of a number, multiply all the numbers sequentially from 1 to the number.

An example of an iterative function to compute a factorial is given below.
'''
def factorial(n): 
  answer = 1
  while n != 0:
    answer *= n
    n -= 1
  return answer

'''
Recursively Find Minimum in List
We can use recursion to find the element with the minimum value in a list, as shown in the code below.
'''
def find_min(my_list):
  if len(my_list) == 0:
    return None
  if len(my_list) == 1:
    return my_list[0]
  #compare the first 2 elements
  temp = my_list[0] if my_list[0] < my_list[1] else my_list[1]
  my_list[1] = temp
  return find_min(my_list[1:])

print(find_min([]) == None)
print(find_min([42, 17, 2, -1, 67]) == -1)


















