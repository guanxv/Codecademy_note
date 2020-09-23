

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-PYTHON FUNCTION PYTHON FUNCTION PYTHON FUNCTION PYTHON FUNCTION PYTHON FUNCTION FUNCTION#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# The "def" keyword is the start of a function definition
def function_name(parameter1, parameter2):
  # The placeholder variables used inside a function definition are called parameters
  print(parameter1)
  return parameter2
# The outdent signals the end of the function definition

# "Arguments" are the values passed into a function call
argument1 = "argument 1"
argument2 = "argument 2"

# A function call uses the functions name with a pair of parentheses
# and can return a value
return_val = function_name(argument1, argument2)


#---------------------

none_var = None
if none_var:
  print("Hello!")
else:
  print("Goodbye")

# Prints "Goodbye"

#None is falsy, meaning that it evaluates to False in an if statement, which is why the above code prints “Goodbye”.

#------------

# first we define session_id as None
session_id = None

if session_id is None:
  print("session ID is None!")
  # this prints out "session ID is None!"

# we can assign something to session_id
if active_session:
  session_id = active_session.id

# but if there's no active_session, we don't send sensitive data
if session_id is not None:
  send_sensitive_data(session_id)

#-----------------------------

def no_return():
  print("You've hit the point of no return")
  # notice no return statement

here_it_is = no_return()

print(here_it_is)
# Prints out "None"

#Above we defined a function called no_return() and saved the result to a variable here_it_is. When we print() the value of here_it_is we get None, referring to Python’s None.

def fifty_percent_off(item):
  # Check if item.cost exists
  if hasattr(item, 'cost'):
    return item.cost / 2

  # If not, return None 
  return

sale_price = fifty_percent_off(product)

if sale_price is None:
  print("This product is not for sale!")
  
#Here we have implemented a function that returns the cost of a product with “50% Off” (or “half price”). We check if the item passed to our function has a cost attribute. If it exists, we return half the cost. If not, we simply return, which returns None.
#----------------------

# store the result of this print() statement in the variable prints_return
prints_return = print("What does this print function return?")

# print out the value of prints_return
print(prints_return)

# call sort_this_list.sort() and save that in list_sort_return
sort_this_list = [14, 631, 4, 51358, 50000000]

list_sort_return = sort_this_list.sort()

# print out the value of list_sort_return
print(list_sort_return)


#----------------------

# Function definition with two required parameters
def create_user(username, is_admin):
  create_email(username)
  set_permissions(is_admin)

# Function call with all two required arguments
user1 = create_user('johnny_thunder', True)

# Raises a "TypeError: Missing 1 required positional argument"
user2 = create_user('djohansen')

#----

# Function defined with one required and one optional parameter
def create_user(username, is_admin=False):
  create_email(username)
  set_permissions(is_admin)

# Calling with two arguments uses the default value for is_admin
user2 = create_user('djohansen')

#-----

# We can make all three parameters optional
def create_user(username=None, is_admin=False):
  if username is None:
    username = "Guest"
  else:
    create_email(username)
  set_permissions(is_admin)

# So we can call with just one value
user3 = create_user('ssylvain')
# Or no value at all, which would create a Guest user
user4 = create_user()

#--------------------------------

import os

def make_folders(folders_list, nest = False):
  if nest:
    """
    Nest all the folders, like
    ./Music/fun/parliament
    """
    path_to_new_folder = ""
    for folder in folders_list:
      path_to_new_folder += "/{}".format(folder)
      try:
        os.makedirs(path_to_new_folder)
      except FileExistsError:
        continue
  else:
    """
    Makes all different folders, like
    ./Music/ ./fun/ and ./parliament/
    """
    for folder in folders_list:
      try:
        os.makedirs(folder)

      except FileExistsError:
        continue

make_folders(['./Music', './fun', './parliament'])

#---------------------------------------------------

#Python will only accept functions defined with their parameters in a specific order. The required parameters need to occur before any parameters with a default argument.

# Raises a TypeError
def create_user(is_admin=False, username, password):
  create_email(username, password)
  set_permissions(is_admin)
  
# Works perfectly
def create_user(username, password, is_admin=False):
  create_email(username, password)
  set_permissions(is_admin)

#-------------------------------------------

# Define a function with a bunch of default arguments
def log_message(logging_style="shout", message="", font="Times", date=None):
  if logging_style == 'shout':
    # capitalize the message
    message = message.upper()
  print(message, date)

# Now call the function with keyword arguments
log_message(message="Hello from the past", date="November 20, 1693")

#Above we defined log_message(), which can take from 0 to 4 arguments. Since it’s not clear which order the four arguments might be defined in, we can use the parameter names to call the function. Notice that in our function call we use this syntax: message="Hello from the past". The key word message here needs to be the name of the parameter we are trying to pass the argument to.

#-----------------------------------------
def populate_list(list_to_populate=[], length=1):
  for num in range(length):
    list_to_populate.append(num)
  return list_to_populate
  
returned_list = populate_list(length=4)
print(returned_list)
# Prints [0, 1, 2, 3] -- this is expected

returned_list = populate_list(length=6)
print(returned_list)
# Prints [0, 1, 2, 3, 0, 1, 2, 3, 4, 5] -- this is a surprise!

#When we call populate_list a second time we’d expect the list [0, 1, 2, 3, 4, 5]. But the same list is used both times the function is called, causing some side-effects from the first function call to creep into the second. This is because a list is a mutable object.

#A mutable object refers to various data structures in Python that are intended to be mutated, or changed. A list has append and remove operations that change the nature of a list. Sets and dictionaries are two other mutable objects in Python.

#It might be helpful to note some of the objects in Python that are not mutable (and therefore OK to use as default arguments). int, float, and other numbers can’t be mutated (arithmetic operations will return a new number). tuples are a kind of immutable list. Strings are also immutable — operations that update a string will all return a completely new string.

#------------------------------

def add_author(authors_books, current_books=None):
  if current_books is None:
    current_books = []

  current_books.extend(authors_books)
  return current_books
  
#---------------------------------

def update_order(new_item, current_order=[]):
  current_order.append(new_item)
  return current_order

# First order, burger and a soda
order1 = update_order({'item': 'burger', 'cost': '3.50'})
order1 = update_order({'item': 'soda', 'cost': '1.50'}, order1)

# Second order, just a soda
order2 = update_order({'item': 'soda', 'cost': '1.50'})

# What's in that second order again?
print(order2)

#---------------- fix of above code --------------------

def update_order(new_item, current_order=None):
  if current_order is None:
    current_order = []
  current_order.append(new_item)
  return current_order

# First order, burger and a soda
order1 = update_order({'item': 'burger', 'cost': '3.50'})
order1 = update_order({'item': 'soda', 'cost': '1.50'}, order1)

# Second order, just a soda
order2 = update_order({'item': 'soda', 'cost': '1.50'})

# What's in that second order again?
print(order2)

#------------------------
def multiple_returns(cool_num1, cool_num2):
  sum_nums = cool_num1 + cool_num2
  div_nums = cool_num1 / cool_num2
  return sum_nums, div_nums

  
sum_and_div = multiple_returns(20, 10)

print(sum_and_div)
# Prints "(30, 2)"

print(sum_and_div[0])
# Prints "30"

#So we get those two values back in what’s called a tuple, an immutable list-like object indicated by parentheses. 

#What if we wanted to save these two results in separate variables? 
#Well we can by unpacking the function return. We can list our new variables, comma-separated, that correspond to the number of values returned:

sum, div = sum_and_div(18, 9)

print(sum)
# Prints "27"

print(div)
# Prints "2"

#----------------------

#The first method is called positional argument unpacking, because it unpacks arguments given by position.

def shout_strings(*args):
  for argument in args:
    print(argument.upper())

shout_strings("hi", "what do we have here", "cool, thanks!")
# Prints out:
# "HI"
# "WHAT DO WE HAVE HERE"
# "COOL, THANKS!"

#In shout_strings() we use a single asterisk (*) to indicate we’ll accept any number of positional arguments passed to the function. Our parameter args is a tuple of all of the arguments passed.

#Note that args is just a parameter name, and we aren’t limited to that name (although it is rather standard practice). We can also have other positional parameters before our *args parameter. 

def truncate_sentences(length, *sentences):
  for sentence in sentences:
    print(sentence[:length])

truncate_sentences(8, "What's going on here", "Looks like we've been cut off")
# Prints out:
# "What's g"
# "Looks li"

#------

#The Python library os.path has a function called join(). join() takes an arbitrary number of paths as arguments.

from os.path import join

path_segment_1 = "/Home/User"
path_segment_2 = "Codecademy/videos"
path_segment_3 = "cat_videos/surprised_cat.mp4"

# join all three of the paths here!

print(join(path_segment_1, path_segment_2, path_segment_3))

def myjoin2(*args):
  joined_string = args[0]
  for arg in args[1:]:
    joined_string += arg
  return joined_string

print(myjoin2(path_segment_1, path_segment_2, path_segment_3))


#----------------------------------

#Python doesn’t stop at allowing us to accept unlimited positional parameters, it gives us the power to define functions with unlimited keyword parameters too. The syntax is very similar, but uses two asterisks (**) instead of one. Instead of args, we call this kwargs — as a shorthand for keyword arguments.


def arbitrary_keyword_args(**kwargs):
  print(type(kwargs))
  print(kwargs)
  # See if there's an "anything_goes" keyword arg
  # and print it
  print(kwargs.get('anything_goes'))

arbitrary_keyword_args(this_arg="wowzers", anything_goes=101)
# Prints "<class 'dict'>
# Prints "{'this_arg': 'wowzers', 'anything_goes': 101}"
# Prints "101"

#As you can see, **kwargs gives us a dictionary with all the keyword arguments that were passed to arbitrary_keyword_args. We can access these arguments using standard dictionary methods.

#------------------------------------------------

def create_products(**products_dict):
  for name, price in products_dict.items():
    create_product(name, price)

create_products(Baseball = 3 , Shirt = 14, Guitar  = 70)

#-------------- is qual to  -------

def create_products(products_dict):
  for name, price in products_dict.items():
    create_product(name, price)

create_products({
  'Baseball': '3',
  'Shirt': '14',
  'Guitar': '70',


#---------------------------------

#This keyword argument unpacking syntax can be used even if the function takes other parameters. However, the parameters must be listed in this order:

#All named positional parameters
#An unpacked positional parameter (*args)
#All named keyword parameters
#An unpacked keyword parameter (**kwargs)
#Here’s a function with all possible types of parameter:

def main(filename, *args, user_list=None, **kwargs):
  if user_list is None:
    user_list = []

  if '-a' in args:
    user_list = all_users()

  if kwargs.get('active'):
    user_list = [user for user_list if user.active]

  with open(filename) as user_file:
    user_file.write(user_list)

#Looking at the signature of main() we define four different kinds of parameters. The first, filename is a normal required positional parameter. The second, *args, is all positional arguments given to a function after that organized as a tuple in the parameter args. The third, user_list, is a keyword parameter with a default value. Lastly, **kwargs is all other keyword arguments assembled as a dictionary in the parameter kwargs.

#A possible call to the function could look like this:

main("files/users/userslist.txt", 
     "-d", 
     "-a", 
     save_all_records=True, 
     user_list=current_users)

#In the body of main() these arguments would look like this:

#filename == "files/users/userslist.txt"
#args == ('-d', '-a)
#user_list == current_users
#kwargs == {'save_all_records': True}
	
#We can use all four of these kinds of parameters to create functions that handle a lot of possible arguments being passed to them.

#---------------------------------------------------------

#Not only can we accept arbitrarily many parameters to a function in our definition, but Python also supports a syntax that makes deconstructing any data that you have on hand to feed into a function that accepts these kinds of unpacked arguments. The syntax is very similar, but is used when a function is called, not when it’s defined.

def takes_many_args(*args):
  print(','.join(args))

long_list_of_args = [145, "Mexico City", 10.9, "85C"]

# We can use the asterisk "*" to deconstruct the container.
# This makes it so that instead of a list, a series of four different
# positional arguments are passed to the function
takes_many_args(*long_list_of_args)
# Prints "145,Mexico City,10.9,85C"

#We can use the * when calling the function that takes a series of positional parameters to unwrap a list or a tuple. This makes it easy to provide a sequence of arguments to a function even if that function doesn’t take a list as an argument. Similarly we can use ** to destructure a dictionary.

def pour_from_sink(temperature="Warm", flow="Medium")
  set_temperature(temperature)
  set_flow(flow)
  open_sink()

# Our function takes two keyword arguments
# If we make a dictionary with their parameter names...
sink_open_kwargs = {
  'temperature': 'Hot',
  'flow': "Slight",
}

# We can destructure them an pass to the function
pour_from_sink(**sink_open_kwargs)
# Equivalent to the following:
# pour_from_sink(temperature="Hot", flow="Slight")

#So we can also use dictionaries and pass them into functions as keyword arguments with that syntax. Notice that our pour_from_sink() function doesn’t even accept arbitrary **kwargs. We can use this destructuring syntax even when the function has a specific number of keyword or positional arguments it accepts. We just need to be careful that the object we’re destructuring matches the length (and names, if a dictionary) of the signature of the function we’re passing it into.

#---------------------- project -------------------------------------------

#----------------------- script.py-----------------------------------------------

from nile import get_distance, format_price, SHIPPING_PRICES
from test import test_function

# Define calculate_shipping_cost() here:
def calculate_shipping_cost(from_coords, to_coords, shipping_type = 'Overnight'):
  #from_lat, from_long = from_coords
  #to_lat, to_long = to_coords
  from_lat = from_coords[0]
  from_long = from_coords[1]
  to_lat = to_coords[0]
  to_long = to_coords[1]
  #print ("from_lat = " + str(from_lat))
  #print ("from_long = " + str(from_long))
  #print ("to_lat = " + str(to_lat))
  #print ("to_long = " + str(to_long))
  
  distance = get_distance(from_lat, from_long, to_lat, to_long)
  #distance = get_distance(*from_coords, *to_coords) #also works (unpacking)
  
  shipping_rate = SHIPPING_PRICES.get(shipping_type)
  
  price = distance * shipping_rate
  
  return format_price(price)
  

# Test the function by calling 
test_function(calculate_shipping_cost)

# Define calculate_driver_cost() here
def calculate_driver_cost(distance, *drivers):
  
  cheapest_driver = None
  cheapest_driver_price = None
  
  for driver in drivers:
    driver_time = driver.speed * distance
    price_for_driver = driver.salary * driver_time
    
    if cheapest_driver == None:
      cheapest_driver = driver
      cheapest_driver_price =price_for_driver
    elif price_for_driver < cheapest_driver_price:
      cheapest_driver = driver
      cheapest_driver_price =price_for_driver
  
  return cheapest_driver_price , cheapest_driver 

# Test the function by calling 
test_function(calculate_driver_cost)

# Define calculate_money_made() here
def calculate_money_made(**trips):
  total_money_made = 0
  for key, trip in trips.items():
    trip_revenue = trip.cost - trip.driver.cost
    total_money_made += trip_revenue
  
  return total_money_made   


# Test the function by calling 
test_function(calculate_money_made)

#------------------------------------------------- end of script.py-----------------------------------------------

#-------------------------------------------------- nile.py------------------------------------------------------

from math import sin, cos, atan2, sqrt

def get_distance(from_lat, from_long, to_lat, to_long):
  dlon = to_long - from_long
  dlat = from_lat - to_lat
  a = (sin(dlat/2)) ** 2 + cos(from_lat) * cos(to_lat) * (sin(dlon/2)) ** 2
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  distance = a * c
  return distance

SHIPPING_PRICES = {
  'Ground': 1,
  'Priority': 1.6,
  'Overnight': 2.3,
}

def format_price(price):
  return "${0:.2f}".format(price)

#------------------------------------------------- nile.py----------------------------------------------------------

#-------------------------------------------------- test.py ----------------------------------------------------------

def test_function(fn):
  if fn.__name__ == "calculate_shipping_cost":
    test_shipping(fn)
  if fn.__name__ == "calculate_driver_cost":
    test_driver(fn)
  if fn.__name__ == "calculate_money_made":
    test_money(fn)

def test_shipping(f):
  try:
    costs = f((0, 0), (1, 1))
  except TypeError:
    print("calculate_shipping_cost() did not provide default argument for shipping_type")
    return
  if not type(costs) is str:
    print("calculate_shipping_cost() did not format the result in a string")
    return
  if costs != "$1.04":
    print("calculate_shipping_cost((0, 0), (1, 1)) returned {}. Expected result is {}".format(costs, "$1.04"))
    return
  print("OK! calculate_shipping_cost() passes tests")

class Driver:
  def __init__(self, speed, salary):
    self.speed = speed
    self.salary = salary

  def __repr__(self):
    return "Nile Driver speed {} salary {}".format(self.speed, self.salary)

driver1 = Driver(2, 10)
driver2 = Driver(7, 20)

def test_driver(f):
  try:
    price, driver = f(80, driver1, driver2)
  except TypeError:
    print("calculate_driver_cost() doesn't expect multiple driver arguments")
    return
  if type(driver) is not Driver:
    print("calculate_driver_cost() did not return driver")
    return
  if price != 1600:
    print("calculate_driver_cost() did not provide correct final price (expected {}, received {})".format(price,1600))
    return
  if driver is not driver1:
    print("calculate_driver_cost() did not provide least expensive driver")
    return
  print("OK! calculate_driver_cost() passes tests")

class Trip:
  def __init__(self, cost, driver, driver_cost):
    self.cost = cost
    driver.cost = driver_cost
    self.driver = driver

trip1 = Trip(200, driver1, 15)
trip2 = Trip(300, driver2, 40)

def test_money(f):
  try:
    money = f(UEXODI=trip1, DEFZXIE=trip2)
  except TypeError:
    print("calculate_money_made() doesn't expect multiple trip keyword arguments")
    return
  if type(money) not in (int, float):
    print("calculate_driver_cost() did not return a number")
    return
  if money != 445:
    print("calculate_driver_cost() did not provide correct final price (expected {}, received {})".format(money, 445))
    return
  print("OK! calculate_money_made() passes tests")

  
#------------------------------------------------ test.py----------------------------------------------------------

#------------------------------------ end of project ----------------------------------------------------------------


# Fuction is an object

def add_five(num):
	print num + 5

print (add_five)

#this means add_five is a object , just as a number or a list

# funciton inside a funciton 

def add_five(num):
	def add_two(num):
		return num + 2
	
	num_plus_two = add_two(num)
	print(num_plus_two + 3)

add_two(7)

add_five(10)

# Returning funciton from functions

def get_math_function(operation): # + or -
	def add(n1, n2):
		return n1 + n2
	def sub(n1, n2):
		return n1 - n2
	
	if operation == "+":
		return add
	elif operation == "-":
		return sub

add_function = get_math_function("+")
print (add_function(4,6))
#print out 10

#decorate a function 

def title_decorator(print_name_function):
	def wrapper():
		print("Professor:")
		print_name_function()
	return wrapper

def print_my_name():
	print("mike")

print_my_name()

decorated_function = title_decorator(print_my_name)

decorated_function()

#should print out  Professor:
#					mike

#Decorator 
def title_decorator(print_name_function):
	def wrapper():
		print("Professor:")
		print_name_function()
	return wrapper

@title_decorator
def print_my_name():
	print("mike")

print_my_name()

#this should do the same thing like above code
#Professor:
#mike

# decorator with parameters
def title_decorator(print_name_function):
	def wrapper(*args, **kwargs):
		print("Professor:")
		print_name_function(*args, **kwargs)
	return wrapper

@title_decorator
def print_my_name(nane):
	print(name)
	
print_my_name("shelby")

#----------------------------------------------------------------------------
from datetime import datetime

birthday = datetime(1982, 08, 06, 16, 30,0)

birthday.year
#return 1982

birthday.month
#return 08

birthday.weekday()
#return 0 for monday

datetime.now()
#return current date time 

datetime(2019, 01, 01) - datetime(2018, 01, 01)
#return datetime.timedelta(days = 365)

parsed_date = datetime.strptime('Jan 15, 2018', '%b %d, %Y)
#convert the string in to datetime

date_string = datetime.strftime(datetime.now(),%b %d, %Y) 
#convert datetime into string

#-----------------------------------------------------
#-----------------------------------------------------
#-#-Cheatsheets / Learn Python 3

Function Arguments
Print PDF icon
Print Cheatsheet

TOPICS
Syntax
Functions
Control Flow
Lists
Loops
Strings
Modules
Dictionaries
Files
Classes
Function Arguments
Default argument is fallback value
In Python, a default parameter is defined with a fallback value as a default argument. Such parameters are optional during a function call. If no argument is provided, the default value is used, and if an argument is provided, it will overwrite the default value.

def greet(name, msg="How do you do?"):
  print("Hello ", name + ', ' + msg)

greet("Ankit")
greet("Ankit", "How do you do?")

"""
this code will print the following for both the calls -
`Hello  Ankit, How do you do?`
"""
Mutable Default Arguments
Python’s default arguments are evaluated only once when the function is defined, not each time the function is called. This means that if a mutable default argument is used and is mutated, it is mutated for all future calls to the function as well. This leads to buggy behaviour as the programmer expects the default value of that argument in each function call.

# Here, an empty list is used as a default argument of the function.
def append(number, number_list=[]):
  number_list.append(number)
  print(number_list)
  return number_list

# Below are 3 calls to the `append` function and their expected and actual outputs:
append(5) # expecting: [5], actual: [5]
append(7) # expecting: [7], actual: [5, 7]
append(2) # expecting: [2], actual: [5, 7, 2]
Python Default Arguments
A Python function cannot define a default argument in its signature before any required parameters that do not have a default argument. Default arguments are ones set using the form parameter=value. If no input value is provided for such arguments, it will take on the default value.

# Correct order of declaring default argments in a function
def greet(name, msg = "Good morning!"):
  print("Hello ", name + ', ' + msg)
  
# The function can be called in the following ways
greet("Ankit")
greet("Kyla","How are you?")

# The following function definition would be incorrect
def greet(msg = "Good morning!", name):
  print("Hello ", name + ', ' + msg) 
# It would cause a "SyntaxError: non-default argument follows default argument"
Python function default return value
If we do not not specify a return value for a Python function, it returns None. This is the default behaviour.

# Function returning None
def my_function(): pass

print(my_function())

#Output 
None
Python variable None check
To check if a Python variable is None we can make use of the statement variable is None.

If the above statement evaluates to True, the variable value is None.

# Variable check for None
if variable_name is None:
    print "variable is None"
else:
    print "variable is NOT None"
Python function arguments
A function can be called using the argument name as a keyword instead of relying on its positional value. Functions define the argument names in its composition then those names can be used when calling the function.

# The function will take arg1 and arg2
def func_with_args(arg1, arg2):
  print(arg1 + ' ' + arg2)
  
func_with_args('First', 'Second')
# Prints:
# First Second

func_with_args(arg2='Second', arg1='First')
# Prints
# First Second

#-------------------------------------------------------------------
#-------------------------------------------------------------------



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#----------FUNCTION CHEATSHEET FUNCTION CHEATSHEET FUNCTION CHEATSHEET FUNCTION CHEATSHEET     ---------------------#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Cheatsheets / Learn Python 3

Functions
Print PDF icon
Print Cheatsheet

TOPICS
Syntax
Functions
Control Flow
Lists
Loops
Strings
Modules
Dictionaries
Files
Classes
Function Arguments
Function Parameters
Sometimes functions require input to provide data for their code. This input is defined using parameters.

Parameters are variables that are defined in the function definition. They are assigned the values which were passed as arguments when the function was called, elsewhere in the code.

For example, the function definition defines parameters for a character, a setting, and a skill, which are used as inputs to write the first sentence of a book.'''

def write_a_book(character, setting, special_skill):
  print(character + " is in " + 
        setting + " practicing her " + 
        special_skill)

'''
Multiple Parameters
Python functions can have multiple parameters. Just as you wouldn’t go to school without both a backpack and a pencil case, functions may also need more than one input to carry out their operations.

To define a function with multiple parameters, parameter names are placed one after another, separated by commas, within the parentheses of the function definition.'''

def ready_for_school(backpack, pencil_case):
  if (backpack == 'full' and pencil_case == 'full'):
    print ("I'm ready for school!")
    
'''    
Functions
Some tasks need to be performed multiple times within a program. Rather than rewrite the same code in multiple places, a function may be defined using the def keyword. Function definitions may include parameters, providing data input to the function.

Functions may return a value using the return keyword followed by the value to return.
'''

# Define a function my_function() with parameter x

def my_function(x):
  return x + 1

# Invoke the function

print(my_function(2))      # Output: 3
print(my_function(3 + 5))  # Output: 9

'''
Function Indentation
Python uses indentation to identify blocks of code. Code within the same block should be indented at the same level. A Python function is one type of code block. All code under a function declaration should be indented to identify it as part of the function. There can be additional indentation within a function to handle other statements such as for and if so long as the lines are not indented less than the first line of the function code.
'''

# Indentation is used to identify code blocks

def testfunction(number):
  # This code is part of testfunction
  print("Inside the testfunction")
  sum = 0
  for x in range(number):
    # More indentation because 'for' has a code block
    # but still part of he function
    sum += x
  return sum
print("This is not part of testfunction")
'''

Calling Functions
Python uses simple syntax to use, invoke, or call a preexisting function. A function can be called by writing the name of it, followed by parentheses.

For example, the code provided would call the doHomework() method.
'''

doHomework()

'''
Function Arguments
Parameters in python are variables — placeholders for the actual values the function needs. When the function is called, these values are passed in as arguments.

For example, the arguments passed into the function .sales() are the “The Farmer’s Market”, “toothpaste”, and “$1” which correspond to the parameters grocery_store, item_on_sale, and cost.
'''

def sales(grocery_store, item_on_sale, cost):
  print(grocery_store + " is selling " + item_on_sale + " for " + cost) 

sales("The Farmer’s Market", "toothpaste", "$1")

'''
Function Keyword Arguments
Python functions can be defined with named arguments which may have default values provided. When function arguments are passed using their names, they are referred to as keyword arguments. The use of keyword arguments when calling a function allows the arguments to be passed in any order — not just the order that they were defined in the function. If the function is invoked without a value for a specific argument, the default value will be used.
'''

def findvolume(length=1, width=1, depth=1):
  print("Length = " + str(length))
  print("Width = " + str(width))
  print("Depth = " + str(depth))
  return length * width * depth;

findvolume(1, 2, 3)
findvolume(length=5, depth=2, width=4)
findvolume(2, depth=3, width=4)
'''

Returning Multiple Values
Python functions are able to return multiple values using one return statement. All values that should be returned are listed after the return keyword and are separated by commas.

In the example, the function square_point() returns x_squared, y_squared, and z_squared.
'''

def square_point(x, y, z):
  x_squared = x * x
  y_squared = y * y
  z_squared = z * z
  # Return all three values:
  return x_squared, y_squared, z_squared

three_squared, four_squared, five_squared = square_point(3, 4, 5)

'''
The Scope of Variables
In Python, a variable defined inside a function is called a local variable. It cannot be used outside of the scope of the function, and attempting to do so without defining the variable outside of the function will cause an error.

In the example, the variable a is defined both inside and outside of the function. When the function f1() is implemented, a is printed as 2 because it is locally defined to be so. However, when printing a outside of the function, a is printed as 5 because it is implemented outside of the scope of the function.
'''

a = 5

def f1():
  a = 2
  print(a)
  
print(a)   # Will print 5
f1()       # Will print 2

'''
Returning Value from Function
A return keyword is used to return a value from a Python function. The value returned from a function can be assigned to a variable which can then be used in the program.

In the example, the function check_leap_year returns a string which indicates if the passed parameter is a leap year or not.
'''

def check_leap_year(year): 
  if year % 4 == 0:
    return str(year) + " is a leap year."
  else:
    return str(year) + " is not a leap year."

year_to_check = 2018
returned_value = check_leap_year(year_to_check)
print(returned_value) # 2018 is not a leap year.

'''
Global Variables
A variable that is defined outside of a function is called a global variable. It can be accessed inside the body of a function.

In the example, the variable a is a global variable because it is defined outside of the function prints_a. It is therefore accessible to prints_a, which will print the value of a.
'''

a = "Hello"

def prints_a():
  print(a)
  
# will print "Hello"
prints_a()

'''
Parameters as Local Variables
Function parameters behave identically to a function’s local variables. They are initialized with the values passed into the function when it was called.

Like local variables, parameters cannot be referenced from outside the scope of the function.

In the example, the parameter value is defined as part of the definition of my_function, and therefore can only be accessed within my_function. Attempting to print the contents of value from outside the function causes an error.
'''

def my_function(value):
  print(value)   
  
# Pass the value 7 into the function
my_function(7) 

# Causes an error as `value` no longer exists
print(value) 

'''

Cheatsheets / Learn Python 3

Function Arguments
Print PDF icon
Print Cheatsheet


Default argument is fallback value
In Python, a default parameter is defined with a fallback value as a default argument. Such parameters are optional during a function call. If no argument is provided, the default value is used, and if an argument is provided, it will overwrite the default value.
'''

def greet(name, msg="How do you do?"):
  print("Hello ", name + ', ' + msg)

greet("Ankit")
greet("Ankit", "How do you do?")

"""
this code will print the following for both the calls -
`Hello  Ankit, How do you do?`
"""
'''

Mutable Default Arguments
Python’s default arguments are evaluated only once when the function is defined, not each time the function is called. This means that if a mutable default argument is used and is mutated, it is mutated for all future calls to the function as well. This leads to buggy behaviour as the programmer expects the default value of that argument in each function call.
'''

# Here, an empty list is used as a default argument of the function.
def append(number, number_list=[]):
  number_list.append(number)
  print(number_list)
  return number_list

# Below are 3 calls to the `append` function and their expected and actual outputs:
append(5) # expecting: [5], actual: [5]
append(7) # expecting: [7], actual: [5, 7]
append(2) # expecting: [2], actual: [5, 7, 2]

'''
Python Default Arguments
A Python function cannot define a default argument in its signature before any required parameters that do not have a default argument. Default arguments are ones set using the form parameter=value. If no input value is provided for such arguments, it will take on the default value.
'''

# Correct order of declaring default argments in a function
def greet(name, msg = "Good morning!"):
  print("Hello ", name + ', ' + msg)
  
# The function can be called in the following ways
greet("Ankit")
greet("Kyla","How are you?")

# The following function definition would be incorrect
def greet(msg = "Good morning!", name):
  print("Hello ", name + ', ' + msg) 
# It would cause a "SyntaxError: non-default argument follows default argument"

'''
Python function default return value
If we do not not specify a return value for a Python function, it returns None. This is the default behaviour.
'''

# Function returning None
def my_function(): pass

print(my_function())

#Output 
None

'''
Python variable None check
To check if a Python variable is None we can make use of the statement variable is None.

If the above statement evaluates to True, the variable value is None.
'''

# Variable check for None
if variable_name is None:
    print "variable is None"
else:
    print "variable is NOT None"

'''
Python function arguments
A function can be called using the argument name as a keyword instead of relying on its positional value. Functions define the argument names in its composition then those names can be used when calling the function.
'''

# The function will take arg1 and arg2
def func_with_args(arg1, arg2):
  print(arg1 + ' ' + arg2)
  
func_with_args('First', 'Second')
# Prints:
# First Second

func_with_args(arg2='Second', arg1='First')
# Prints
# First Second


