
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#PYTHON CLASSES PYTHON CLASS PYTHON CLASSES PYTHON CLASS #-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

class Shouter:
  def __init__(self):
    print("HELLO?!")

shout1 = Shouter()
# prints "HELLO?!"

shout2 = Shouter()
# prints "HELLO?!"

#---------------------

class Shouter:
  def __init__(self, phrase):
    # make sure phrase is a string
    if type(phrase) == str:

      # then shout it out
      print(phrase.upper())

shout1 = Shouter("shout")
# prints "SHOUT"

shout2 = Shouter("shout")
# prints "SHOUT"

shout3 = Shouter("let it all out")
# prints "LET IT ALL OUT"

#------------------------------

class Store:
  pass

alternative_rocks = Store()
isabelles_ices = Store()

alternative_rocks.store_name = 'Alternative Rocks'
isabelles_ices.store_name = 'Isabelle\'s Ices'

#-------------------------------

class NoCustomAttributes:
  pass

attributeless = NoCustomAttributes()

try:
  attributeless.fake_attribute
except AttributeError:
  print("This text gets printed!")

# prints "This text gets printed!"

#--------------------------------

class NoCustomAttributes:
  pass

attributeless = NoCustomAttributes()

try:
  attributeless.fake_attribute
except AttributeError:
  print("This text gets printed!")

# prints "This text gets printed!"

hasattr(attributeless, "fake_attribute")
# returns False

getattr(attributeless, "other_fake_attribute", 800)
# returns 800, the default value

#---------------------------

how_many_s = [{'s': False}, "sassafrass", 18, ["a", "c", "s", "d", "s"]]

no_s = 2

for item in how_many_s:
  if hasattr(item, "count"):
    no_s += item.count("s")

print(no_s)

#-----------------------------------


#Create a function named more_than_n that has three parameters named lst, item, and n.

#The function should return True if item appears in the list more than n times. The function should return False otherwise.

#Write your function here
def more_than_n(lst, item, n):
  if lst.count(item) > n:
    return True
  else:
    return False
  

#Uncomment the line below when your function is done
print(more_than_n([2, 4, 6, 2, 3, 2, 1, 2], 2, 3))

#---------------------------------

class SearchEngineEntry:
  def __init__(self, url):
    self.url = url

codecademy = SearchEngineEntry("www.codecademy.com")
wikipedia = SearchEngineEntry("www.wikipedia.org")

print(codecademy.url)
# prints "www.codecademy.com"

print(wikipedia.url)
# prints "www.wikipedia.org"
    
#----------------------------------

class SearchEngineEntry:
  secure_prefix = "https://"
  def __init__(self, url):
    self.url = url

  def secure(self):
    return "{prefix}{site}".format(prefix=self.secure_prefix, site=self.url)

codecademy = SearchEngineEntry("www.codecademy.com")

print(codecademy.secure())
# prints "https://www.codecademy.com"

print(wikipedia.secure())
# prints "https://www.wikipedia.org"

#------------------------------------

class Circle:
  pi = 3.14
  def __init__(self, diameter):
    print("Creating circle with diameter {d}".format(d=diameter))
    # Add assignment for self.radius here:
    self.radius = 0.5 * diameter
  def circumference(self):
    return (self.radius * 2 * self.pi)  #when use pi , should refer self.pi
	
#------------------------------

class FakeDict:
  pass

fake_dict = FakeDict()
fake_dict.attribute = "Cool"

dir(fake_dict)
# Prints ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'attribute']
  
fun_list = [10, "string", {'abc': True}]

type(fun_list)
# Prints <class 'list'>

dir(fun_list)  #print(dir(fun_list)
# Prints ['__add__', '__class__', [...], 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']

#--------------------------------
class Employee():
  def __init__(self, name):
    self.name = name

argus = Employee("Argus Filch")
print(argus)
# prints "<__main__.Employee object at 0x104e88390>"

#------------------------------

class Employee():
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name

argus = Employee("Argus Filch")
print(argus)
# prints "Argus Filch"

#-----------------------------

class Student():
  def __init__(self, name, year):
    self.name = name
    self.year = year
    self.grades = []
    
  def add_grade(self, grade):
    if type(grade) == Grade:
      self.grades.append(grade)
    else:
      print("This is Not a Grade")
    
  def get_average(self):
    sum = 0
    for score in self.grades:
      sum += score.score
    print( sum/len(self.grades))


class Grade():
  minimum_passing = 65
  def __init__(self, score):
    self.score = score
  
  def is_passing(self):
    if self.score >= 65:
      print("Contratulation! You Passed!")
    else:
      print("Sorry, You did not Pass.")

        
roger = Student("Roger van der Weyden", 10)
sandro = Student('Sandro Botticelli', 12)
pieter = Student('Pieter Bruegel the Elder', 8)

pieter_grade1 = Grade(100)
pieter_grade2 = Grade(50)

pieter.add_grade(pieter_grade1)
pieter.add_grade(pieter_grade2)

pieter.get_average()


#pieter_grade.is_passing()

#-------------------------------------------

class User:
  is_admin = False
  def __init__(self, username)
    self.username = username

class Admin(User):
  is_admin = True
  
issubclass(Admin, User)
#returns True

#-----------------------------
  
issubclass(ZeroDivisionError, Exception)
# Returns True

#---------------------------------

class KitchenException(Exception):
  """
  Exception that gets thrown when a kitchen appliance isn't working
  """

class MicrowaveException(KitchenException):
  """
  Exception for when the microwave stops working
  """

class RefrigeratorException(KitchenException):
  """
  Exception for when the refrigerator stops working
  """
  
#In this code, we define three exceptions. First, we define a KitchenException that acts as the parent to our other, specific kitchen appliance exceptions. KitchenException subclasses Exception, so it behaves in the same way that regular Exceptions do. Afterward we define MicrowaveException and RefrigeratorException as subclasses.

#Since our exceptions are subclassed in this way, we can catch any of KitchenException‘s subclasses by catching KitchenException. For example:
  
def get_food_from_fridge():
  if refrigerator.cooling == False:
    raise RefrigeratorException
  else:
    return food

def heat_food(food):
  if microwave.working == False:
    raise MicrowaveException
  else:
    microwave.cook(food)
    return food

try:
  food = get_food_from_fridge()
  food = heat_food(food)
except KitchenException:
  food = order_takeout()
  
#------------------------------------------

# Define your exception up here:
class OutOfStock(Exception):
  pass

# Update the class below to raise OutOfStock
class CandleShop:
  name = "Here's a Hot Tip: Buy Drip Candles"
  def __init__(self, stock):
    self.stock = stock
    
  def buy(self, color):
    if self.stock[color] == 0:
      raise OutOfStock
    else:
      self.stock[color] = self.stock[color] - 1

candle_shop = CandleShop({'blue': 6, 'red': 2, 'green': 0})
candle_shop.buy('blue')

# This should raise OutOfStock:
# candle_shop.buy('green')

#--------------------Redefind method in sub class------------------------

class User:
  def __init__(self, username, permissions):
    self.username = username
    self.permissions = permissions

  def has_permission_for(self, key):
    if self.permissions.get(key):
      return True
    else:
      return False
	  
class Admin(User):
  def has_permission_for(self, key): #this method is redefined 
    return True

#--------------------------------------
class Sink:
  def __init__(self, basin, nozzle):
    self.basin = basin
    self.nozzle = nozzle

class KitchenSink(Sink):
  def __init__(self, basin, nozzle, trash_compactor=None):
    super().__init__(basin, nozzle)
    if trash_compactor:
      self.trash_compactor = trash_compactor
	  
#----------------------------------------
#When two classes have the same method names and attributes, we say they implement the same interface. An interface in Python usually refers to the names of the methods and the arguments they take.

class InsurancePolicy:
  def __init__(self, price_of_item):
    self.price_of_insured_item = price_of_item
  
class VehicleInsurance(InsurancePolicy):
  def get_rate(self):
    return 0.001 * self.price_of_insured_item

class HomeInsurance(InsurancePolicy):
  def get_rate(self):
    return 0.00005 * self.price_of_insured_item

#-------------------------------------------------
#Look at all the different things that + does! The hope is that all of these things are, for the arguments given to them, the intuitive result of adding them together. Polymorphism is the term used to describe the same syntax (like the + operator here, but it could be a method name) doing different actions depending on the type of data.


# For an int and an int, + returns an int
2 + 4 == 6

# For a float and a float, + returns a float
3.1 + 2.1 == 5.2

# For a string and a string, + returns a string
"Is this " + "addition?" == "Is this addition?"

# For a list and a list, + returns a list
[1, 2] + [3, 4] == [1, 2, 3, 4]

#----
#Is the same operation happening for each? How is it different? How is it similar? Does using len() to refer to these different operations make sense?

a_list = [1, 18, 32, 12]
a_dict = {'value': True}
a_string = "Polymorphism is cool!"

print(len(a_list))

print(len(a_dict))

print(len(a_string))

#----------------------------------------------------------


class Color:
  def __init__(self, red, blue, green):
    self.red = red
    self.blue = blue
    self.green = green

  def __repr__(self):
    return "Color with RGB = ({red}, {blue}, {green})".format(red=self.red, blue=self.blue, green=self.green)

  def add(self, other):
    """
    Adds two RGB colors together
    Maximum value is 255
    """
    new_red = min(self.red + other.red, 255)
    new_blue = min(self.blue + other.blue, 255)
    new_green = min(self.green + other.green, 255)

    return Color(new_red, new_blue, new_green)

red = Color(255, 0, 0)
blue = Color(0, 255, 0)

magenta = red.add(blue)
print(magenta)
# Prints "Color with RGB = (255, 255, 0)"


#
#In this code we defined a Color class that implements an addition function. Unfortunately, red.add(blue) is a little verbose for something that we have an intuitive symbol for (i.e., the + symbol). Well, Python offers the dunder method __add__ for this very reason! If we rename the add() method above to something that looks like this:
#

class Color: 
  def __add__(self, other):
    """
    Adds two RGB colors together
    Maximum value is 255
    """
    new_red = min(self.red + other.red, 255)
    new_blue = min(self.blue + other.blue, 255)
    new_green = min(self.green + other.green, 255)

    return Color(new_red, new_blue, new_green)
	
#
red = Color(255, 0, 0)
blue = Color(0, 255, 0)
green = Color(0, 0, 255)
#We can add them together using the + operator!
#

# Color with RGB: (255, 255, 0)
magenta = red + blue

# Color with RGB: (0, 255, 255)
cyan = blue + green

# Color with RGB: (255, 0, 255)
yellow = red + green

# Color with RGB: (255, 255, 255)
white = red + blue + green

#---------------------------------------------------------------------

class UserGroup:
  def __init__(self, users, permissions):
    self.user_list = users
    self.permissions = permissions

  def __iter__(self):
    return iter(self.user_list)

  def __len__(self):
    return len(self.user_list)

  def __contains__(self, user):
    return user in self.user_list

#__init__, our constructor, which sets a list of users to the instance variable self.user_list and sets the group’s permissions when we create a new UserGroup.

#__iter__, the iterator, we use the iter() function to turn the list self.user_list into an iterator so we can use for user in user_group syntax. For more information on iterators, review Python’s documentation of Iterator Types.

#__len__, the length method, so when we call len(user_group) it will return the length of the underlying self.user_list list.

#__contains__, the check for containment, allows us to use user in user_group syntax to check if a User exists in the user_list we have.

class User:
  def __init__(self, username):
    self.username = username

diana = User('diana')
frank = User('frank')
jenn = User('jenn')

can_edit = UserGroup([diana, frank], {'can_edit_page': True})
can_delete = UserGroup([diana, jenn], {'can_delete_posts': True})

print(len(can_edit))
# Prints 2

for user in can_edit:
  print(user.username)
# Prints "diana" and "frank"

if frank in can_delete:
  print("Since when do we allow Frank to delete things? Does no one remember when he accidentally deleted the site?")

#--------------------------------------------






#___________________________________Class project_______________________________________________________


class Menu():
  def __init__(self, name, items, start_time, end_time):
    self.name = name
    self.items = items
    self.start_time = start_time
    self.end_time = end_time
  def __repr__(self):
    return "{menu} menu available from {start_time} to {end_time}".format(menu = self.name, start_time = self.start_time, end_time = self.end_time)
  def calculate_bill(self, purchased_items):
    tot_price = 0
    for item in purchased_items:
      tot_price += self.items.get(item, 0)
    return tot_price

class Franchise():
  def __init__(self, name, address, menus):
    self.name = name
    self.address = address
    self.menus = menus
  
  def __repr__(self):
    return "{restaurant} Located at {address}".format(restaurant = self.name, address = self.address)
  
  def available_menus(self, time):
    avaliable_menu = []
    for menu in self.menus:
      if (menu.start_time <= time) and (time <= menu.end_time):
        avaliable_menu.append(menu)
    return avaliable_menu
  
class Business():
  def __init__(self, name, franchises):
    self.name = name
    self.franchises = franchises
       
brunch = Menu('Brunch', {'pancakes': 7.50, 'waffles': 9.00, 'burger': 11.00, 'home fries': 4.50, 'coffee': 1.50, 'espresso': 3.00, 'tea': 1.00, 'mimosa': 10.50, 'orange juice': 3.50}, 11, 16 )
# we can save the menus into a bruch_menu variable, and then use the variable for the input of menu class


early_bird = Menu('Early Bird', {'salumeria plate': 8.00, 'salad and breadsticks (serves 2, no refills)': 14.00, 'pizza with quattro formaggi': 9.00, 'duck ragu': 17.50, 'mushroom ravioli (vegan)': 13.50, 'coffee': 1.50, 'espresso': 3.00,
},15 , 18)

dinner = Menu("Dinner", {'crostini with eggplant caponata': 13.00, 'ceaser salad': 16.00, 'pizza with quattro formaggi': 11.00, 'duck ragu': 19.50, 'mushroom ravioli (vegan)': 13.50, 'coffee': 2.00, 'espresso': 3.00,}, 17, 23)

kid = Menu('Kids', {'chicken nuggets': 6.50, 'fusilli with wild mushrooms': 12.00, 'apple juice': 3.00}, 11, 21)

arepas = Menu('Arepas', {'arepa pabellon': 7.00, 'pernil arepa': 8.50, 'guayanes arepa': 8.00, 'jamon arepa': 7.50
}, 10, 20 )

flagship_store = Franchise("Flagship Store", "1232 West End Road", [brunch, early_bird, dinner, kid])

new_installment = Franchise("New Installment", "12 East Mulberry Street", [brunch, early_bird, dinner, kid])

arepas_place = Franchise("Arepas Place", "189 Fitzgerald Avenue", [arepas])


basta_fazoolin = Business("Basta Fazoolin' with my Heart", [flagship_store , new_installment])

take_a_arepa = Business("Take a' Arepa", [arepas_place])

#print(kid)
#print(brunch.calculate_bill(["pancakes", "mimosa"]))
#print(early_bird.calculate_bill(["salumeria plate", "mushroom ravioli (vegan)"]))

#print(flagship_store)

#print(flagship_store.available_menus(12))
#print(flagship_store.available_menus(17))    




#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#PYTHON CLASSES CHEATSHEET PYTHON CLASS CHEATSHEET CLASS #-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Python repr method
The Python __repr__() method is used to tell Python what the string representation of the class should be. It can only have one parameter, self, and it should return a string.
'''

class Employee:
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name

john = Employee('John')
print(john) # John

'''
Python class methods
In Python, methods are functions that are defined as part of a class. It is common practice that the first argument of any method that is part of a class is the actual object calling the method. This argument is usually called self.
'''

# Dog class
class Dog:
  # Method of the class
  def bark(self):
    print("Ham-Ham")

# Create a new instance
charlie = Dog()

# Call the method
charlie.bark()
# This will output "Ham-Ham"

'''
Instantiate Python Class
In Python, a class needs to be instantiated before use.

As an analogy, a class can be thought of as a blueprint (Car), and an instance is an actual implementation of the blueprint (Ferrari).
'''

class Car:
  "This is an empty class"
  pass

# Class Instantiation
ferrari = Car()

'''
Python Class Variables
In Python, class variables are defined outside of all methods and have the same value for every instance of the class.

Class variables are accessed with the instance.variable or class_name.variable syntaxes.
'''

class my_class:
  class_variable = "I am a Class Variable!"
  
x = my_class()
y = my_class()

print(x.class_variable) #I am a Class Variable!
print(y.class_variable) #I am a Class Variable!

'''
Python init method
In Python, the .__init__() method is used to initialize a newly created object. It is called every time the class is instantiated.
'''

class Animal:
  def __init__(self, voice):
    self.voice = voice

# When a class instance is created, the instance variable
# 'voice' is created and set to the input value.
cat = Animal('Meow')
print(cat.voice) # Output: Meow

dog = Animal('Woof') 
print(dog.voice) # Output: Woof

'''
Python type() function
The Python type() function returns the data type of the argument passed to it.
'''

a = 1
print type(a) # <type 'int'>

a = 1.1
print type(a) # <type 'float'>

a = 'b'
print type(a) # <type 'str'>

a = None
print type(a) # <type 'NoneType'>

'''
Python class
In Python, a class is a template for a data type. A class can be defined using the class keyword.
'''

# Defining a class
class Animal:
  def __init__(self, name, number_of_legs):
    self.name = name
    self.number_of_legs = number_of_legs

'''
Python dir() function
In Python, the built-in dir() function, without any argument, returns a list of all the attributes in the current scope.

With an object as argument, dir() tries to return all valid object attributes.
'''

class Employee:
  def __init__(self, name):
    self.name = name

  def print_name(self):
    print("Hi, I'm " + self.name)


print(dir())
# ['Employee', '__builtins__', '__doc__', '__file__', '__name__', '__package__', 'new_employee']

print(dir(Employee))
# ['__doc__', '__init__', '__module__', 'print_name']

'''
__main__ in Python
In Python, __main__ is an identifier used to reference the current file context. When a module is read from standard input, a script, or from an interactive prompt, its __name__ is set equal to __main__.

Suppose we create an instance of a class called CoolClass. Printing the type() of the instance will result in:

'''
<class '__main__.CoolClass'>
'''

This means that the class CoolClass was defined in the current script file.

Super() Function in Python Inheritance
Python’s super() function allows a subclass to invoke its parent’s version of an overridden method.
'''

class ParentClass:
  def print_test(self):
    print("Parent Method")

class ChildClass(ParentClass):
  def print_test(self):
    print("Child Method")
    # Calls the parent's version of print_test()
    super().print_test() 
          
child_instance = ChildClass()
child_instance.print_test()
# Output:
# Child Method
# Parent Method
'''

User-defined exceptions in Python
In Python, new exceptions can be defined by creating a new class which has to be derived, either directly or indirectly, from Python’s Exception class.
'''

class CustomError(Exception):
  pass

'''
Polymorphism in Python
When two Python classes offer the same set of methods with different implementations, the classes are polymorphic and are said to have the same interface. An interface in this sense might involve a common inherited class and a set of overridden methods. This allows using the two objects in the same way regardless of their individual types.

When a child class overrides a method of a parent class, then the type of the object determines the version of the method to be called. If the object is an instance of the child class, then the child class version of the overridden method will be called. On the other hand, if the object is an instance of the parent class, then the parent class version of the method is called.
'''

class ParentClass:
  def print_self(self):
    print('A')

class ChildClass(ParentClass):
  def print_self(self):
    print('B')

obj_A = ParentClass()
obj_B = ChildClass()

obj_A.print_self() # A
obj_B.print_self() # B
'''

Dunder methods in Python
Dunder methods, which stands for “Double Under” (Underscore) methods, are special methods which have double underscores at the beginning and end of their names.

We use them to create functionality that can’t be represented as a normal method, and resemble native Python data type interactions. A few examples for dunder methods are: __init__, __add__, __len__, and __iter__.

The example code block shows a class with a definition for the __init__ dunder method.
'''

class String:
  # Dunder method to initialize object
  def __init__(self, string): 
    self.string = string
          
string1 = String("Hello World!") 
print(string1.string) # Hello World!
'''

Method Overriding in Python
In Python, inheritance allows for method overriding, which lets a child class change and redefine the implementation of methods already defined in its parent class.

The following example code block creates a ParentClass and a ChildClass which both define a print_test() method.

As the ChildClass inherits from the ParentClass, the method print_test() will be overridden by ChildClasssuch that it prints the word “Child” instead of “Parent”.
'''

class ParentClass:
  def print_self(self):
    print("Parent")

class ChildClass(ParentClass):
  def print_self(self):
    print("Child")

child_instance = ChildClass()
child_instance.print_self() # Child
'''

Python issubclass() Function
The Python issubclass() built-in function checks if the first argument is a subclass of the second argument.

In the example code block, we check that Member is a subclass of the Family class.
'''

class Family:
  def type(self):
    print("Parent class")
    
class Member(Family):
  def type(self):
    print("Child class")
     
print(issubclass(Member, Family)) # True
'''

Python Inheritance
Subclassing in Python, also known as “inheritance”, allows classes to share the same attributes and methods from a parent or superclass. Inheritance in Python can be accomplished by putting the superclass name between parentheses after the subclass or child class name.

In the example code block, the Dog class subclasses the Animal class, inheriting all of its attributes.
'''

class Animal: 
  def __init__(self, name, legs):
    self.name = name
    self.legs = legs
        
class Dog(Animal):
  def sound(self):
    print("Woof!")

Yoki = Dog("Yoki", 4)
print(Yoki.name) # YOKI
print(Yoki.legs) # 4
Yoki.sound() # Woof!
'''

+ Operator
In Python, the + operation can be defined for a user-defined class by giving that class an .__add()__ method.
'''

class A:
  def __init__(self, a):
    self.a = a 
  def __add__(self, other):
    return self.a + other.a 
    
obj1 = A(5)
obj2 = A(10)
print(obj1 + obj2) # 15


