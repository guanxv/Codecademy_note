#check more info here https://docs.python.org/3/tutorial/modules.html

import my_module

course = ["History", "Math", "Physics", "CompSci"]

# when import like this:
#     print funciton will be Run

#     when you use funciton inside this module, you have to do my_module.find_index

index = my_module.find_index(course, "Math")

# print(index)

# print (my_module.test)

#---------------

import my_module as mm #import like this to give a short name of the main module

index = mm.find_index(course, "Math")

#---------------

from my_module import find_index # if you want to also import test, just write  ",test"

from my_module import find_index as fi , test # you can also import function and give nick name

#---------------

from my_module import * # this way of using is not suggested. because it not clear what is imported from that module. 

#---------------

#when import where did phython look for the files ?

import sys

print(sys.path) # this is the paths , python looking through to find the module. which is LOCAL ==> Venv ==> python path ==> std library

# if you sys can not find the module. you can do:

    # add the locaiton to sys.path

sys.path.append('/Users/Guanx/desktop/....')

# you can change the system path (enviromental variable)



import random # a sample of a standard library

random_course = random.choice(course)

print(random_course)


import math

rads = math.radians(90)

print(math.sin(rads))

import datetime
import calendar

today = datetime.date.today()
print(today)

print(calendar.isleap(2020))

import os

print(os.getcwd())

print(os.__file__) # if you print a module's __file__ fuction, you can find where it is on your PC.

import antigravity










