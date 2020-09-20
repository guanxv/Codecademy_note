# 10 tip and tricks ---------- from corey --------------

#1.-----------

condition = True

x = 1 if condition else 0

print (x)

#2.--------------

num1 = 10_000_000_000
num2 = 100_000_000

total = num1 + num2

print (f'{total:,}')


#3.--------resource manage--------

with open('test.txt', 'r') as f:
	file_content = f.read()

#instead of doing

f = open('test.txt', 'r')
file_content = f.read()
f.close()

#4.-------------------------

names = ['corey', 'chris', 'dave', 'travis']

for index, name in enumerate(names, start = 1):
	print (index, name)

#5.-------------------

names = ['Peter Parker', 'Clark Kent', 'Wade Wilson', 'Bruce Wayne']
heroes = ['Spiderman', 'Superman', 'Deadpool', 'Batman']

for names, heroes in zip(names, heroes):
	print(f'{name} is actually {hero}')
	
# zip can also used for more than 2 lists
# zip will stop at shortest list

#6.----unpack values----------------

a, b = (1, 2)

print(a)
print(b)


a, _ = (1, 2)

print(a)
#print(b)

# avoid IDE to give waring of not using Varialbe b

a, b , c = (1, 2)

a, b , c = (1, 2 ,3 ,4, 5)

# both will raise error

a, b , *c = (1, 2 ,3 ,4, 5)

#a = 1
#b = 2
#c = [3,4,5]

a, b , *_ = (1, 2 ,3 ,4, 5)

#a = 1
#b = 2


a, b , *c, d = (1, 2 ,3 ,4, 5)

#a = 1
#b = 2
#c = [3, 4]
#d = 5

#7--------------------------------
class Person():
	pass

person = Person()

#you can set the attribute for class by doing this

person.first = 'Corey'
person.last = 'Schafer'

#you can also do this

first_key = 'first'
first_val = 'Corey'

setattr(person, 'first', 'Corey') #or to use the variable
setattr(person, first_key, first_val)

#to get the value

first = getattr(person, first_key)

#for the code could look like this
class Person():
	pass

person = Person()

person_info = {'first': 'Corey', 'last' : 'Schafer'}

for key, value in person_info.items():
	setattr(person, key, value)

for key in person_info.keys():
	print(getattr(person, key))


#8.---------------------------------------

username = input('Username:')
password = input('Password:')

print('Logging In...')

#to hide the password inputing on screen

from getpass import getpass

username = input('Username:')
password = getpass('Password:')

print('Logging In...')

#9---------------------------
#in command line
python password.py
python -m password # system will search to find the moudel

python -m password -c Debuge # -c will be for the password 

#10--------------------------

#study more on moudle

# in command line type python

#>>> import datetime
#>>> help(datetime)

#>>> dir(datetime)
# to list all the method and attribute


#>>> datetime.fold
#>>> datetime.max
#>>> datetime.today
# to understand the method and attribur

#>>> datetime.today()
# to use the method