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





	







#Error Handling-------------------------------

try:
	pass

except Exception:
	pass

else:
	pass

finally:
	pass
	
#sample 

try:
	f = open('text.txt') #this file did not exist.
	var = bad_var #bad_var is not defined

except BadVarError:
	print('sorry. This file does not exist')
	
except FileNotFoundError:
	print('sorry. This file does not exist')

except Exception: #try to put more specific error on top , more general error at bottom
	print('sorry. something went wrong') 	
	
#Or this code can be

try:
	f = open('text.txt') #this file did not exist.
	var = bad_var #bad_var is not defined

except BadVarError as e:
	print(e)
	
except FileNotFoundError as e:
	print(e)

except Exception as e:
	print(e) 
	
# explain Else and Finally

try:
	f = open('text.txt') #this file did not exist.
	var = bad_var #bad_var is not defined

except BadVarError as e:
	print(e)
	
except FileNotFoundError as e:
	print(e)

except Exception as e:
	print(e) 

else: # if no error happend
	print(f.read())
	f.close()

finally: # else only run when no error happend, finally will run no mater what every happend
	print('Executing Finally...')
	
#raise your own exception

if f.name == 'currupt_file.txt':
	raise Exception
	
#----Function sample---------------------------------------------

#Write your function here
def double_index(lst, index):
  if index >= len(lst):
    return lst
  else:
    new_lst = lst[0:index]
    new_lst.append(lst[index]*2)
    new_lst = new_lst + lst[index+1:]
    return new_lst
  
def double_index_gux(lst, index):
  if index -1 <= len(lst):
    new_lst = lst
    new_lst[index] = new_lst[index] * 2
    return new_lst
  else:
    return lst

print(double_index_gux([3, 8, -10, 12], 2))

#Uncomment the line below when your function is done
print(double_index([3, 8, -10, 12], 2))
	
#------------ simple loop -------------------------
#Write your function here
def append_sum(lst):
  for i in range(3):
    new = lst[-1] + lst [-2]
    lst.append(new)
  return lst

#Uncomment the line below when your function is done
print(append_sum([1, 1, 2]))

#-------------lambda---------------------------------

#Write your function here

#def larger_list(lst1, lst2):
#  if len(lst1) >= len(lst2):
#    return lst1[-1]
#  else:
#    return lst2[-1]

larger_list = lambda lst1, lst2 : lst1[-1] if len(lst1) >= len(lst2) else lst2[-1]


#Uncomment the line below when your function is done
print(larger_list([4, 10, 2, 5], [-10, 2, 5, 10]))
	
#-----------------------------------------

lst.sort()

new_lst = sorted(lst)

#--------------------------------------

#Write your function here
every_three_nums = lambda start: list(range(start, 101, 3))

#def every_three_nums(start):
#  x = list(range(start, 101, 3))
#  return x


#Uncomment the line below when your function is done
print(every_three_nums(91))

#--------------------------------------

items_on_sale = ["blue_shirt", "striped_socks", "knit_dress", "red_headband", "dinosaur_onesie"]

print("Checking the sale list!")
for item in items_on_sale:
  print(item)
  if item == "knit_dress":
    AQ3WCTF,V"
	SZEX/JX. H,GHFCBDV SA
	
	
print("End of search!")

#------------------------
dog_breeds = ['french_bulldog', 'dalmation', 'shihtzu', 'poodle', 'collie']
for breed in dog_breeds:
    print(breed)
	
#--------------------------

for i in range(3):
  print("WARNING!")

#------------------------------
big_number_list = [1, 2, -1, 4, -5, 5, 2, -9]

for i in big_number_list:
  if i < 0:
    continue
  print(i)
  
# resutl  1 2 4 5 2

#-----------------------
dog_breeds = ['bulldog', 'dalmation', 'shihtzu', 'poodle', 'collie']

index = 0
while index < len(dog_breeds):
  print(dog_breeds[index])
  index += 1
  
#-------------------------

project_teams = [["Ava", "Samantha", "James"], ["Lucille", "Zed"], ["Edgar", "Gabriel"]]
for team in project_teams:


  for student in team:
    print(student)

#------------------------

sales_data = [[12, 17, 22], [2, 10, 3], [5, 12, 13]]
scoops_sold = 0

for location in sales_data:
  print(location)
  for data in location:
      scoops_sold += data

print(scoops_sold)


#--------------------list comprehension------------

words = ["@coolguy35", "#nofilter", "@kewldawg54", "reply", "timestamp", "@matchamom", "follow", "#updog"]
usernames = []

for word in words:
  if word[0] == '@':
    usernames.append(word)

#is equal with following code

usernames = [word for word in words if word[0] == '@']

#-------------------------------

>>> print(usernames)
["@coolguy35", "@kewldawg54", "@matchamom"]

messages = [user + " please follow me!" for user in usernames]
#-----------------------------

single_digits = range(10)

squares = []

for x in single_digits:
  print(x)
  squares.append(x**2)

print(squares)

cubes = [x**3 for x in single_digits]

print(cubes)

#-----------------------------

hairstyles = ["bouffant", "pixie", "dreadlocks", "crew", "bowl", "bob", "mohawk", "flattop"]

prices = [30, 25, 40, 20, 20, 35, 50, 35]

last_week = [2, 3, 5, 8, 4, 4, 6, 2]

total_price = 0
total_revenue  = 0
average_daily_revenue = 0

for price in prices:
  total_price += price

average_price = total_price / len(prices)


print("Average Haircut Price: " + str(average_price))

new_prices = [x -5 for x in prices]

#print(new_prices)

for i in range(len(hairstyles)):
  total_revenue += prices[i] * last_week[i]

print('Total Revenue: ' + str(total_revenue))

average_daily_revenue = total_revenue / 7

print('Daily Average Revenue: ' + str(average_daily_revenue))

cuts_under_30 = [hairstyles[i] for i in range(len(hairstyles)) if new_prices[i] < 30]

print(cuts_under_30)

#---------------------------

points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i]


#------------- tuple --------

# to create an 1 element tuple

one_element_tuple = (1,)

#-----------------------------

def letter_check(word, letter):
  contain = False
  for l in word:
    if l == letter:
      contain = True
  return contain

print(letter_check("game", "s"))

#--------------------------------

password = "theycallme\"crazy\"91"

#------------------------------

def common_letters(string_one, string_two):
  common = []
  for letter in string_one:
    if (letter in string_two) and not (letter in common):
      common.append(letter)
  return common

#------------------------------------
def password_generator(username):
  password = username[-1]
  for i in range(len(username)-1):
    password += username[i]
  return password

print (password_generator("abc"))
#resule cab

#-------------------------------------

#Here’s an example of .lower() in action:

#>>> favorite_song = 'SmOoTH'
#>>> favorite_song_lowercase = favorite_song.lower()
#>>> favorite_song_lowercase
#'smooth'
#Every character was changed to lowercase! It’s important to remember that string methods can only create new strings, they do not change the original string.


#---------------------------------

#.split() is performed on a string, takes one argument, and returns a list of substrings found between the given argument (which in the case of .split() is known as the delimiter). The following syntax should be used:

string_name.split(delimiter)
#If you do not provide an argument for .split() it will default to splitting at spaces.

line_one = "The sky has given over"

line_one_words = line_one.split()

print (line_one_words)

['The', 'sky', 'has', 'given', 'over']

#-------------------------------------


authors = "Audre Lorde, William Carlos Williams, Gabriela Mistral, Jean Toomer, An Qi, Walt Whitman, Shel Silverstein, Carmen Boullosa, Kamala Suraiyya, Langston Hughes, Adrienne Rich, Nikki Giovanni"

author_names = authors.split(',')

print(author_names)

author_last_names = [x.split()[-1] for x in author_names]

print (author_last_names)

#-----------------------------------------

#We can also split strings using escape sequences. Escape sequences are used to indicate that we want to split by something in a string that is not necessarily a character. The two escape sequences we will cover here are

\n Newline
\t Horizontal Tab



spring_storm_text = \
"""The sky has given over 
its bitterness. 
Out of the dark change 
all day long 
rain falls and falls 
as if it would never end. 
Still the snow keeps 
its hold on the ground. 
But water, water 
from a thousand runnels! 
It collects swiftly, 
dappled with black 
cuts a way for itself 
through green ice in the gutters. 
Drop after drop it falls 
from the withered grass-stems 
of the overhanging embankment."""

spring_storm_lines = spring_storm_text.split("\n")

print (spring_storm_lines)

['The sky has given over ', 'its bitterness. ', 'Out of the dark change ', 'all day long ', 'rain falls and falls ', 'as if it would never end. ', 'Still the snow keeps ', 'its hold on the ground. ', 'But water, water ', 'from a thousand runnels! ', 'It collects swiftly, ', 'dappled with black ', 'cuts a way for itself ', 'through green ice in the gutters. ', 'Drop after drop it falls ', 'from the withered grass-stems ', 'of the overhanging embankment.']

#--------------------------------------

'delimiter'.join(list_you_want_to_join)

>>> my_munequita = ['My', 'Spanish', 'Harlem', 'Mona', 'Lisa']
>>> ' '.join(my_munequita)
'My Spanish Harlem Mona Lisa'



>>> santana_songs_csv = ','.join(santana_songs)
>>> santana_songs_csv
'Oye Como Va,Smooth,Black Magic Woman,Samba Pa Ti,Maria Maria'



smooth_fifth_verse_lines = ['Well I\'m from the barrio', 'You hear my rhythm on your radio', 'You feel the turning of the world so soft and slow', 'Turning you \'round and \'round']

smooth_fifth_verse = '\n'.join(smooth_fifth_verse_lines)

print(smooth_fifth_verse)

["Well I'm from the barrio", 'You hear my rhythm on your radio', 'You feel the turning of the world so soft and slow', "Turning you 'round and 'round"]


#--------------------------------------------
>>> featuring = "           rob thomas                 "
>>> featuring.strip()
'rob thomas'

>>> featuring = "!!!rob thomas       !!!!!"
>>> featuring.strip('!')
'rob thomas     



love_maybe_lines = ['Always    ', '     in the middle of our bloodiest battles  ', 'you lay down your arms', '           like flowering mines    ','\n' ,'   to conquer me home.    ']

love_maybe_lines_stripped = [x.strip() for x in love_maybe_lines]


love_maybe_lines = ['Always    ', '     in the middle of our bloodiest battles  ', 'you lay down your arms', '           like flowering mines    ','\n' ,'   to conquer me home.    ']

#--------

love_maybe_lines_stripped = []

for line in love_maybe_lines:
  love_maybe_lines_stripped.append(line.strip())
  
love_maybe_full = '\n'.join(love_maybe_lines_stripped)

print(love_maybe_full)


#-----------------------------------

string_name.replace(character_being_replaced, new_character)

>>> with_spaces = "You got the kind of loving that can be so smooth"
>>> with_underscores = with_spaces.replace(' ', '_')
>>> with_underscores
'You_got_the_kind_of_loving_that_can_be_so_smooth'

#------------------------------------------

>>> 'smooth'.find('t')
'4'

>>>"smooth".find('oo')
'2'

#-------------------------------------

def favorite_song_statement(song, artist):
  return "My favorite song is {} by {}.".format(song, artist)
  

def favorite_song_statement(song, artist):
    return "My favorite song is {song} by {artist}.".format(song=song, artist=artist)

	
def favorite_song_statement(song, artist):
    return "My favorite song is {song_in_str} by {artist_in_str}.".format(song_in_str=song, artist_in_str=artist)
	
#---------------------------------------

highlighted_poems = "Afterimages:Audre Lorde:1997,  The Shadow:William Carlos Williams:1915, Ecstasy:Gabriela Mistral:1925,   Georgia Dusk:Jean Toomer:1923,   Parting Before Daybreak:An Qi:2014, The Untold Want:Walt Whitman:1871, Mr. Grumpledump's Song:Shel Silverstein:2004, Angel Sound Mexico City:Carmen Boullosa:2013, In Love:Kamala Suraiyya:1965, Dream Variations:Langston Hughes:1994, Dreamwood:Adrienne Rich:1987"

print(highlighted_poems)
print("\n")

highlighted_poems_list = highlighted_poems.split(',')

print(highlighted_poems_list)
print("\n")

highlighted_poems_stripped =[]

for a in highlighted_poems_list:
  highlighted_poems_stripped.append(a.strip())

print(highlighted_poems_stripped)
print("\n")

highlighted_poems_details = []
for a in highlighted_poems_stripped:
  highlighted_poems_details.append(a.split(':'))

print(highlighted_poems_details)
print("\n")

titles = []
poets = []
dates = []

for a in highlighted_poems_details:
  titles.append(a[0])
  poets.append(a[1])
  dates.append(a[2])

print(titles)
print("\n")

print(poets)
print("\n")

print(dates)
print("\n")

for i in range(len(titles)):
  print("The poem {TITLE} was published by {POET} in {DATE}.".format(TITLE = titles[i], POET = poets[i], DATE = dates[i] ))
  
#---------------------------------------------------------

daily_sales = \
"""Edith Mcbride   ;,;$1.21   ;,;   white ;,; 
09/15/17   ,Herbert Tran   ;,;   $7.29;,; 
white&blue;,;   09/15/17 ,Paul Clarke ;,;$12.52 
;,;   white&blue ;,; 09/15/17 ,Lucille Caldwell   
;,;   $5.13   ;,; white   ;,; 09/15/17,
Eduardo George   ;,;$20.39;,; white&yellow 
;,;09/15/17   ,   Danny Mclaughlin;,;$30.82;,;   
purple ;,;09/15/17 ,Stacy Vargas;,; $1.85   ;,; 
purple&yellow ;,;09/15/17,   Shaun Brock;,; 
$17.98;,;purple&yellow ;,; 09/15/17 , 
Erick Harper ;,;$17.41;,; blue ;,; 09/15/17, 
Michelle Howell ;,;$28.59;,; blue;,;   09/15/17   , 
Carroll Boyd;,; $14.51;,;   purple&blue   ;,;   
09/15/17   , Teresa Carter   ;,; $19.64 ;,; 
white;,;09/15/17   ,   Jacob Kennedy ;,; $11.40   
;,; white&red   ;,; 09/15/17, Craig Chambers;,; 
$8.79 ;,; white&blue&red   ;,;09/15/17   , Peggy Bell;,; $8.65 ;,;blue   ;,; 09/15/17,   Kenneth Cunningham ;,;   $10.53;,;   green&blue   ;,; 
09/15/17   ,   Marvin Morgan;,;   $16.49;,; 
green&blue&red   ;,;   09/15/17 ,Marjorie Russell 
;,; $6.55 ;,;   green&blue&red;,;   09/15/17 ,
Israel Cummings;,;   $11.86   ;,;black;,;  
09/15/17,   June Doyle   ;,;   $22.29 ;,;  
black&yellow ;,;09/15/17 , Jaime Buchanan   ;,;   
$8.35;,;   white&black&yellow   ;,;   09/15/17,   
Rhonda Farmer;,;$2.91 ;,;   white&black&yellow   
;,;09/15/17, Darren Mckenzie ;,;$22.94;,;green 
;,;09/15/17,Rufus Malone;,;$4.70   ;,; green&yellow 
;,; 09/15/17   ,Hubert Miles;,;   $3.59   
;,;green&yellow&blue;,;   09/15/17   , Joseph Bridges  ;,;$5.66   ;,; green&yellow&purple&blue 
;,;   09/15/17 , Sergio Murphy   ;,;$17.51   ;,;   
black   ;,;   09/15/17 , Audrey Ferguson ;,; 
$5.54;,;black&blue   ;,;09/15/17 ,Edna Williams ;,; 
$17.13;,; black&blue;,;   09/15/17,   Randy Fleming;,;   $21.13 ;,;black ;,;09/15/17 ,Elisa Hart;,; $0.35   ;,; black&purple;,;   09/15/17   ,
Ernesto Hunt ;,; $13.91   ;,;   black&purple ;,;   
09/15/17,   Shannon Chavez   ;,;$19.26   ;,; 
yellow;,; 09/15/17   , Sammy Cain;,; $5.45;,;   
yellow&red ;,;09/15/17 ,   Steven Reeves ;,;$5.50   
;,;   yellow;,;   09/15/17, Ruben Jones   ;,; 
$14.56 ;,;   yellow&blue;,;09/15/17 , Essie Hansen;,;   $7.33   ;,;   yellow&blue&red
;,; 09/15/17   ,   Rene Hardy   ;,; $20.22   ;,; 
black ;,;   09/15/17 ,   Lucy Snyder   ;,; $8.67   
;,;black&red  ;,; 09/15/17 ,Dallas Obrien ;,;   
$8.31;,;   black&red ;,;   09/15/17,   Stacey Payne 
;,;   $15.70   ;,;   white&black&red ;,;09/15/17   
,   Tanya Cox   ;,;   $6.74   ;,;yellow   ;,; 
09/15/17 , Melody Moran ;,;   $30.84   
;,;yellow&black;,;   09/15/17 , Louise Becker   ;,; 
$12.31 ;,; green&yellow&black;,;   09/15/17 ,
Ryan Webster;,;$2.94 ;,; yellow ;,; 09/15/17 
,Justin Blake ;,; $22.46   ;,;white&yellow ;,;   
09/15/17,   Beverly Baldwin ;,;   $6.60;,;   
white&yellow&black ;,;09/15/17   ,   Dale Brady   
;,;   $6.27 ;,; yellow   ;,;09/15/17 ,Guadalupe Potter ;,;$21.12   ;,; yellow;,; 09/15/17   , 
Desiree Butler ;,;$2.10   ;,;white;,; 09/15/17  
,Sonja Barnett ;,; $14.22 ;,;white&black;,;   
09/15/17, Angelica Garza;,;$11.60;,;white&black   
;,;   09/15/17   ,   Jamie Welch   ;,; $25.27   ;,; 
white&black&red ;,;09/15/17   ,   Rex Hudson   
;,;$8.26;,;   purple;,; 09/15/17 ,   Nadine Gibbs 
;,;   $30.80 ;,;   purple&yellow   ;,; 09/15/17   , 
Hannah Pratt;,;   $22.61   ;,;   purple&yellow   
;,;09/15/17,Gayle Richards;,;$22.19 ;,; 
green&purple&yellow ;,;09/15/17   ,Stanley Holland 
;,; $7.47   ;,; red ;,; 09/15/17 , Anna Dean;,;$5.49 ;,; yellow&red ;,;   09/15/17   ,
Terrance Saunders ;,;   $23.70  ;,;green&yellow&red 
;,; 09/15/17 ,   Brandi Zimmerman ;,; $26.66 ;,; 
red   ;,;09/15/17 ,Guadalupe Freeman ;,; $25.95;,; 
green&red ;,;   09/15/17   ,Irving Patterson 
;,;$19.55 ;,; green&white&red ;,;   09/15/17 ,Karl Ross;,;   $15.68;,;   white ;,;   09/15/17 , Brandy Cortez ;,;$23.57;,;   white&red   ;,;09/15/17, 
Mamie Riley   ;,;$29.32;,; purple;,;09/15/17 ,Mike Thornton   ;,; $26.44 ;,;   purple   ;,; 09/15/17, 
Jamie Vaughn   ;,; $17.24;,;green ;,; 09/15/17   , 
Noah Day ;,;   $8.49   ;,;green   ;,;09/15/17   
,Josephine Keller ;,;$13.10 ;,;green;,;   09/15/17 ,   Tracey Wolfe;,;$20.39 ;,; red   ;,; 09/15/17 ,
Ignacio Parks;,;$14.70   ;,; white&red ;,;09/15/17 
, Beatrice Newman ;,;$22.45   ;,;white&purple&red 
;,;   09/15/17, Andre Norris   ;,;   $28.46   ;,;   
red;,;   09/15/17 ,   Albert Lewis ;,; $23.89;,;   
black&red;,; 09/15/17,   Javier Bailey   ;,;   
$24.49   ;,; black&red ;,; 09/15/17   , Everett Lyons ;,;$1.81;,;   black&red ;,; 09/15/17 ,   
Abraham Maxwell;,; $6.81   ;,;green;,;   09/15/17   
,   Traci Craig ;,;$0.65;,; green&yellow;,; 
09/15/17 , Jeffrey Jenkins   ;,;$26.45;,; 
green&yellow&blue   ;,;   09/15/17,   Merle Wilson 
;,;   $7.69 ;,; purple;,; 09/15/17,Janis Franklin   
;,;$8.74   ;,; purple&black   ;,;09/15/17 ,  
Leonard Guerrero ;,;   $1.86   ;,;yellow  
;,;09/15/17,Lana Sanchez;,;$14.75   ;,; yellow;,;   
09/15/17   ,Donna Ball ;,; $28.10  ;,; 
yellow&blue;,;   09/15/17   , Terrell Barber   ;,; 
$9.91   ;,; green ;,;09/15/17   ,Jody Flores;,; 
$16.34 ;,; green ;,;   09/15/17,   Daryl Herrera 
;,;$27.57;,; white;,;   09/15/17   , Miguel Mcguire;,;$5.25;,; white&blue   ;,;   09/15/17 ,   
Rogelio Gonzalez;,; $9.51;,;   white&black&blue   
;,;   09/15/17   ,   Lora Hammond ;,;$20.56 ;,; 
green;,;   09/15/17,Owen Ward;,; $21.64   ;,;   
green&yellow;,;09/15/17,Malcolm Morales ;,;   
$24.99   ;,;   green&yellow&black;,; 09/15/17 ,   
Eric Mcdaniel ;,;$29.70;,; green ;,; 09/15/17 
,Madeline Estrada;,;   $15.52;,;green;,;   09/15/17 
, Leticia Manning;,;$15.70 ;,; green&purple;,; 
09/15/17 ,   Mario Wallace ;,; $12.36 ;,;green ;,; 
09/15/17,Lewis Glover;,;   $13.66   ;,;   
green&white;,;09/15/17,   Gail Phelps   ;,;$30.52   
;,; green&white&blue   ;,; 09/15/17 , Myrtle Morris 
;,;   $22.66   ;,; green&white&blue;,;09/15/17"""

#------------------------------------------------
# Start coding below!

daily_sales_replaced = daily_sales.replace(';,;','|')

daily_transactions = daily_sales_replaced.split(",")

daily_transactions_split = []

for transaction in daily_transactions:
  daily_transactions_split.append(transaction.split('|'))
  
transactions_clean = []

for transaction in daily_transactions_split:
  for item in transaction:
    transactions_clean.append(item.strip().strip("\n"))
    
customers = []
sales = []
thread_sold = []

for i in range(int((len(transactions_clean)+1)/4)):
  k = i*4
  j = k + 1
  l = j + 1
  customers.append(transactions_clean[k])
  sales.append(transactions_clean[j])
  thread_sold.append(transactions_clean[l])

#print(customers)
#print(sales)
#print(thread_sold)

total_sales = 0

for sale in sales:
  total_sales += float(sale.strip('$'))

#print(total_sales)

thread_sold_split = []

for color in thread_sold:
  if "&" not in color:
    thread_sold_split.append(color)
  else:
    thread_sold_split.append(color.split('&')[0])
    thread_sold_split.append(color.split('&')[1])

print(thread_sold_split)

def color_count(color):
  count = 0
  for col in thread_sold_split:
    if color == col:
      count += 1
  return count

print(color_count('white'))

color = ['red','yellow','green','white','black','blue','purple']

for c in color:
  print('{c} thread has been sold {times}'.format(c = c, times = color_count(c)))

  
#----------------

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Write your unique_english_letters function here:
def unique_english_letters(word):
  unique_letter = ''
  for letter in word:
    print(letter)
    if letter not in unique_letter:
      unique_letter += letter
    
  return len(unique_letter)

  


# Uncomment these function calls to test your function:
print(unique_english_letters("mississippi"))
# should print 4
#print(unique_english_letters("Apple"))
# should print 4

#-----------------------------------

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Write your unique_english_letters function here:
def unique_english_letters(word):
  uniques = 0
  for letter in letters:
    if letter in word:
      uniques += 1
  return uniques

# Uncomment these function calls to test your tip function:
print(unique_english_letters("mississippi"))
# should print 4
print(unique_english_letters("Apple"))
# should print 4

#----------------------------------------------

# Write your count_char_x function here:
def count_char_x(word, x):
  count = 0
  for letter in word:
    if letter == x:
      count += 1
  return count

# Uncomment these function calls to test your tip function:
print(count_char_x("mississippi", "s"))
# should print 4
print(count_char_x("mississippi", "m"))
# should print 1

#--------------------------------

# Write your count_multi_char_x function here:
def count_multi_char_x(word, x):
  count = 0
  length_letter = len(x)
  length_word = len(word)
  for i in range(0,length_word-length_letter):
    if word[i:i+length_letter] == x:
      count += 1
  return count  
  
# Uncomment these function calls to test your function:
print(count_multi_char_x("mississippi", "iss"))
# should print 2
#print(count_multi_char_x("apple", "pp"))
# should print 1

#----------------------------------

# Write your count_multi_char_x function here:
def count_multi_char_x(word, x):
  splits = word.split(x)
  print(splits)
  return(len(splits)-1)

# Uncomment these function calls to test your  function:
print(count_multi_char_x("mississippi", "iss"))
# should print 2
print(count_multi_char_x("apple", "pp"))
# should print 1

#--------------------------------------

# Write your substring_between_letters function here:
def substring_between_letters(word, start, end):
  s = 0
  e = 0
  s = word.find(start)
  print(word.find(start))
  e = word.find(end)
  print(word.find(end))
  if s >= 0 and e >= 0 :
    return word[s + 1 : e]
  else:
    return word

# Uncomment these function calls to test your function:
#print(substring_between_letters("apple", "p", "e"))
# should print "pl"
print(substring_between_letters("apple", "p", "c"))
# should print "apple"

#-----------------------------------

# Write your x_length_words function here:
def x_length_words(sentence , x):
  word_list = []
  word_list = sentence.split()
  for word in word_list:
    if len(word) >= x:
      continue
    else:
      return False
  return True


# Uncomment these function calls to test your tip function:
print(x_length_words("i like apples", 2))
# should print False
print(x_length_words("he likes apples", 2))
# should print True

#---------------------------------

# Write your every_other_letter function here:
def every_other_letter(word):
  every_other = ""
  for i in range(0, len(word), 2):
    every_other += word[i]
  return every_other

# Uncomment these function calls to test your tip function:
print(every_other_letter("Codecademy"))
# should print Cdcdm
print(every_other_letter("Hello world!"))
# should print Hlowrd
print(every_other_letter(""))
# should print 

#----------------------------------------------

# Write your reverse_string function here:


def reverse_string(word):
  rev_word = ""
  for i in range(len(word)):
    k = (i + 1) * -1
    rev_word += word[k]
    
  return rev_word
  
  
  # Uncomment these function calls to test your  function:
#print(reverse_string("Codecademy"))
# should print ymedacedoC
#print(reverse_string("Hello world!"))
# should print !dlrow olleH
#print(reverse_string(""))
# should print

#---------------------------------------------------
# Write your make_spoonerism function here:
def make_spoonerism(word1, word2):
  return word2[0]+word1[1:]+" "+word1[0]+word2[1:]

# Uncomment these function calls to test your tip function:
print(make_spoonerism("Codecademy", "Learn"))
# should print Lodecademy Cearn
print(make_spoonerism("Hello", "world!"))
# should print wello Horld!
print(make_spoonerism("a", "b"))
# should print b a

#--------------------------------------------

# Write your add_exclamation function here:
def add_exclamation(word):
  if len(word) >= 20:
    return word
  else:
    word += (20 - len(word)) * "!"
  return word
  
# Uncomment these function calls to test your function:
print(add_exclamation("Codecademy"))
# should print Codecademy!!!!!!!!!!
print(add_exclamation("Codecademy is the best place to learn"))
# should print Codecademy is the best place to learn
  
#-----------


def add_exclamation(word):
  while(len(word) < 20):
    word += "!"
  return word

#--------------------------------------
  
# Import random below:
import random

# Create random_list below:
random_list = [random.randint(1,101) for i in range(101)]

# Create randomer_number below:
randomer_number = random.choice(random_list)

# Print randomer_number below:
print(randomer_number)
  
#-------------------------------

import codecademylib3_seaborn

# Add your code below:
from matplotlib import pyplot as plt

import random

numbers_a = range(1, 13)

numbers_b = random.sample(range(1000), 12)

plt.plot(numbers_a, numbers_b)

plt.show()
#----------------------

# Import Decimal below:

from decimal import Decimal


# Fix the floating point math below:
two_decimal_points = Decimal('0.2') + Decimal('0.69')

print(two_decimal_points)

four_decimal_points = Decimal('0.53') * Decimal('0.65')

print(four_decimal_points)

#-------------------------------------
#--------script.py---------------

# Import library below:
from library import always_three


# Call your function below:
print(always_three())  

#---------library.py---------------
# Add your always_three() function below:
def always_three():
  return 3

#=---------------------------------------------
def unit_price(weight):
  unit_price_check = 0
  if weight <= 2:
    unit_price_check = 1.5
  elif weight <= 6:
    unit_price_check = 3
  elif weight <= 10:
    unit_price_check = 4
  elif weight > 10:
    unit_price_check = 4.75
  return unit_price_check

def ground_shipping(weight):
  cost = 0
  cost = unit_price(weight) * weight
  cost += 20
  return cost

def drone_shipping(weight):
  cost = 0
  cost = unit_price(weight) * 3 * weight
  cost += 0
  return cost

premium_ground_shipping = 125

#print(ground_shipping(10))
#print(drone_shipping(1.5))

def cheaper_shipping(weight):
  ground = ground_shipping(weight)
  drone = drone_shipping(weight)
  premium = premium_ground_shipping
  method = ""
  price = 0
  
  if (ground <= drone) and (ground <= premium):
    method = "Ground"
    price = ground
  elif (drone <= ground) and (drone <= premium):
    method = "Drone"
    price = drone
  elif (premium <= ground) and (premium <= drone):
    method = "Premium"
    price = premium
  print("{method} is the cheapest way to ship and the price is ${price}".format(method = method, price = price))

cheaper_shipping(10)
  
#--------------------------------
import random

money = 100

#Write your game of chance functions here
def flip_coin(guess, bet):
  i = 0
  i = random.randint(0,1)
  if i == 0:
    coin = "Heads"
  else:
    coin = "Tails"
  print("The Result is \"{coin}\"!".format(coin = coin))
  print("Your Guess is \"{guess}\"!".format(guess = guess))
    
  if guess.lower() !="Heads".lower() and guess.lower() !="Tails".lower():
    print("You have to bet on \"Heads\" or \"Tails\" !!!")
    return 0
 
  if guess.lower() == coin.lower():
    print("You Win!!!")
    return bet
  
  else:
    print("You Lose!!!")
    return bet*-1
#------------------------------
def cho_han(guess, bet):
  i = random.randint(1,6)
  j = random.randint(1,6)
  total = i + j
  
  print ("Dice 1 is {i}\
          Dice 2 is {j}\
          Total is {Total}\
          Your Bet is {guess}".format(i = i, j = j, Total = total, guess = guess))
  
  if guess.lower() != "odd" and guess.lower() != "even":
    print("You have to bet \"Odd\" or \"Even\"")
    return 0
  
  if (total % 2 == 0) and (guess.lower() == "even"):
    print("You Win!!!")
    return bet
  elif (total % 2 == 1) and (guess.lower() == "odd"):
    print("You Win!!!")
    return bet
  else:
    print("You Lost!!!")
    return bet*-1

#-------------------------
def card_bet(bet):
  i = random.randint(1,13)
  j = random.randint(1,13)
  
  card_list = [0, "A", 2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q" , "K"]
  
  print("You get a \"{card}\"!".format(card = card_list[i]))
  print("Computer get a \"{card}\"!".format(card = card_list[j]))
  if i > j:
    print ("You Win !!!")
    return bet
  elif i < j:
    print ("You Lost !!!")
    return bet*-1
  else:
    print ("Tie !!! Try Again.")
    return 0
#-----------------------
def roulette(bet):
  

#Call your game of chance functions here


#----------------------------------------

with open('welcome.txt') as text_file:
  text_data = text_file.read()
  
print(text_data)

with open('how_many_lines.txt') as lines_doc:
  for line in lines_doc.readlines():
    print(line)

with open('millay_sonnet.txt') as sonnet_doc:
  first_line = sonnet_doc.readline()
  second_line = sonnet_doc.readline()
  print(second_line)
 
 with open('bad_bands.txt', 'w') as bad_bands_doc:
  bad_bands_doc.write("abc")
  
 with open('cool_dogs.txt', 'a') as cool_dogs_file:
  cool_dogs_file.write("Air Buddy")
  


#--------------  
fun_cities_file = open('fun_cities.txt', 'a')

# We can now append a line to "fun_cities".
fun_cities_file.write("Montréal")

# But we need to remember to close the file
fun_cities_file.close()
#---------------

with open('logger.csv') as log_csv_file:
  a = log_csv_file.read()
print(a)

#-----------------
##########
users.csv

Name,Username,Email
Roger Smith,rsmith,wigginsryan@yahoo.com
Michelle Beck,mlbeck,hcosta@hotmail.com
Ashley Barker,a_bark_x,a_bark_x@turner.com
Lynn Gonzales,goodmanjames,lynniegonz@hotmail.com
###########

import csv

list_of_email_addresses = []
with open('users.csv', newline='') as users_csv:
  user_reader = csv.DictReader(users_csv)
  for row in user_reader:
    list_of_email_addresses.append(row['Email'])
	
	
import csv
isbn_list =[]
with open('books.csv') as books_csv:
  books_reader = csv.DictReader(books_csv, delimiter='@')
  for row in books_reader:
    isbn_list.append(row['ISBN'])
  
print(isbn_list)
    
#--------------------------------------

big_list = [{'name': 'Fredrick Stein', 'userid': 6712359021, 'is_admin': False}, {'name': 'Wiltmore Denis, 'userid': 2525942, 'is_admin': False}, {'name': 'Greely Plonk', 'userid': 15890235, 'is_admin': False}, {'name': 'Dendris Stulo', 'userid': 572189563, 'is_admin': True}] 

import csv

with open('output.csv', 'w') as output_csv:
  fields = ['name', 'userid', 'is_admin']
  output_writer = csv.DictWriter(output_csv, fieldnames=fields)

  output_writer.writeheader()
  for item in big_list:
    output_writer.writerow(item)

#---------------------------------------	
access_log = [{'time': '08:39:37', 'limit': 844404, 'address': '1.227.124.181'}, {'time': '13:13:35', 'limit': 543871, 'address': '198.51.139.193'}, {'time': '19:40:45', 'limit': 3021, 'address': '172.1.254.208'}, {'time': '18:57:16', 'limit': 67031769, 'address': '172.58.247.219'}, {'time': '21:17:13', 'limit': 9083, 'address': '124.144.20.113'}, {'time': '23:34:17', 'limit': 65913, 'address': '203.236.149.220'}, {'time': '13:58:05', 'limit': 1541474, 'address': '192.52.206.76'}, {'time': '10:52:00', 'limit': 11465607, 'address': '104.47.149.93'}, {'time': '14:56:12', 'limit': 109, 'address': '192.31.185.7'}, {'time': '18:56:35', 'limit': 6207, 'address': '2.228.164.197'}]
fields = ['time', 'address', 'limit']

import csv

with open("logger.csv", "w") as logger_csv:
  log_writer = csv.DictWriter(logger_csv, fieldnames = fields )
  log_writer.writeheader()
  for item in access_log:
    log_writer.writerow(item)

with open("logger.csv") as textfile:
  log_reader = textfile.read()
  print(log_reader)
  
#----------------------------------------

import json

with open('purchase_14781239.json') as purchase_json:
  purchase_data = json.load(purchase_json)

print(purchase_data['user'])
# Prints 'ellen_greg'

#-------------------------
turn_to_json = {
  'eventId': 674189,
  'dateTime': '2015-02-12T09:23:17.511Z',
  'chocolate': 'Semi-sweet Dark',
  'isTomatoAFruit': True
}


import json

with open('output.json', 'w') as json_file:
  json.dump(turn_to_json, json_file)

  
#---------------------

my_dict = {}

my_dict["new_key"] = "new_value"

print(my_dict)

#-------------------

user_ids = {"teraCoder": 9018293, "proProgrammer": 119238}

user_ids.update({"theLooper" : 138475, "stringQueen" : 85739})

print(user_ids)

#--------------------

names = ['Jenny', 'Alexus', 'Sam', 'Grace']
heights = [61, 70, 67, 64]

students = {key:value for key, value in zip(names, heights)}
#students is now {'Jenny': 61, 'Alexus': 70, 'Sam': 67, 'Grace': 64}

#---------------------------

drinks = ["espresso", "chai", "decaf", "drip"]
caffeine = [64, 40, 0, 120]

zipped_drinks = zip(drinks, caffeine)

drinks_to_caffeine = {drinks: x for drinks, x in zipped_drinks}

#-------------------------------------

songs = ["Like a Rolling Stone", "Satisfaction", "Imagine", "What's Going On", "Respect", "Good Vibrations"]
playcounts = [78, 29, 44, 21, 89, 5]

plays = {songs: x for songs, x in zip(songs, playcounts)}

#print(plays)

plays["Purple Haze"] = 1

plays["Respect"] = 94

library = {"The Best Songs" : plays, "Sunday Feelings" : {} }

print(library)

#----------------------------------


zodiac_elements = {"water": ["Cancer", "Scorpio", "Pisces"], "fire": ["Aries", "Leo", "Sagittarius"], "earth": ["Taurus", "Virgo", "Capricorn"], "air":["Gemini", "Libra", "Aquarius"]}


if "energy" in zodiac_elements:
  print(zodiac_elements["energy"])
else:
  print("Not a Zodiac element")
  
#-----------------------------------

key_to_check = "Landmark 81"
try:
  print(building_heights[key_to_check])
except KeyError:
  print("That key doesn't exist!")

#--------------------------

building_heights = {"Burj Khalifa": 828, "Shanghai Tower": 632, "Abraj Al Bait": 601, "Ping An": 599, "Lotte World Tower": 554.5, "One World Trade": 541.3}

#this line will return 632:
building_heights.get("Shanghai Tower")

#this line will return None:
building_heights.get("My House")

building_heights.get('Shanghai Tower', 0)
# return 632

building_heights.get('Mt Olympus', 0)
# return 0

building_heights.get('Kilimanjaro', 'No Value')
# return 'No Value'

#--------------------

raffle = {223842: "Teddy Bear", 872921: "Concert Tickets", 320291: "Gift Basket", 412123: "Necklace", 298787: "Pasta Maker"}

>>> raffle.pop(320291, "No Prize")
"Gift Basket"
>>> raffle
{223842: "Teddy Bear", 872921: "Concert Tickets", 412123: "Necklace", 298787: "Pasta Maker"}
>>> raffle.pop(100000, "No Prize")
"No Prize"
>>> raffle
{223842: "Teddy Bear", 872921: "Concert Tickets", 412123: "Necklace", 298787: "Pasta Maker"}
>>> raffle.pop(872921, "No Prize")
"Concert Tickets"
>>> raffle
{223842: "Teddy Bear", 412123: "Necklace", 298787: "Pasta Maker"}

#----------------------------


available_items = {"health potion": 10, "cake of the cure": 5, "green elixir": 20, "strength sandwich": 25, "stamina grains": 15, "power stew": 30}
health_points = 20

print(health_points)
print(available_items)

health_points += available_items.pop("stamina grains", 0)

print(health_points)
print(available_items)

health_points += available_items.pop("power stew", 0)

print(health_points)
print(available_items)

health_points += available_items.pop("mystic bread", 0)


print(health_points)
print(available_items)

#----------------------------------
test_scores = {"Grace":[80, 72, 90], "Jeffrey":[88, 68, 81], "Sylvia":[80, 82, 84], "Pedro":[98, 96, 95], "Martin":[78, 80, 78], "Dina":[64, 60, 75]}

>>> list(test_scores)
["Grace", "Jeffrey", "Sylvia", "Pedro", "Martin", "Dina"]

#---------------------------------

#Dictionaries also have a .keys() method that returns a dict_keys object. A dict_keys object is a view object, which provides a look at the current state of the dicitonary, without the user being able to modify anything. The dict_keys object returned by .keys() is a set of the keys in the dictionary. You cannot add or remove elements from a dict_keys object, but it can be used in the place of a list for iteration:

for student in test_scores.keys():
  print(student)
will yield:

"Grace"
"Jeffrey"
"Sylvia"
"Pedro"
"Martin"
"Dina"

#---------------------------------------
test_scores = {"Grace":[80, 72, 90], "Jeffrey":[88, 68, 81], "Sylvia":[80, 82, 84], "Pedro":[98, 96, 95], "Martin":[78, 80, 78], "Dina":[64, 60, 75]}

for score_list in test_scores.values():
  print(score_list)
  
  will yield:
F
[80, 72, 90]
[88, 68, 81]
[80, 82, 84]
[98, 96, 95]
[78, 80, 78]
[64, 60, 75]


#There is no built-in function to get all of the values as a list, but if you really want to, you can use:

list(test_scores.values())

#-------------------------------

biggest_brands = {"Apple": 184, "Google": 141.7, "Microsoft": 80, "Coca-Cola": 69.7, "Amazon": 64.8}

for company, value in biggest_brands.items():
  print(company + " has a value of " + str(value) + " billion dollars. ")
  
#which would yield this output:

Apple has a value of 184 billion dollars.
Google has a value of 141.7 billion dollars.
Microsoft has a value of 80 billion dollars.
Coca-Cola has a value of 69.7 billion dollars.
Amazon has a value of 64.8 billion dollars.

#-----------------------------------------------

pct_women_in_occupation = {"CEO": 28, "Engineering Manager": 9, "Pharmacist": 58, "Physician": 40, "Lawyer": 37, "Aerospace Engineer": 9}

for position , percentage in pct_women_in_occupation. items():
  print("Women make up {position} percent of {percentage}s.".format(position = percentage, percentage = position))

  
#------------------------

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
points = [1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 4, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10]

letters_low = []

for letter in letters:
  letters_low.append(letter.lower())

letters = letters + letters_low
points = points + points

#print(letters)
#print(points)

letter_to_points = {letters : x for letters, x in zip(letters, points )}

letter_to_points[' '] = 0

print(letter_to_points)

def score_word(word):
  point_total = 0
  for letter in word:
    point_total += letter_to_points.get(letter, 0)
  return point_total

brownie_points = score_word("BROWNIE")

print(brownie_points)
#should return 15

player_to_words = {
  "player1" : ["BLUE", "TENNIS", "EXIT"],
  "wordNerd" : ["EARTH", "EYES", "MACHINE"],
  "Lexi Con" : ["ERASER", "BELLY", "HUSKY"],
  "Prof Reader" : ["ZAP", "COMA", "PERIOD"]
}

player_to_points = {}

for player in player_to_words:
  player_points = 0
  for word in player_to_words[player]:
    word_points = score_word(word)
    player_points += word_points
  player_to_points[player] = player_points

print(player_to_points)

def play_word(player, word):
  #check player name
  try:
    player_to_words[player]
  except KeyError:
    print("Player Does Not Exist!")
    return
  #attatch the word
  if word in player_to_words[player]:
    print("The Word is Already Exist!")
    return 
  else:
    player_to_words[player].append(word)

#play_word("pyer1", "BLe")
#print(player_to_words)

letters_low = []

for letter in letters:
  letters_low.append(letter.lower())

print(letters_low)


#------------------------

# Write your max_key function here:
def max_key(my_dictionary):
  max_value = 0
  max_value = max(list(my_dictionary.values()))
  print(max_value)
  for key in my_dictionary:
    if my_dictionary[key] == max_value:
      return key
    
    
  
  
  # Uncomment these function calls to test your  function:
print(max_key({1:100, 2:1, 3:4, 4:10}))
# should print 1
print(max_key({"a":100, "b":10, "c":1000}))
# should print "c"

#---------------------------------------

# Write your max_key function here:
def max_key(my_dictionary):
  largest_key = float("-inf")
  largest_value = float("-inf")
  for key, value in my_dictionary.items():
    if value > largest_value:
      largest_value = value
      largest_key = key
  return largest_key

# Uncomment these function calls to test your  function:
print(max_key({1:100, 2:1, 3:4, 4:10}))
# should print 1
print(max_key({"a":100, "b":10, "c":1000}))
# should print "c"
  
  
#-----------------------------------------

# Write your count_first_letter function here:
def count_first_letter(names):
  first_letter = []
  name_dict = {}
  for key in names:
    if key[0] not in first_letter:
      first_letter.append(key[0])
      name_dict[key[0]] = 0
    
  for key, value in names.items():
    name_dict[key[0]] += len(value)
  
  return name_dict
    
    
    
# Uncomment these function calls to test your  function:
print(count_first_letter({"Stark": ["Ned", "Robb", "Sansa"], "Snow" : ["Jon"], "Lannister": ["Jaime", "Cersei", "Tywin"]}))
# should print {"S": 4, "L": 3}
print(count_first_letter({"Stark": ["Ned", "Robb", "Sansa"], "Snow" : ["Jon"], "Sannister": ["Jaime", "Cersei", "Tywin"]}))
# should print {"S": 7}

#----------------------------------
  
 # Write your count_first_letter function here:
def count_first_letter(names):
  letters = {}
  for key in names:
    first_letter = key[0]
    if first_letter not in letters:
      letters[first_letter] = 0
    letters[first_letter] += len(names[key])
  return letters

# Uncomment these function calls to test your  function:
print(count_first_letter({"Stark": ["Ned", "Robb", "Sansa"], "Snow" : ["Jon"], "Lannister": ["Jaime", "Cersei", "Tywin"]}))
# should print {"S": 4, "L": 3}
print(count_first_letter({"Stark": ["Ned", "Robb", "Sansa"], "Snow" : ["Jon"], "Sannister": ["Jaime", "Cersei", "Tywin"]}))
# should print {"S": 7}

#----------------------------
class Dog():
  dog_time_dilation = 7

  def time_explanation(self):
    print("Dogs experience {} years for every 1 human year.".format(self.dog_time_dilation))

pipi_pitbull = Dog()
pipi_pitbull.time_explanation()
# Prints "Dogs experience 7 years for every 1 human year."

#-------------------------------

class DistanceConverter:
  kms_in_a_mile = 1.609
  def how_many_kms(self, miles):
    return miles * self.kms_in_a_mile

converter = DistanceConverter()
kms_in_5_miles = converter.how_many_kms(5)
print(kms_in_5_miles)
# prints "8.045"

#------------------------------

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


#-------------------------------------------------

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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#------------------------------------RECURSION----------------------------------------------------------------------#
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


































'''

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-RECURSION CHEATSHEETS#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
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
'''







	

	

   