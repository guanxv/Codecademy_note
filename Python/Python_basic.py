
def intersect # retrun item in both list

return [x for x in list1 if x in list2]


lst.sort()    #sort original list

new_lst = sorted(lst)   #return to a new list



# print things on one line
print(x, end='')

	
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

#-------------LAMBDA LAMBDA LAMBDA ---------------------------------

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
    AQ3WCTF,V
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


#--------------------LIST COMPREHENSION LIST COMPREHENSION LIST COMPREHENSION------------

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

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#PYTHON CHEATSHEET PYTHON CHEATSHEET PYTHON CHEATSHEET   #-#-#--#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
'''
Cheatsheets / Learn Python 3

Strings Strings Strings Strings Strings Strings Strings Strings Strings
Print PDF icon
Print Cheatsheet


In computer science, sequences of characters are referred to as strings. Strings can be any length and can include any character such as letters, numbers, symbols, and whitespace (spaces, tabs, new lines).

Escaping Characters
Backslashes (\) are used to escape characters in a Python string.

For instance, to print a string with quotation marks, the given code snippet can be used.
'''

txt = "She said \"Never let go\"."
print(txt) # She said "Never let go".
'''

The in Syntax
The in syntax is used to determine if a letter or a substring exists in a string. It returns True if a match is found, otherwise False is returned.
'''

game = "Popular Nintendo Game: Mario Kart"

print("l" in game) # Prints: True
print("x" in game) # Prints: False

'''
Indexing and Slicing Strings
Python strings can be indexed using the same notation as lists, since strings are lists of characters. A single character can be accessed with bracket notation ([index]), or a substring can be accessed using slicing ([start:end]).

Indexing with negative numbers counts from the end of the string.
'''

str = 'yellow'
str[1]     # => 'e'
str[-1]    # => 'w'
str[4:6]   # => 'ow'
str[:4]    # => 'yell'
str[-3:]   # => 'low'

'''
Iterate String
To iterate through a string in Python, “for…in” notation is used.
'''

str = "hello"
for c in str:
  print(c)
  
# h
# e
# l
# l
# o

'''
Built-in Function len()
In Python, the built-in len() function can be used to determine the length of an object. It can be used to compute the length of strings, lists, sets, and other countable objects.
'''

length = len("Hello")
print(length)
# Output: 5

colors = ['red', 'yellow', 'green']
print(len(colors))
# Output: 3

'''
String Concatenation
To combine the content of two strings into a single string, Python provides the + operator. This process of joining strings is called concatenation.
'''

x = 'One fish, '
y = 'two fish.'

z = x + y

print(z)
# Output: One fish, two fish.

'''
Immutable strings
Strings are immutable in Python. This means that once a string has been defined, it can’t be changed.

There are no mutating methods for strings. This is unlike data types like lists, which can be modified once they are created.


IndexError
When indexing into a string in Python, if you try to access an index that doesn’t exist, an IndexError is generated. For example, the following code would create an IndexError:
'''

fruit = "Berry"
indx = fruit[6]

'''
Python String .format()
The Python string method .format() replaces empty brace ({}) placeholders in the string with its arguments.

If keywords are specified within the placeholders, they are replaced with the corresponding named arguments to the method.
'''

msg1 = 'Fred scored {} out of {} points.'
msg1.format(3, 10)
# => 'Fred scored 3 out of 10 points.'

msg2 = 'Fred {verb} a {adjective} {noun}.'
msg2.format(adjective='fluffy', verb='tickled', noun='hamster')
# => 'Fred tickled a fluffy hamster.'

'''
String Method .lower()
The string method .lower() returns a string with all uppercase characters converted into lowercase.
'''

greeting = "Welcome To Chili's"

print(greeting.lower())
# Prints: welcome to chili's

'''
String Method .strip()
The string method .strip() can be used to remove characters from the beginning and end of a string.

A string argument can be passed to the method, specifying the set of characters to be stripped. With no arguments to the method, whitespace is removed.
'''

text1 = '   apples and oranges   '
text1.strip()       # => 'apples and oranges'

text2 = '...+...lemons and limes...-...'

# Here we strip just the "." characters
text2.strip('.')    # => '+...lemons and limes...-'

# Here we strip both "." and "+" characters
text2.strip('.+')   # => 'lemons and limes...-'

# Here we strip ".", "+", and "-" characters
text2.strip('.+-')  # => 'lemons and limes'

'''
String Method .title()
The string method .title() returns the string in title case. With title case, the first character of each word is capitalized while the rest of the characters are lowercase.
'''

my_var = "dark knight"
print(my_var.title()) 

# Prints: Dark Knight

'''
String Method .split()
The string method .split() splits a string into a list of items:

If no argument is passed, the default behavior is to split on whitespace.
If an argument is passed to the method, that value is used as the delimiter on which to split the string.
'''

text = "Silicon Valley"

print(text.split())     
# Prints: ['Silicon', 'Valley']

print(text.split('i'))  
# Prints: ['S', 'l', 'con Valley']

'''
Python string method .find()
The Python string method .find() returns the index of the first occurrence of the string passed as the argument. It returns -1 if no occurrence is found.
'''

mountain_name = "Mount Kilimanjaro"
print(mountain_name.find("o")) # Prints 1 in the console.

'''
String replace
The .replace() method is used to replace the occurence of the first argument with the second argument within the string.

The first argument is the old substring to be replaced, and the second argument is the new substring that will replace every occurence of the first one within the string.
'''

fruit = "Strawberry"
print(fruit.replace('r', 'R'))

# StRawbeRRy

'''
String Method .upper()
The string method .upper() returns the string with all lowercase characters converted to uppercase.
'''

dinosaur = "T-Rex"

print(dinosaur.upper()) 
# Prints: T-REX

'''
String Method .join()
The string method .join() concatenates a list of strings together to create a new string joined with the desired delimiter.

The .join() method is run on the delimiter and the array of strings to be concatenated together is passed in as an argument.
'''

x = "-".join(["Codecademy", "is", "awesome"])

print(x) 
# Prints: Codecademy-is-awesome


#
#
#
#
#
#
#
#
#
#
'''
Cheatsheets / Learn Python 3

Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists Lists
Print PDF icon
Print Cheatsheet


In Python, lists are ordered collections of items that allow for easy use of a set of data.

List values are placed in between square brackets [ ], separated by commas. It is good practice to put a space between the comma and the next value. The values in a list do not need to be unique (the same value can be repeated).

Empty lists do not contain any values within the square brackets.
'''

primes = [2, 3, 5, 7, 11]
print(primes)

empty_list = []

'''
Adding Lists Together
In Python, lists can be added to each other using the plus symbol +. As shown in the code block, this will result in a new list containing the same items in the same order with the first list’s items coming first.

Note: This will not work for adding one item at a time (use .append() method). In order to add one item, create a new list with a single value and then use the plus symbol to add the list.
'''

items = ['cake', 'cookie', 'bread']
total_items = items + ['biscuit', 'tart']
print(total_items)
# Result: ['cake', 'cookie', 'bread', 'biscuit', 'tart']

'''
Python Lists: Data Types
In Python, lists are a versatile data type that can contain multiple different data types within the same square brackets. The possible data types within a list include numbers, strings, other objects, and even other lists.
'''

numbers = [1, 2, 3, 4, 10]
names = ['Jenny', 'Sam', 'Alexis']
mixed = ['Jenny', 1, 2]
list_of_lists = [['a', 1], ['b', 2]]

'''
List Method .append()
In Python, you can add values to the end of a list using the .append() method. This will place the object passed in as a new element at the very end of the list. Printing the list afterwards will visually show the appended value. This .append() method is not to be confused with returning an entirely new list with the passed object.
'''

orders = ['daisies', 'periwinkle']
orders.append('tulips')
print(orders)
# Result: ['daisies', 'periwinkle', 'tulips']

'''
Aggregating Iterables Using zip()
In Python, data types that can be iterated (called iterables) can be used with the zip() function to aggregate data based on the iterables passed in.

As shown in the example, zip() is aggregating the data between the owners’ names and the dogs’ names to match the owner to their dogs. zip() returns an iterator containing the data based on what the user passes through and can be printed to visually represent the aggregated data. Empty iterables passed in will result in an empty iterator.
'''

owners_names = ['Jenny', 'Sam', 'Alexis']
dogs_names = ['Elphonse', 'Dr. Doggy DDS', 'Carter']
owners_dogs = zip(owners_names, dogs_names)
print(owners_dogs)
# Result: [('Jenny', 'Elphonse'), ('Sam', 'Dr.Doggy DDS'), ('Alexis', 'Carter')]

'''
List Item Ranges Including First or Last Item
In Python, when selecting a range of list items, if the first item to be selected is at index 0, no index needs to be specified before the :. Similarly, if the last item selected is the last item in the list, no index needs to be specified after the :.
'''

items = [1, 2, 3, 4, 5, 6]

# All items from index `0` to `3`
print(items[:4])

# All items from index `2` to the last item, inclusive
print(items[2:])

'''
List Method .count()
The .count() Python list method searches a list for whatever search term it receives as an argument, then returns the number of matching entries found.
'''

backpack = ['pencil', 'pen', 'notebook', 'textbook', 'pen', 'highlighter', 'pen']
numPen = backpack.count('pen')
print(numPen)
# Output: 3
'''

Determining List Length with len()
The Python len() function can be used to determine the number of items found in the list it accepts as an argument.
'''

knapsack = [2, 4, 3, 7, 10]
size = len(knapsack)
print(size) 
# Output: 5

'''
Zero-Indexing
In Python, list index begins at zero and ends at the length of the list minus one. For example, in this list, 'Andy' is found at index 2.
'''

names = ['Roger', 'Rafael', 'Andy', 'Novak']

'''
List Method .sort()
The .sort() Python list method will sort the contents of whatever list it is called on. Numerical lists will be sorted in ascending order, and lists of Strings will be sorted into alphabetical order. It modifies the original list, and has no return value.
'''

exampleList = [4, 2, 1, 3]
exampleList.sort()
print(exampleList)
# Output: [1, 2, 3, 4]

'''
List Indices
Python list elements are ordered by index, a number referring to their placement in the list. List indices start at 0 and increment by one.

To access a list element by index, square bracket notation is used: list[index].
'''

berries = ["blueberry", "cranberry", "raspberry"]

berries[0]   # "blueberry"
berries[2]   # "raspberry"

'''
Negative List Indices
Negative indices for lists in Python can be used to reference elements in relation to the end of a list. This can be used to access single list elements or as part of defining a list range. For instance:

To select the last element, my_list[-1].
To select the last three elements, my_list[-3:].
To select everything except the last two elements, my_list[:-2].
'''

soups = ['minestrone', 'lentil', 'pho', 'laksa']
soups[-1]   # 'laksa'
soups[-3:]  # 'lentil', 'pho', 'laksa'
soups[:-2]  # 'minestrone', 'lentil'

'''
List Slicing
A slice, or sub-list of Python list elements can be selected from a list using a colon-separated starting and ending point.

The syntax pattern is myList[START_NUMBER:END_NUMBER]. The slice will include the START_NUMBER index, and everything until but excluding the END_NUMBER item.

When slicing a list, a new list is returned, so if the slice is saved and then altered, the original list remains the same.
'''

tools = ['pen', 'hammer', 'lever']
tools_slice = tools[1:3] # ['hammer', 'lever']
tools_slice[0] = 'nail'

# Original list is unaltered:
print(tools) # ['pen', 'hammer', 'lever']
sorted() Function

'''
The Python sorted() function accepts a list as an argument, and will return a new, sorted list containing the same elements as the original. Numerical lists will be sorted in ascending order, and lists of Strings will be sorted into alphabetical order. It does not modify the original, unsorted list.
'''

unsortedList = [4, 2, 1, 3]
sortedList = sorted(unsortedList)
print(sortedList)
# Output: [1, 2, 3, 4]

#
#
#
#
#
#
#
#
#
#
'''
Cheatsheets / Learn Python 3

Dictionaries Dictionaries Dictionaries Dictionaries Dictionaries Dictionaries Dictionaries Dictionaries Dictionaries
Print PDF icon
Print Cheatsheet


Accessing and writing data in a Python dictionary
Values in a Python dictionary can be accessed by placing the key within square brackets next to the dictionary. Values can be written by placing key within square brackets next to the dictionary and using the assignment operator (=). If the key already exists, the old value will be overwritten. Attempting to access a value with a key that does not exist will cause a KeyError.

To illustrate this review card, the second line of the example code block shows the way to access the value using the key "song". The third line of the code block overwrites the value that corresponds to the key "song".
'''

my_dictionary = {"song": "Estranged", "artist": "Guns N' Roses"}
print(my_dictionary["song"])
my_dictionary["song"] = "Paradise City"

'''
Syntax of the Python dictionary
The syntax for a Python dictionary begins with the left curly brace ({), ends with the right curly brace (}), and contains zero or more key : value items separated by commas (,). The key is separated from the value by a colon (:).
'''

roaster = {"q1": "Ashley", "q2": "Dolly"}

'''
Merging Dictionaries with the .update() Method in Python
Given two dictionaries that need to be combined, Python makes this easy with the .update() function.

For dict1.update(dict2), the key-value pairs of dict2 will be written into the dict1 dictionary.

For keys in both dict1 and dict2, the value in dict1 will be overwritten by the corresponding value in dict2.
'''

dict1 = {'color': 'blue', 'shape': 'circle'}
dict2 = {'color': 'red', 'number': 42}

dict1.update(dict2)

# dict1 is now {'color': 'red', 'shape': 'circle', 'number': 42}

''' 
Dictionary value types
Python allows the values in a dictionary to be any type – string, integer, a list, another dictionary, boolean, etc. However, keys must always be an immutable data type, such as strings, numbers, or tuples.

In the example code block, you can see that the keys are strings or numbers (int or float). The values, on the other hand, are many varied data types.
'''

dictionary = {
  1: 'hello', 
  'two': True, 
  '3': [1, 2, 3], 
  'Four': {'fun': 'addition'}, 
  5.0: 5.5
}

'''
Python dictionaries
A python dictionary is an unordered collection of items. It contains data as a set of key: value pairs.
'''

my_dictionary = {1: "L.A. Lakers", 2: "Houston Rockets"}

'''
Dictionary accession methods
When trying to look at the information in a Python dictionary, there are multiple methods that access the dictionary and return lists of its contents.

.keys() returns the keys (the first object in the key-value pair), .values() returns the values (the second object in the key-value pair), and .items() returns both the keys and the values as a tuple.
'''

ex_dict = {"a": "anteater", "b": "bumblebee", "c": "cheetah"}

ex_dict.keys()
# ["a","b","c"]

ex_dict.values()
# ["anteater", "bumblebee", "cheetah"]

ex_dict.items()
# [("a","anteater"),("b","bumblebee"),("c","cheetah")]

'''
get() Method for Dictionary
Python provides a .get() method to access a dictionary value if it exists. This method takes the key as the first argument and an optional default value as the second argument, and it returns the value for the specified key if key is in the dictionary. If the second argument is not specified and key is not found then None is returned.
'''

# without default
{"name": "Victor"}.get("name")
# returns "Victor"

{"name": "Victor"}.get("nickname")
# returns None

# with default
{"name": "Victor"}.get("nickname", "nickname is not a key")
# returns "nickname is not a key"

'''
The .pop() Method for Dictionaries in Python
Python dictionaries can remove key-value pairs with the .pop() method. The method takes a key as an argument and removes it from the dictionary. At the same time, it also returns the value that it removes from the dictionary.
'''
famous_museums = {'Washington': 'Smithsonian Institution', 'Paris': 'Le Louvre', 'Athens': 'The Acropolis Museum'}
famous_museums.pop('Athens')
print(famous_museums) # {'Washington': 'Smithsonian Institution', 'Paris': 'Le Louvre'}

#
#
#
#
#
#
#
#
#
#
#
#
#
'''
Cheatsheets / Learn Python 3

Control Flow Control Flow Control Flow Control Flow Control Flow Control Flow Control Flow Control Flow Control Flow 
Print PDF icon
Print Cheatsheet


elif Statement
The Python elif statement allows for continued checks to be performed after an initial if statement. An elif statement differs from the else statement because another expression is provided to be checked, just as with the initial if statement.

If the expression is True, the indented code following the elif is executed. If the expression evaluates to False, the code can continue to an optional else statement. Multiple elif statements can be used following an initial if to perform a series of checks. Once an elif expression evaluates to True, no further elif statements are executed.
'''

# elif Statement

pet_type = "fish"

if pet_type == "dog":
  print("You have a dog.")
elif pet_type == "cat":
  print("You have a cat.")
elif pet_type == "fish":
  # this is performed
  print("You have a fish")
else:
  print("Not sure!")
  
'''  
Handling Exceptions in Python
A try and except block can be used to handle error in code block. Code which may raise an error can be written in the try block. During execution, if that code block raises an error, the rest of the try block will cease executing and the except code block will execute.
'''

def check_leap_year(year): 
  is_leap_year = False
  if year % 4 == 0:
    is_leap_year = True

try:
  check_leap_year(2018)
  print(is_leap_year) 
  # The variable is_leap_year is declared inside the function
except:
  print('Your code raised an error!')

'''  
or Operator
The Python or operator combines two Boolean expressions and evaluates to True if at least one of the expressions returns True. Otherwise, if both expressions are False, then the entire expression evaluates to False.
'''

True or True      # Evaluates to True
True or False     # Evaluates to True
False or False    # Evaluates to False
1 < 2 or 3 < 1    # Evaluates to True
3 < 1 or 1 > 6    # Evaluates to False
1 == 1 or 1 < 2   # Evaluates to True

'''
Equal Operator ==
The equal operator, ==, is used to compare two values, variables or expressions to determine if they are the same.

If the values being compared are the same, the operator returns True, otherwise it returns False.

The operator takes the data type into account when making the comparison, so a string value of "2" is not considered the same as a numeric value of 2.
'''

# Equal operator

if 'Yes' == 'Yes':
  # evaluates to True
  print('They are equal')

if (2 > 1) == (5 < 10):
  # evaluates to True
  print('Both expressions give the same result')

c = '2'
d = 2

if c == d:
  print('They are equal')
else:
  print('They are not equal')
  
'''  
Not Equals Operator !=
The Python not equals operator, !=, is used to compare two values, variables or expressions to determine if they are NOT the same. If they are NOT the same, the operator returns True. If they are the same, then it returns False.

The operator takes the data type into account when making the comparison so a value of 10 would NOT be equal to the string value "10" and the operator would return True. If expressions are used, then they are evaluated to a value of True or False before the comparison is made by the operator.
'''

# Not Equals Operator

if "Yes" != "No":
  # evaluates to True
  print("They are NOT equal")

val1 = 10
val2 = 20

if val1 != val2:
  print("They are NOT equal")

if (10 > 1) != (10 > 1000):
  # True != False
  print("They are NOT equal")
  
'''  
Comparison Operators
In Python, relational operators compare two values or expressions. The most common ones are:

< less than
> greater than
<= less than or equal to
>= greater than or equal too
If the relation is sound, then the entire expression will evaluate to True. If not, the expression evaluates to False.
'''

a = 2
b = 3
a < b  # evaluates to True
a > b  # evaluates to False
a >= b # evaluates to False
a <= b # evaluates to True
a <= a # evaluates to True

'''
if Statement
The Python if statement is used to determine the execution of code based on the evaluation of a Boolean expression.

If the if statement expression evaluates to True, then the indented code following the statement is executed.
If the expression evaluates to False then the indented code following the if statement is skipped and the program executes the next line of code which is indented at the same level as the if statement.
'''

# if Statement

test_value = 100

if test_value > 1:
  # Expression evaluates to True
  print("This code is executed!")

if test_value > 1000:
  # Expression evaluates to False
  print("This code is NOT executed!")

print("Program continues at this point.")

'''
else Statement
The Python else statement provides alternate code to execute if the expression in an if statement evaluates to False.

The indented code for the if statement is executed if the expression evaluates to True. The indented code immediately following the else is executed only if the expression evaluates to False. To mark the end of the else block, the code must be unindented to the same level as the starting if line.
'''

# else Statement

test_value = 50

if test_value < 1:
  print("Value is < 1")
else:
  print("Value is >= 1")

test_string = "VALID"

if test_string == "NOT_VALID":
  print("String equals NOT_VALID")
else:
  print("String equals something else!")

'''
and Operator
The Python and operator performs a Boolean comparison between two Boolean values, variables, or expressions. If both sides of the operator evaluate to True then the and operator returns True. If either side (or both sides) evaluates to False, then the and operator returns False. A non-Boolean value (or variable that stores a value) will always evaluate to True when used with the and operator.
'''

True and True     # Evaluates to True
True and False    # Evaluates to False
False and False   # Evaluates to False
1 == 1 and 1 < 2  # Evaluates to True
1 < 2 and 3 < 1   # Evaluates to False
"Yes" and 100     # Evaluates to True

'''
Boolean Values
Booleans are a data type in Python, much like integers, floats, and strings. However, booleans only have two values:

True
False
Specifically, these two values are of the bool type. Since booleans are a data type, creating a variable that holds a boolean value is the same as with other data types.
'''

is_true = True
is_false = False

print(type(is_true)) 
# will output: <class 'bool'>

'''
not Operator
The Python Boolean not operator is used in a Boolean expression in order to evaluate the expression to its inverse value. If the original expression was True, including the not operator would make the expression False, and vice versa.
'''

not True     # Evaluates to False
not False    # Evaluates to True
1 > 2        # Evaluates to False
not 1 > 2    # Evaluates to True
1 == 1       # Evaluates to True
not 1 == 1   # Evaluates to False
#
#
#
#
#
#
#
#
#
#
#
#
'''
Cheatsheets / Learn Python 3

Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops Loops 
Print PDF icon
Print Cheatsheet


break Keyword
In a loop, the break keyword escapes the loop, regardless of the iteration number. Once break executes, the program will continue to execute after the loop.

In this example, the output would be:

0
254
2
Negative number detected!
'''

numbers = [0, 254, 2, -1, 3]

for num in numbers:
  if (num < 0):
    print("Negative number detected!")
    break
  print(num)
  
# 0
# 254
# 2
# Negative number detected!

'''
Python List Comprehension
Python list comprehensions provide a concise way for creating lists. It consists of brackets containing an expression followed by a for clause, then zero or more for or if clauses: [EXPRESSION for ITEM in LIST <if CONDITIONAL>].

The expressions can be anything - any kind of object can go into a list.

A list comprehension always returns a list.
'''

# List comprehension for the squares of all even numbers between 0 and 9
result = [x**2 for x in range(10) if x % 2 == 0]

print(result)
# [0, 4, 16, 36, 64]

'''
Python For Loop
A Python for loop can be used to iterate over a list of items and perform a set of actions on each item. The syntax of a for loop consists of assigning a temporary value to a variable on each successive iteration.

When writing a for loop, remember to properly indent each action, otherwise an IndentationError will result.
'''

for <temporary variable> in <list variable>:
  <action statement>
  <action statement>
 
#each num in nums will be printed below
nums = [1,2,3,4,5]
for num in nums: 
  print(num)

'''
The Python continue Keyword
In Python, the continue keyword is used inside a loop to skip the remaining code inside the loop code block and begin the next loop iteration.
'''

big_number_list = [1, 2, -1, 4, -5, 5, 2, -9]

# Print only positive numbers:
for i in big_number_list:
  if i < 0:
    continue
  print(i)

'''
Python for Loops
Python for loops can be used to iterate over and perform an action one time for each element in a list.

Proper for loop syntax assigns a temporary value, the current item of the list, to a variable on each successive iteration: for <temporary value> in <a list>:

for loop bodies must be indented to avoid an IndentationError.
'''

dog_breeds = ["boxer", "bulldog", "shiba inu"]

# Print each breed:
for breed in dog_breeds:
  print(breed)
  
'''  
Python Loops with range().
In Python, a for loop can be used to perform an action a specific number of times in a row.

The range() function can be used to create a list that can be used to specify the number of iterations in a for loop.
'''

# Print the numbers 0, 1, 2:
for i in range(3):
  print(i)

# Print "WARNING" 3 times:
for i in range(3):
  print("WARNING")

'''
Infinite Loop
An infinite loop is a loop that never terminates. Infinite loops result when the conditions of the loop prevent it from terminating. This could be due to a typo in the conditional statement within the loop or incorrect logic. To interrupt a Python program that is running forever, press the Ctrl and C keys together on your keyboard.

Python while Loops
In Python, a while loop will repeatedly execute a code block as long as a condition evaluates to True.

The condition of a while loop is always checked first before the block of code runs. If the condition is not met initially, then the code block will never run.
'''

# This loop will only run 1 time
hungry = True
while hungry:
  print("Time to eat!")
  hungry = False

# This loop will run 5 times
i = 1
while i < 6:
  print(i)
  i = i + 1
  
'''  
Python Nested Loops
In Python, loops can be nested inside other loops. Nested loops can be used to access items of lists which are inside other lists. The item selected from the outer loop can be used as the list for the inner loop to iterate over.
'''

groups = [["Jobs", "Gates"], ["Newton", "Euclid"], ["Einstein", "Feynman"]]

# This outer loop will iterate over each list in the groups list
for group in groups:
  # This inner loop will go through each name in each list
  for name in group:
    print(name)

#
#
#
#
#
#
#
#
#
#
#
#
#
'''
Cheatsheets / Learn Python 3

Modules
Print PDF icon
Print Cheatsheet

Date and Time in Python
Python provides a module named datetime to deal with dates and times.

It allows you to set date ,time or both date and time using the date(),time()and datetime() functions respectively, after importing the datetime module .
'''

import datetime
feb_16_2019 = datetime.date(year=2019, month=2, day=16)
feb_16_2019 = datetime.date(2019, 2, 16)
print(feb_16_2019) #2019-02-16

time_13_48min_5sec = datetime.time(hour=13, minute=48, second=5)
time_13_48min_5sec = datetime.time(13, 48, 5)
print(time_13_48min_5sec) #13:48:05

timestamp= datetime.datetime(year=2019, month=2, day=16, hour=13, minute=48, second=5)
timestamp = datetime.datetime(2019, 2, 16, 13, 48, 5)
print (timestamp) #2019-01-02 13:48:05

'''
Aliasing with ‘as’ keyword
In Python, the as keyword can be used to give an alternative name as an alias for a Python module or function.

'''
# Aliasing matplotlib.pyplot as plt
from matplotlib import pyplot as plt
plt.plot(x, y)

# Aliasing calendar as c
import calendar as c
print(c.month_name[1])
'''

Import Python Modules
The Python import statement can be used to import Python modules from other files.

Modules can be imported in three different ways: import module, from module import functions, or from module import *. from module import * is discouraged, as it can lead to a cluttered local namespace and can make the namespace unclear.

'''
# Three different ways to import modules:
# First way
import module
module.function()

# Second way
from module import function
function()

# Third way
from module import *
function()

'''
random.randint() and random.choice()
In Python, the random module offers methods to simulate non-deterministic behavior in selecting a random number from a range and choosing a random item from a list.

The randint() method provides a uniform random selection from a range of integers. The choice() method provides a uniform selection of a random element from a sequence.
'''

# Returns a random integer N in a given range, such that start <= N <= end
# random.randint(start, end)
r1 = random.randint(0, 10)  
print(r1) # Random integer where 0 <= r1 <= 10

# Prints a random element from a sequence
seq = ["a", "b", "c", "d", "e"]
r2 = random.choice(seq)
print(r2) # Random element in the sequence

'''
Module importing
In Python, you can import and use the content of another file using import filename, provided that it is in the same folder as the current file you are writing.
'''

# file1 content
# def f1_function():
#	  return "Hello World"

# file2
import file1

# Now we can use f1_function, because we imported file1
f1_function()



