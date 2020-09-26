# import os
# print(os.getcwd())

import re

text_to_search = """
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T

cat
mat
pat
bat
"""

emails = """
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
"""

urls = """
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
"""


# raw string (not handle the  "\")

# print ('\tTab')

# will print
#   Tab
# this is not a raw string

# print(r'\tTab')
# will print all the string

pattern = re.compile(
    r"abc"
)  # this search is case sensitive and order / sequence sensitive

matches = pattern.finditer(text_to_search)

for match in matches:

    pass
    # print(match)

# <re.Match object; span=(1, 4), match='abc'>

# print(text_to_search[1:4])

# abc

pattern = re.compile(r".")

matches = pattern.finditer(text_to_search)

for match in matches:

    pass
    # print(match)

# <re.Match object; span=(1, 2), match='a'>
# <re.Match object; span=(2, 3), match='b'>
# <re.Match object; span=(3, 4), match='c'>
# .....

# The right way to search . is to escape it.

pattern = re.compile(r"\.")

# to search url
pattern = re.compile(r"coreyms\.com")

# to search url
pattern = re.compile(r"\d")  # find any digits

pattern = re.compile(r"\D")  # find any non digits

pattern = re.compile(
    r"\bHa"
)  # match (word Boundary + Ha) will return 2 Ha, the third Ha is in middel of the word

# <re.Match object; span=(66, 68), match='Ha'>
# <re.Match object; span=(69, 71), match='Ha'>

pattern = re.compile(r"\BHa")  # match a ha in middle of word.

# <re.Match object; span=(71, 73), match='Ha'>

matches = pattern.finditer(text_to_search)

for match in matches:

    # print(match)
    pass


sentence = "Start a sentence and then bring it to an end"

pattern = re.compile(r"^Start")  # will find a string start with "Start"
pattern = re.compile(r"end$")  # will find a string end with "end"

matches = pattern.finditer(sentence)

for match in matches:

    # print(match)
    pass


pattern = re.compile(r"\d\d\d\D\d\d\d\D\d\d\d\d")  # find the phone number

pattern = re.compile(r"\d{3}.\d{3}.\d{4}")  # simplify the pattern with quantifier

pattern = re.compile(
    r"\d\d\d[-.]\d\d\d[-.]\d\d\d\d"
)  # only find the - or . seperator by using [] for charactor, [] only represent 1 charactor

pattern = re.compile(
    r"[89]\d\d[-.]\d\d\d[-.]\d\d\d\d"
)  # find numbers start with only 8 or 9

pattern = re.compile(r"[1-5]")  # find only 1 or 2 or 3 or 4 or 5 , only 1 digit

pattern = re.compile(r"[a-zA-Z]")  # find all upper or lower case letters

pattern = re.compile(
    r"[^a-zA-Z]"
)  # the ^ string will negate the set, in this case it will find anything but not a letter.

pattern = re.compile(r"[^b]at")  # will find every *at but not bat

pattern = re.compile(
    r"Mr\.?\s[A-Z]\w*"
)  # quantifier ? indicate with 0 or 1 "." , quantifier * indicate with 0 or more letter (\w)

pattern = re.compile(
    r"M(r|s|rs)\.?\s[A-Z]\w*"
)  # by using the group , we can match all the names, Mr Mrs Ms

pattern = re.compile(
    r"(Mr|Ms|Mrs)\.?\s[A-Z]\w*"
)  # To write this is more understandable


matches = pattern.finditer(text_to_search)

for match in matches:

    # print(match)
    pass


# emails
pattern = re.compile(
    r"[a-zA-Z]+@[a-zA-Z]+\.com"
)  # this will match only the first email
pattern = re.compile(
    r"[a-zA-Z.]+@[a-zA-Z]+\.(com|edu)"
)  # this will match first and second email (with . added in the set and edu in the group)
pattern = re.compile(
    r"[a-zA-Z0-9.-]+@[a-zA-Z-]+\.(com|edu|net)"
)  # this will match all 3 emails

matches = pattern.finditer(emails)

for match in matches:

    # print(match)
    pass

# urls

pattern = re.compile(r"https?://(www\.)?\w+\.(com|gov)")  # this will match all the urls

pattern = re.compile(
    r"https?://(www\.)?(\w+)(\.(com|gov))"
)  # adding parentacy for the parts we want to exact

matches = pattern.finditer(urls)

for match in matches:

    # print(match)
    # print(match.group(0))
    # print(match.group(1))
    # print(match.group(2))
    # print(match.group(3))
    pass

# result

# https://www.google.com
# www.
# google
# .com
# http://coreyms.com
# None
# coreyms
# .com
# https://youtube.com
# None
# youtube
# .com
# https://www.nasa.gov
# www.
# nasa
# .gov

subbed_urls = pattern.sub(r"\2\3", urls)  # \2 and \3 means for group 2 and 3.

# print(subbed_urls)

# result
# google.com
# coreyms.com
# youtube.com
# nasa.gov

with open("Python_re/data.txt", "r", encoding="utf8") as f:
    content = f.read()

    matches = pattern.finditer(content)

    for match in matches:

        # print(match)
        pass


pattern = re.compile(r"(Mr|Ms|Mrs)\.?\s[A-Z]\w*")
matches = pattern.findall(
    text_to_search
)  # if there is 1 group, findall will return only first group, if there are many groups , findall return groups in tuple. if no groups, find all return all matches
for match in matches:

    # print(match)
    pass



sentence = "Start a sentence and then bring it to an end"


pattern = re.compile(r"Start") 
pattern = re.compile(r"sentence") 

pattern = re.compile(r"Start", re.IGNORECASE) #this will ignore the case when search 


matches = pattern.match(sentence) #only match from begining of the string  #this will return none


matches = pattern.search("sentence") # will search and return first found
#result 
# <re.Match object; span=(0, 8), match='sentence'>



print(matches)









# ---- snippest ----

"""
.       - Any Character Except New Line
\d      - Digit (0-9)
\D      - Not a Digit (0-9)
\w      - Word Character (a-z, A-Z, 0-9, _)
\W      - Not a Word Character
\s      - Whitespace (space, tab, newline)
\S      - Not Whitespace (space, tab, newline)

\b      - Word Boundary
\B      - Not a Word Boundary
^       - Beginning of a String
$       - End of a String

[]      - Matches Characters in brackets
[^ ]    - Matches Characters NOT in brackets
|       - Either Or
( )     - Group

Quantifiers:
*       - 0 or More
+       - 1 or More
?       - 0 or One
{3}     - Exact Number
{3,4}   - Range of Numbers (Minimum, Maximum)


#### Sample Regexs ####

[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+


"""
