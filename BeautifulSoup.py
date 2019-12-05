WEB SCRAPING WITH BEAUTIFUL SOUP
Requests
In order to get the HTML of the website, we need to make a request to get the content of the webpage. To learn more about requests in a general sense, you can check out this article.

Python has a requests library that makes getting content really easy. All we have to do is import the library, and then feed in the URL we want to GET:

import requests

webpage = requests.get('https://www.codecademy.com/articles/http-requests')
print(webpage.text)
This code will print out the HTML of the page.

We don’t want to unleash a bunch of requests on any one website in this lesson, so for the rest of this lesson we will be scraping a local HTML file and pretending it’s an HTML file hosted online.

-------------------------------------------------------------------------------

WEB SCRAPING WITH BEAUTIFUL SOUP
The BeautifulSoup Object
When we printed out all of that HTML from our request, it seemed pretty long and messy. How could we pull out the relevant information from that long string?

BeautifulSoup is a Python library that makes it easy for us to traverse an HTML page and pull out the parts we’re interested in. We can import it by using the line:

from bs4 import BeautifulSoup
Then, all we have to do is convert the HTML document to a BeautifulSoup object!

If this is our HTML file, rainbow.html:

<body>
  <div>red</div>
  <div>orange</div>
  <div>yellow</div>
  <div>green</div>
  <div>blue</div>
  <div>indigo</div>
  <div>violet</div>
</body>
soup = BeautifulSoup("rainbow.html", "html.parser")
"html.parser" is one option for parsers we could use. There are other options, like "lxml" and "html5lib" that have different advantages and disadvantages, but for our purposes we will be using "html.parser" throughout.

With the requests skills we just learned, we can use a website hosted online as that HTML:

webpage = requests.get("http://rainbow.com/rainbow.html", "html.parser")
soup = BeautifulSoup(webpage.content)
When we use BeautifulSoup in combination with pandas, we can turn websites into DataFrames that are easy to manipulate and gain insights from.

-----------------------------------------------------------------------------------------

WEB SCRAPING WITH BEAUTIFUL SOUP
Object Types
BeautifulSoup breaks the HTML page into several types of objects.

Tags
A Tag corresponds to an HTML Tag in the original document. These lines of code:

soup = BeautifulSoup('<div id="example">An example div</div><p>An example p tag</p>')
print(soup.div)
Would produce output that looks like:

<div id="example">An example div</div>
Accessing a tag from the BeautifulSoup object in this way will get the first tag of that type on the page.

You can get the name of the tag using .name and a dictionary representing the attributes of the tag using .attrs:

print(soup.div.name)
print(soup.div.attrs)
div
{'id': 'example'}
NavigableStrings
NavigableStrings are the pieces of text that are in the HTML tags on the page. You can get the string inside of the tag by calling .string:

print(soup.div.string)
An example div

---------------------------------------------------------------------------------------

WEB SCRAPING WITH BEAUTIFUL SOUP
Navigating by Tags
To navigate through a tree, we can call the tag names themselves. Imagine we have an HTML page that looks like this:

<h1>World's Best Chocolate Chip Cookies</h1>
<div class="banner">
  <h1>Ingredients</h1>
</div>
<ul>
  <li> 1 cup flour </li>
  <li> 1/2 cup sugar </li>
  <li> 2 tbsp oil </li>
  <li> 1/2 tsp baking soda </li>
  <li> ½ cup chocolate chips </li> 
  <li> 1/2 tsp vanilla <li>
  <li> 2 tbsp milk </li>
</ul>
If we made a soup object out of this HTML page, we have seen that we can get the first h1 element by calling:

print(soup.h1)
<h1>World's Best Chocolate Chip Cookies</h1>
We can get the children of a tag by accessing the .children attribute:

for child in soup.ul.children:
    print(child)
<li> 1 cup flour </li>
<li> 1/2 cup sugar </li>
<li> 2 tbsp oil </li>
<li> 1/2 tsp baking soda </li>
<li> ½ cup chocolate chips </li> 
<li> 1/2 tsp vanilla <li>
<li> 2 tbsp milk </li>
We can also navigate up the tree of a tag by accessing the .parents attribute:

for parent in soup.li.parents:
    print(parent)
This loop will first print:

<ul>
<li> 1 cup flour </li>
<li> 1/2 cup sugar </li>
<li> 2 tbsp oil </li>
<li> 1/2 tsp baking soda </li>
<li> ½ cup chocolate chips </li>
<li> 1/2 tsp vanilla </li>
<li> 2 tbsp milk </li>
</ul>
Then, it will print the tag that contains the ul (so, the body tag of the document). Then, it will print the tag that contains the body tag (so, the html tag of the document).

------------------------------------------------------------------------------------------------------
WEB SCRAPING WITH BEAUTIFUL SOUP
Find All
If we want to find all of the occurrences of a tag, instead of just the first one, we can use .find_all().

This function can take in just the name of a tag and returns a list of all occurrences of that tag.

print(soup.find_all("h1"))
['<h1>World's Best Chocolate Chip Cookies</h1>', '<h1>Ingredients</h1>']
.find_all() is far more flexible than just accessing elements directly through the soup object. With .find_all(), we can use regexes, attributes, or even functions to select HTML elements more intelligently.

Using Regex
What if we want every <ol> and every <ul> that the page contains? We can select both of these types of elements with a regex in our .find_all():

import re
soup.find_all(re.compile("[ou]l"))
What if we want all of the h1 - h9 tags that the page contains? Regex to the rescue again!

import re
soup.find_all(re.compile("h[1-9]"))
Using Lists
We can also just specify all of the elements we want to find by supplying the function with a list of the tag names we are looking for:

soup.find_all(['h1', 'a', 'p'])
Using Attributes
We can also try to match the elements with relevant attributes. We can pass a dictionary to the attrs parameter of find_all with the desired attributes of the elements we’re looking for. If we want to find all of the elements with the "banner" class, for example, we could use the command:

soup.find_all(attrs={'class':'banner'})
Or, we can specify multiple different attributes! What if we wanted a tag with a "banner" class and the id "jumbotron"?

soup.find_all(attrs={'class':'banner', 'id':'jumbotron'})
Using A Function
If our selection starts to get really complicated, we can separate out all of the logic that we’re using to choose a tag into its own function. Then, we can pass that function into .find_all()!

def has_banner_class_and_hello_world(tag):
    return tag.attr('class') == "banner" and tag.string == "Hello world"

soup.find_all(has_banner_class_and_hello_world)
This command would find an element that looks like this:

<div class="banner">Hello world</div>
but not an element that looks like this:

<div>Hello world</div>
Or this:

<div class="banner">What's up, world!</div>

#-----------------------------------------------------------

WEB SCRAPING WITH BEAUTIFUL SOUP
Select for CSS Selectors
Another way to capture your desired elements with the soup object is to use CSS selectors. The .select() method will take in all of the CSS selectors you normally use in a .css file!

<h1 class='results'>Search Results for: <span class='searchTerm'>Funfetti</span></h1>
<div class='recipeLink'><a href="spaghetti.html">Funfetti Spaghetti</a></div>
<div class='recipeLink' id="selected"><a href="lasagna.html">Lasagna de Funfetti</a></div>
<div class='recipeLink'><a href="cupcakes.html">Funfetti Cupcakes</a></div>
<div class='recipeLink'><a href="pie.html">Pecan Funfetti Pie</a></div>
If we wanted to select all of the elements that have the class 'recipeLink', we could use the command:

soup.select(".recipeLink")
If we wanted to select the element that has the id 'selected', we could use the command:

soup.select("#selected")
Let’s say we wanted to loop through all of the links to these funfetti recipes that we found from our search.

for link in soup.select(".recipeLink > a"):
  webpage = requests.get(link)
  new_soup = BeautifulSoup(webpage)
This loop will go through each link in each .recipeLink div and create a soup object out of the webpage it links to. So, it would first make soup out of <a href="spaghetti.html">Funfetti Spaghetti</a>, then <a href="lasagna.html">Lasagna de Funfetti</a>, and so on.
#-----------------------------------------------------

import requests
from bs4 import BeautifulSoup

prefix = "https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/"
webpage_response = requests.get('https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/shellter.html')

webpage = webpage_response.content
soup = BeautifulSoup(webpage, "html.parser")

turtle_links = soup.find_all("a")
links = []
#go through all of the a tags and get the links associated with them:
for a in turtle_links:
    links.append(prefix+a["href"])
    
#Define turtle_data:
turtle_data = {}

#follow each link:
for link in links:
  webpage = requests.get(link)
  turtle = BeautifulSoup(webpage.content, "html.parser")
  #Add your code here:
  turtle_name = turtle.select(".name")[0]
  print (turtle_name)
  turtle_data[turtle_name] = []
  
print(turtle_data)

#---------------------------

WEB SCRAPING WITH BEAUTIFUL SOUP
Reading Text
When we use BeautifulSoup to select HTML elements, we often want to grab the text inside of the element, so that we can analyze it. We can use .get_text() to retrieve the text inside of whatever tag we want to call it on.

<h1 class="results">Search Results for: <span class='searchTerm'>Funfetti</span></h1>
If this is the HTML that has been used to create the soup object, we can make the call:

soup.get_text()
Which will return:

'Search Results for: Funfetti'
Notice that this combined the text inside of the outer h1 tag with the text contained in the span tag inside of it! Using get_text(), it looks like both of these strings are part of just one longer string. If we wanted to separate out the texts from different tags, we could specify a separator character. This command would use a . character to separate:

soup.get_text('|')
Now, the command returns:

'Search Results for: |Funfetti'
```

#-----------------------------------------------------

import requests
from bs4 import BeautifulSoup

prefix = "https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/"
webpage_response = requests.get('https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/shellter.html')

webpage = webpage_response.content
soup = BeautifulSoup(webpage, "html.parser")

turtle_links = soup.find_all("a")
links = []
#go through all of the a tags and get the links associated with them"
for a in turtle_links:
    links.append(prefix+a["href"])
    
#Define turtle_data:
turtle_data = {}

#follow each link:
for link in links:
  webpage = requests.get(link)
  turtle = BeautifulSoup(webpage.content, "html.parser")
  turtle_name = turtle.select(".name")[0].get_text()
  
  stats = turtle.find("ul")
  stats_text = stats.get_text("|")
  turtle_data[turtle_name] = stats_text.split("|")
  

#--------------------------------------------------------------------

LEARN WEB SCRAPING WITH BEAUTIFUL SOUP
Chocolate Scraping with Beautiful Soup
After eating chocolate bars your whole life, you’ve decided to go on a quest to find the greatest chocolate bar in the world.

You’ve found a website that has over 1700 reviews of chocolate bars from all around the world. It’s displayed in the web browser on this page.

The data is displayed in a table, instead of in a csv or json. Thankfully, we have the power of BeautifulSoup that will help us transform this webpage into a DataFrame that we can manipulate and analyze.

The rating scale is from 1-5, as described in this review guide. A 1 is “unpleasant” chocolate, while a 5 is a bar that transcends “beyond the ordinary limits”.

Some questions we thought about when we found this dataset were: Where are the best cocoa beans grown? Which countries produce the highest-rated bars? What’s the relationship between cocoa solids percentage and rating?

Can we find a way to answer these questions, or uncover more questions, using BeautifulSoup and Pandas?

Tasks
18/19Complete
Mark the tasks as complete by checking them off
Make Some Chocolate Soup
1.
Explore the webpage displayed in the browser. What elements could be useful to scrape here? Which elements do we not want to scrape?

If you want to use your browser to inspect the website, you may need a refresher on DevTools.

2.
Let’s make a request to this site to get the raw HTML, which we can later turn into a BeautifulSoup object.

The URL is:

https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/cacao/index.html
You can pass this into the .get() method of the requests module to get the HTML.

To make a request to a webpage, you can use the syntax:

webpage = requests.get("http://websitetoscrape.com")
3.
Create a BeautifulSoup object called soup to traverse this HTML.

Use "html.parser" as the parser, and the content of the response you got from your request as the document.

To make the object, you can use syntax like:

soup = BeautifulSoup(your_document, "your_parser")
your_document should be the .content of the webpage response you got from the last step.

4.
If you want, print out the soup object to explore the HTML.

So many table rows! You’re probably very relieved that we don’t have to scrape this information by hand.

How are ratings distributed?
5.
How many terrible chocolate bars are out there? And how many earned a perfect 5? Let’s make a histogram of this data.

The first thing to do is to put all of the ratings into a list.

Use a command on the soup object to get all of the tags that contain the ratings.

It looks like all of the rating tds have a class "Rating".

Remember that we can use .find_all() to get all elements of a class "ClassName" with this syntax:

soup.find_all(attrs={"class": "ClassName"})
6.
Create an empty list called ratings to store all the ratings in.

7.
Loop through the ratings tags and get the text contained in each one. Add it to the ratings list.

As you do this, convert the rating to a float, so that the ratings list will be numerical. This should help with calculations later.

The first element of your tags list probably contains the header string "Ratings", so we probably should leave this off the list. Start your loop at element 1 of the list instead.

You can cast a string x to a float with this syntax:

float(x)
You can get the text of a BeautifulSoup tag using .get_text()

8.
Using Matplotlib, create a histogram of the ratings values:

plt.hist(ratings)
Remember to show the plot using plt.show()!

Your plot will show up at localhost in the web browser. You will have to navigate away from the cacao ratings webpage to see it.

Which chocolatier makes the best chocolate?
9.
We want to now find the 10 most highly rated chocolatiers. One way to do this is to make a DataFrame that has the chocolate companies in one column, and the ratings in another. Then, we can do a groupby to find the ones with the highest average rating.

First, let’s find all the tags on the webpage that contain the company names.

It seems like all of the company tds have the class name "Company".

We can use .select() to grab BeautifulSoup elements by CSS selector:

soup.select(".ClassName")
Do this with the class name "Company" to get all of the right tags.

10.
Just like we did with ratings, we now want to make an empty list to hold company names.

11.
Loop through the tags containing company names, and add the text from each tag to the list you just created.

We might want to use syntax like

for td in company_tags[1:]:
  companies.append(td.get_text())
12.
Create a DataFrame with a column “Company” corresponding to your companies list, and a column “Ratings” corresponding to your ratings list.

You can make a DataFrame with defined columns using this syntax:

d = {"Column 1 Name": column_1_list, "Column 2 Name": column_2_list}
your_df = pd.DataFrame.from_dict(d)
13.
Use .groupby to group your DataFrame by Company and take the average of the grouped ratings.

Then, use the .nlargest command to get the 10 highest rated chocolate companies. Print them out.

Look at the hint if you get stuck on this step!

Your Pandas commands should look something like:

mean_vals = cacao_df.groupby("Company").Rating.mean()
ten_best = mean_ratings.nlargest(10)
print(ten_best)
Is more cacao better?
14.
We want to see if the chocolate experts tend to rate chocolate bars with higher levels of cacao to be better than those with lower levels of cacao.

It looks like the cocoa percentages are in the table under the Cocoa Percent column.

Using the same methods you used in the last couple of tasks, create a list that contains all of the cocoa percentages. Store each percent as an integer, after stripping off the % character.

You’ll have to access the tags with class "CocoaPercent" and loop through them to get the text.

cocoa_percents = []
cocoa_percent_tags = soup.select(".CocoaPercent")

for td in cocoa_percent_tags[1:]:
  percent = int(td.get_text().strip('%'))
  cocoa_percents.append(percent)
15.
Add the cocoa percentages as a column called "CocoaPercentage" in the DataFrame that has companies and ratings in it.

You can add the pairing "CocoaPercentage":cocoa_percents to the dictionary you used to create the DataFrame.

16.
Make a scatterplot of ratings (your_df.Rating) vs percentage of cocoa (your_df.CocoaPercentage).

You can do this in Matplotlib with these commands:

plt.scatter(df.CocoaPercentage, df.Rating)
plt.show()
Call plt.clf() to clear the figure between showing your histogram and this scatterplot.

Remember that your plots will show up at the address localhost in the web browser.

17.
Is there any correlation here? We can use some numpy commands to draw a line of best-fit over the scatterplot.

Copy this code and paste it after you create the scatterplot, but before you call .show():

z = np.polyfit(df.CocoaPercentage, df.Rating, 1)
line_function = np.poly1d(z)
plt.plot(df.CocoaPercentage, line_function(df.CocoaPercentage), "r--")
18.
Is there any correlation here? We can use some numpy commands to draw a line of best-fit over the scatterplot.

Copy this code and paste it after you create the scatterplot, but before you call .show():

z = np.polyfit(df.CocoaPercentage, df.Rating, 1)
line_function = np.poly1d(z)
plt.plot(df.CocoaPercentage, line_function(df.CocoaPercentage), "r--")
Explore!
19.
We have explored a couple of the questions about chocolate that inspired us when we looked at this chocolate table.

What other kinds of questions can you answer here? Try to use a combination of BeautifulSoup and Pandas to explore some more.

For inspiration: Where are the best cocoa beans grown? Which countries produce the highest-rated bars?

#------------------------------------------------------------------------------------------------------------------

import codecademylib3_seaborn
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

webpage_response = requests.get('https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/cacao/index.html')
webpage = webpage_response.content
soup = BeautifulSoup(webpage, 'html.parser')

ratings =[]
companys = []
cacao_percents = []
cacao_pp = []

for rating in soup.find_all("td", attrs={"class":"Rating"}):
		ratings.append(rating.get_text())

ratings.remove("Rating")

ratings2 =[]
for rating in ratings:
  ratings2.append(float(rating))

#plt.hist(ratings2)
#plt.show()

for company in soup.find_all("td", attrs={"class":"Company"}):
		companys.append(company.get_text())

companys.pop(0)


df1 = pd.DataFrame({
  "Company":companys,
  "Ratings":ratings2
})

top_company = df1.groupby("Company").Ratings.mean()
ten_best = top_company.nlargest(10)

#print (ten_best)

for cacao_p in soup.find_all("td",attrs = {'class' : 'CocoaPercent'}):
  cacao_percents.append(cacao_p.get_text())
cacao_percents.pop(0)

for cacao_p in cacao_percents:
  cacao = cacao_p.split("%")[0]
  cacao_pp.append(float(cacao))

#print (cacao_pp)

df1['CocoaPercentage'] = cacao_pp

plt.scatter(df1.CocoaPercentage, df1.Ratings)

z = np.polyfit(df1.CocoaPercentage, df1.Ratings, 1)
line_function = np.poly1d(z)
plt.plot(df1.CocoaPercentage, line_function(df1.CocoaPercentage), "r--")

plt.show()

plt.clf()

#------------note from corey schafer--------------------------------------------------------------

pip install beautifulsoup4
			lxml
			html5lib
			requests

from bs4 import BeautifulSoup
improt requests
with open ('simple.html') as html_file:
	soup = BeautifulSoup(html.file, 'lxml')

soup.prettify()

match = soup.title # get first title
print(match.text)

match = soup.div
match = soup.find('div', class = "footer")

article = soup.find('div', class = 'article')

#<div class = "article">
#	<h2> <a herf = "article_l.html"> .....
#	<p> ..... </p>
#	</div>

headline = article.h2.a.text
summary = article.p.text

for article in soup.find_all('div', class = 'article'):
	headline = article.h2.a.text
	summary = article.p.text
	
	
	
	

from bs4 import BeautifulSoup
import requests

source = requests.get('https://xxx.xxx.com').text
soup = BeautifulSoup(source, 'lxml')
soup.prettify()

article = soup.find('article')
headline = article.h2.a.text
summary = articl.find('div', class = 'entry-content').p.text
vid_src = article.find('iframe', class = 'youtubeplay')['src']
#to access the results as a dictionary, use ['src'] at the end

#http://www.youtube.com/embedkdkjfdorij....

vid_id = vid.split(',')[4]

yt-link = f'https://youtube.com/watch?v={vid_id}'

article = soup.find('article')
for article in soup.find_all('article'):
	head = ....
	summary = ....
	vid_src = .....
	
	try:
		vid_scr = .....
		vid_id = ......
		yt-link = f'https://youtube.com/watch?v={vid_id}'
	except Exception as e:
		yt-link = None
		
		








    

