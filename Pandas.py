#----------------------------------------------------------------
import codecademylib
import pandas as pd

orders = pd.read_csv('shoefly.csv')

print (orders.head())

#emails = orders["email"]

emails = orders.email

frances_palmer = orders[(orders.first_name == "Frances") & (orders.last_name == "Palmer")]

comfy_shoes = orders[orders.shoe_type.isin(["clogs", "boots","ballet flats"])]

#------------------------------------------------
MODIFYING DATAFRAMES
Applying a Lambda to a Row
We can also operate on multiple columns at once. If we use apply without specifying a single column and add the argument axis=1, the input to our lambda function will be an entire row, not a column. To access particular values of the row, we use the syntax row.column_name or row[‘column_name’].

Suppose we have a table representing a grocery list:

Item	Price	Is taxed?
Apple	1.00	No
Milk	4.20	No
Paper Towels	5.00	Yes
Light Bulbs	3.75	Yes
If we want to add in the price with tax for each line, we’ll need to look at two columns: Price and Is taxed?.

If Is taxed? is Yes, then we’ll want to multiply Price by 1.075 (for 7.5% sales tax).

If Is taxed? is No, we’ll just have Price without multiplying it.

We can create this column using a lambda function and the keyword axis=1:

df['Price with Tax'] = df.apply(lambda row:
     row['Price'] * 1.075
     if row['Is taxed?'] == 'Yes'
     else row['Price'],
     axis=1
)
Instructions
1.
If an employee worked for more than 40 hours, she needs to be paid overtime (1.5 times the normal hourly wage).

For instance, if an employee worked for 43 hours and made $10/hour, she would receive $400 for the first 40 hours that she worked, and an additional $45 for the 3 hours of overtime, for a total for $445.

Create a lambda function total_earned that accepts an input row with keys hours_worked and hourly_wage and uses an if statement to calculate the hourly wage.

If we were writing a regular function, it would look like this:

def total_earned(row):
   if row['hours_worked'] <= 40:
       return row['hours_worked'] * \
           row['hourly_wage']
    else:
        return (40 * row['hourly_wage'])\
            + (row['hours_worked'] - 40) * \
            (row['hourly_wage'] * 1.50)
2.
Use the lambda function total_earned and apply to add a column total_earned to df with the total amount earned by each employee.

#------------------------------------------------------------------
import codecademylib
import pandas as pd

df = pd.read_csv('employees.csv')

total_earned = lambda row: 
	row['hours_worked'] * row['hourly_wage'] 
	if row['hours_worked'] <= 40 
  else 40 * row['hourly_wage'] + (row['hours_worked'] - 40) * 1.5 * row['hourly_wage']

df['total_earned'] = df.apply(total_earned, axis = 1)

print(df)

#-----------------------------------------------------------------

MODIFYING DATAFRAMES
Review
Great job! In this lesson, you learned how to modify an existing DataFrame. Some of the skills you’ve learned include:

Adding columns to a DataFrame
Using lambda functions to calculate complex quantities
Renaming columns
Let’s practice what you just learned!

Instructions
1.
Once more, you’ll be the data analyst for ShoeFly.com, a fictional online shoe store.

More messy order data has been loaded into the variable orders. Examine the first 5 rows of the data using print and head.

2.
Many of our customers want to buy vegan shoes (shoes made from materials that do not come from animals). Add a new column called shoe_source, which is vegan if the materials is not leather and animal otherwise.

The following lambda function might be helpful:

mylambda = lambda x: 'animal' \
           if x == 'leather' else 'vegan'
3.
Our marketing department wants to send out an email to each customer. Using the columns last_name and gender create a column called salutation which contains Dear Mr. <last_name> for men and Dear Ms. <last_name> for women.

Here are some examples:

last_name	gender	salutation
Smith	Male	Dear Mr. Smith
Jones	Female	Dear Ms. Jones
The following lambda function might be helpful:

mylambda = lambda row: \
  'Dear Mr. {}'.format(row.last_name) \
  if row.gender == 'male' \
  else 'Dear Ms. {}'.format(row.last_name)
Make sure your strings are exactly the same! Spacing and punctuation matter!

#----------------------------------------------------------

import codecademylib
import pandas as pd

inventory = pd.read_csv('inventory.csv')
print (inventory.head(10))

staten_island = inventory.iloc[:10]
product_request = staten_island['product_description']

#print(product_request)

seed_request = inventory[(inventory.location == 'Brooklyn') &(inventory.product_type == 'seeds')]
#print(seed_request)

check_stock = lambda row: True if row['quantity'] > 0 else False 
inventory['in_stock'] = inventory.apply(check_stock , axis = 1)
#print(inventory)

inventory['total_value'] = inventory.price * inventory.quantity
print(inventory)

combine_lambda = lambda row:\
	'{}-{}'.format(row.product_type, row.product_description)

inventory['full_description'] = inventory.apply(combine_lambda, axis = 1)
print(inventory)


#-------------------------------------------------------

AGGREGATES IN PANDAS
Review
This lesson introduced you to aggregates in Pandas. You learned:

How to perform aggregate statistics over individual rows with the same value using groupby.
How to rearrange a DataFrame into a pivot table, a great way to compare data across two dimensions.
Instructions
1.
Let’s examine some more data from ShoeFly.com. This time, we’ll be looking at data about user visits to the website (the same dataset that you saw in the introduction to this lesson).

The data is a DataFrame called user_visits. Use print and head() to examine the first few rows of the DataFrame.

2.
The column utm_source contains information about how users got to ShoeFly’s homepage. For instance, if utm_source = Facebook, then the user came to ShoeFly by clicking on an ad on Facebook.com.

Use a groupby statement to calculate how many visits came from each of the different sources. Save your answer to the variable click_source.

Remember to use reset_index()!

3.
Paste the following code into script.py so that you can see the results of your previous groupby:

print(click_source)
4.
Our Marketing department thinks that the traffic to our site has been changing over the past few months. Use groupby to calculate the number of visits to our site from each utm_source for each month. Save your answer to the variable click_source_by_month.

5.
The head of Marketing is complaining that this table is hard to read. Use pivot to create a pivot table where the rows are utm_source and the columns are month. Save your results to the variable click_source_by_month_pivot.

It should look something like this:

utm_source	1 - January	2 - February	3 - March
email	…	…	…
facebook	…	…	…
google	…	…	…
twitter	…	…	…
yahoo	…	…	…
6.
View your pivot table by pasting the following code into script.py:

print(click_source_by_month_pivot)

#-----------------------------------

import codecademylib
import pandas as pd

user_visits = pd.read_csv('page_visits.csv')

print (user_visits.head())

click_source = user_visits.groupby('utm_source').id.count().reset_index()

print(click_source)

click_source_by_month = user_visits.groupby(['utm_source', 'month']).id.count().reset_index()

click_source_by_month_pivot = click_source_by_month.pivot(
	columns= 'month',
	index = 'utm_source',
	values = 'id').reset_index()

print (click_source_by_month_pivot)


#-----------------------------------------------

DATA MANIPULATION WITH PANDAS
A/B Testing for ShoeFly.com
Our favorite online shoe store, ShoeFly.com is performing an A/B Test. They have two different versions of an ad, which they have placed in emails, as well as in banner ads on Facebook, Twitter, and Google. They want to know how the two ads are performing on each of the different platforms on each day of the week. Help them analyze the data using aggregate measures.

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
10/11Complete
Mark the tasks as complete by checking them off
Analyzing Ad Sources
1.
Examine the first few rows of ad_clicks.

Try pasting the following code:

print(ad_clicks.head())
2.
Your manager wants to know which ad platform is getting you the most views.

How many views (i.e., rows of the table) came from each utm_source?

Try using the following code:

ad_clicks.groupby('utm_source')\
    .user_id.count()\
    .reset_index()
3.
If the column ad_click_timestamp is not null, then someone actually clicked on the ad that was displayed.

Create a new column called is_click, which is True if ad_click_timestamp is not null and False otherwise.

Try using the following code:

ad_clicks['is_click'] = ~ad_clicks\
   .ad_click_timestamp.isnull()
The ~ is a NOT operator, and isnull() tests whether or not the value of ad_click_timestamp is null.

4.
We want to know the percent of people who clicked on ads from each utm_source.

Start by grouping by utm_source and is_click and counting the number of user_id‘s in each of those groups. Save your answer to the variable clicks_by_source.

Try using the following code:

clicks_by_source = ad_clicks\
   .groupby(['utm_source',
             'is_click'])\
   .user_id.count()\
   .reset_index()
5.
Now let’s pivot the data so that the columns are is_click (either True or False), the index is utm_source, and the values are user_id.

Save your results to the variable clicks_pivot.

Try using the following code:

clicks_pivot = clicks_by_source\
   .pivot(index='utm_source',
          columns='is_click',
          values='user_id')\
   .reset_index()
6.
Create a new column in clicks_pivot called percent_clicked which is equal to the percent of users who clicked on the ad from each utm_source.

Was there a difference in click rates for each source?

Try the following code:

clicks_pivot['percent_clicked'] = \
   clicks_pivot[True] / \
   (clicks_pivot[True] + 
    clicks_pivot[False])
clicks_pivot[True] is the number of people who clicked (because is_click was True for those users)

clicks_pivot[False] is the number of people who did not click (because is_click was False for those users)

So, the percent of people who clicked would be (Total Who Clicked) / (Total Who Clicked + Total Who Did Not Click)

Analyzing an A/B Test
7.
The column experimental_group tells us whether the user was shown Ad A or Ad B.

Were approximately the same number of people shown both adds?

We can group by experimental_group and count the number of users.

8.
Using the column is_click that we defined earlier, check to see if a greater percentage of users clicked on Ad A or Ad B.

Group by both experimental_group and is_click and count the number of user_id‘s.

You might want to use a pivot table like we did for the utm_source exercises.

9.
The Product Manager for the A/B test thinks that the clicks might have changed by day of the week.

Start by creating two DataFrames: a_clicks and b_clicks, which contain only the results for A group and B group, respectively.

To create a_clicks:

a_clicks = ad_clicks[
   ad_clicks.experimental_group
   == 'A']
10.
For each group (a_clicks and b_clicks), calculate the percent of users who clicked on the ad by day.

First, group by is_click and day. Next, pivot the data so that the columns are based on is_click. Finally, calculate the percent of people who clicked on the ad.

11.
Compare the results for A and B. What happened over the course of the week?

Do you recommend that your company use Ad A or Ad B?

#-----------------------------------------------------

import codecademylib
import pandas as pd

ad_clicks = pd.read_csv('ad_clicks.csv')
print (ad_clicks.head())

utm_source_count = ad_clicks.groupby('utm_source').user_id.count()
#print(utm_source_count).reset_index()

ad_clicks['is_click'] = ~ad_clicks.ad_click_timestamp.isnull()
#print (ad_clicks)

clicks_by_source = ad_clicks.groupby(['utm_source', 'is_click']).user_id.count().reset_index()
#print (clicks_by_source)

clicks_pivot = clicks_by_source.pivot(
	columns= 'is_click',
	index = 'utm_source',
	values = 'user_id')
#print (clicks_pivot)

clicks_pivot['percent_clicked'] = 100 * clicks_pivot[True] / (clicks_pivot[True] + clicks_pivot[False])
#print (clicks_pivot)

ab_total = ad_clicks.groupby('experimental_group').user_id.count()
#print (ab_total)

ab_groups = ad_clicks.groupby(['experimental_group', 'is_click'])['user_id'].count().reset_index()
print(ab_groups)

ab_groups_pivot = ab_groups.pivot(
	columns = 'is_click',
	index = 'experimental_group',
	values = 'user_id')

ab_groups_pivot['ab_percentage'] = ab_groups_pivot[True] / (ab_groups_pivot[True] + ab_groups_pivot[False])

print ("\n")
print (ab_groups_pivot)

a_clicks = ad_clicks[ad_clicks.experimental_group == "A"]
#print (a_clicks)

b_clicks = ad_clicks[ad_clicks.experimental_group =='B']
#print (b_clicks)

a_clicks_by_day = a_clicks.groupby(['day', 'is_click']).user_id.count().reset_index()

a_clicks_by_day_pivot = a_clicks_by_day.pivot(
  columns = 'is_click',
  index = 'day',
	values = 'user_id')

a_clicks_by_day_pivot['percent'] = a_clicks_by_day_pivot[True] / (a_clicks_by_day_pivot[True] + a_clicks_by_day_pivot[False])

print (a_clicks_by_day_pivot)

b_clicks_by_day = b_clicks.groupby(['day', 'is_click']).user_id.count().reset_index()

b_clicks_by_day_pivot = b_clicks_by_day.pivot (
	columns = 'is_click',
	index = 'day',
	values = 'user_id')

b_clicks_by_day_pivot['precent'] = b_clicks_by_day_pivot[True] / (b_clicks_by_day_pivot[True] + b_clicks_by_day_pivot[False])

print (b_clicks_by_day_pivot)

#-------------------------------------------------------------

import codecademylib
import pandas as pd

sales = pd.read_csv('sales.csv')
print(sales)
targets = pd.read_csv('targets.csv')
print(targets)

sales_vs_targets = pd.merge(sales, targets)

print (sales_vs_targets)

crushing_it = sales_vs_targets[sales_vs_targets.revenue > sales_vs_targets.target]

#-------------------------------------------------------------

import codecademylib
import pandas as pd

sales = pd.read_csv('sales.csv')
print(sales)
targets = pd.read_csv('targets.csv')
print(targets)
men_women = pd.read_csv('men_women_sales.csv')
print (men_women)

all_data = sales.merge(targets).merge(men_women)
print(all_data)

results = all_data[(all_data.revenue > all_data.target)&(all_data.women > all_data.men)]

#-------------------------------------------------------------

WORKING WITH MULTIPLE DATAFRAMES
Merge on Specific Columns
In the previous example, the merge function “knew” how to combine tables based on the columns that were the same between two tables. For instance, products and orders both had a column called product_id. This won’t always be true when we want to perform a merge.

Generally, the products and customers DataFrames would not have the columns product_id or customer_id. Instead, they would both be called id and it would be implied that the id was the product_id for the products table and customer_id for the customers table. They would look like this:

Customers
id	customer_name	address	phone_number
1	John Smith	123 Main St.	212-123-4567
2	Jane Doe	456 Park Ave.	949-867-5309
3	Joe Schmo	798 Broadway	112-358-1321
Products
id	description	price
1	thing-a-ma-jig	5
2	whatcha-ma-call-it	10
3	doo-hickey	7
4	gizmo	3

**How would this affect our merges?**
Because the id columns would mean something different in each table, our default merges would be wrong.

One way that we could address this problem is to use .rename to rename the columns for our merges. In the example below, we will rename the column id to customer_id, so that orders and customers have a common column for the merge.

pd.merge(
    orders,
    customers.rename(columns={'id': 'customer_id'}))
	
#----------------------------------------------------------------------------

import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
print(orders)
products = pd.read_csv('products.csv')
print(products)

orders_products = pd.merge(orders, products.rename(columns = {'id':'product_id'}))

print (orders_products)
#------------------------------------------------------------------

WORKING WITH MULTIPLE DATAFRAMES
Merge on Specific Columns II
In the previous exercise, we learned how to use rename to merge two DataFrames whose columns don’t match.

If we don’t want to do that, we have another option. We could use the keywords left_on and right_on to specify which columns we want to perform the merge on. In the example below, the “left” table is the one that comes first (orders), and the “right” table is the one that comes second (customers). This syntax says that we should match the customer_id from orders to the id in customers.

pd.merge(
    orders,
    customers,
    left_on='customer_id',
    right_on='id')
If we use this syntax, we’ll end up with two columns called id, one from the first table and one from the second. Pandas won’t let you have two columns with the same name, so it will change them to id_x and id_y.

It will look like this:

id_x	customer_id	product_id	quantity	timestamp	id_y	customer_name	address	phone_number
1	2	3	1	2017-01-01 00:00:00	2	Jane Doe	456 Park Ave	949-867-5309
2	2	2	3	2017-01-01 00:00:00	2	Jane Doe	456 Park Ave	949-867-5309
3	3	1	1	2017-01-01 00:00:00	3	Joe Schmo	789 Broadway	112-358-1321
4	3	2	2	2016-02-01 00:00:00	3	Joe Schmo	789 Broadway	112-358-1321
5	3	3	3	2017-02-01 00:00:00	3	Joe Schmo	789 Broadway	112-358-1321
6	1	4	2	2017-03-01 00:00:00	1	John Smith	123 Main St.	212-123-4567
7	1	1	1	2017-02-02 00:00:00	1	John Smith	123 Main St.	212-123-4567
8	1	4	1	2017-02-02 00:00:00	1	John Smith	123 Main St.	212-123-4567
The new column names id_x and id_y aren’t very helpful for us when we read the table. We can help make them more useful by using the keyword suffixes. We can provide a list of suffixes to use instead of “_x” and “_y”.

For example, we could use the following code to make the suffixes reflect the table names:

pd.merge(
    orders,
    customers,
    left_on='customer_id',
    right_on='id',
    suffixes=['_order', '_customer']
)
The resulting table would look like this:

id_order	customer_id	product_id	quantity	timestamp	id_customer	customer_name	address	phone_number
1	2	3	1	2017-01-01 00:00:00	2	Jane Doe	456 Park Ave	949-867-5309
2	2	2	3	2017-01-01 00:00:00	2	Jane Doe	456 Park Ave	949-867-5309
3	3	1	1	2017-01-01 00:00:00	3	Joe Schmo	789 Broadway	112-358-1321
4	3	2	2	2016-02-01 00:00:00	3	Joe Schmo	789 Broadway	112-358-1321
5	3	3	3	2017-02-01 00:00:00	3	Joe Schmo	789 Broadway	112-358-1321
6	1	4	2	2017-03-01 00:00:00	1	John Smith	123 Main St.	212-123-4567
7	1	1	1	2017-02-02 00:00:00	1	John Smith	123 Main St.	212-123-4567
8	1	4	1	2017-02-02 00:00:00	1	John Smith	123 Main St.	212-123-4567

#-----------------------------------------------------------
import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
print(orders)
products = pd.read_csv('products.csv')
print(products)

orders_products = pd.merge(
  orders,
  products,
	left_on = 'product_id',
	right_on = 'id',
	suffixes = ['_orders', '_products'])

print (orders_products)

#------------------------------------------

WORKING WITH MULTIPLE DATAFRAMES
Outer Merge
In the previous exercise, we saw that when we merge two DataFrames whose rows don’t match perfectly, we lose the unmatched rows.

This type of merge (where we only include matching rows) is called an inner merge. There are other types of merges that we can use when we want to keep information from the unmatched rows.

Suppose that two companies, Company A and Company B have just merged. They each have a list of customers, but they keep slightly different data. Company A has each customer’s name and email. Company B has each customer’s name and phone number. They have some customers in common, but some are different.

company_a

name	email
Sally Sparrow	sally.sparrow@gmail.com
Peter Grant	pgrant@yahoo.com
Leslie May	leslie_may@gmail.com
company_b

name	phone
Peter Grant	212-345-6789
Leslie May	626-987-6543
Aaron Burr	303-456-7891
If we wanted to combine the data from both companies without losing the customers who are missing from one of the tables, we could use an Outer Join. An Outer Join would include all rows from both tables, even if they don’t match. Any missing values are filled in with None or nan (which stands for “Not a Number”).

pd.merge(company_a, company_b, how='outer')
The resulting table would look like this:



#-----------------------------------------------------

WORKING WITH MULTIPLE DATAFRAMES
Left and Right Merge
Let’s return to the merge of Company A and Company B.

Left Merge
Suppose we want to identify which customers are missing phone information. We would want a list of all customers who have email, but don’t have phone.

We could get this by performing a Left Merge. A Left Merge includes all rows from the first (left) table, but only rows from the second (right) table that match the first table.

For this command, the order of the arguments matters. If the first DataFrame is company_a and we do a left join, we’ll only end up with rows that appear in company_a.

By listing company_a first, we get all customers from Company A, and only customers from Company B who are also customers of Company A.

pd.merge(company_a, company_b, how='left')
The result would look like this:

name	email	phone
Sally Sparrow	sally.sparrow@gmail.com	None
Peter Grant	pgrant@yahoo.com	212-345-6789
Leslie May	leslie_may@gmail.com	626-987-6543
Now let’s say we want a list of all customers who have phone but no email. We can do this by performing a Right Merge.

Right Merge
Right merge is the exact opposite of left merge. Here, the merged table will include all rows from the second (right) table, but only rows from the first (left) table that match the second table.

By listing company_a first and company_b second, we get all customers from Company B, and only customers from Company A who are also customers of Company B.

pd.merge(company_a, company_b, how="right")
The result would look like this:

name	email	phone
Peter Grant	pgrant@yahoo.com	212-345-6789
Leslie May	leslie_may@gmail.com	626-987-6543
Aaron Burr	None	303-456-7891

#--------------------------------------------------------------------
WORKING WITH MULTIPLE DATAFRAMES
Concatenate DataFrames
Sometimes, a dataset is broken into multiple tables. For instance, data is often split into multiple CSV files so that each download is smaller.

When we need to reconstruct a single DataFrame from multiple smaller DataFrames, we can use the method pd.concat([df1, df2, df2, ...]). This method only works if all of the columns are the same in all of the DataFrames.

For instance, suppose that we have two DataFrames:

df1
name	email
Katja Obinger	k.obinger@gmail.com
Alison Hendrix	alisonH@yahoo.com
Cosima Niehaus	cosi.niehaus@gmail.com
Rachel Duncan	rachelduncan@hotmail.com
df2
name	email
Jean Gray	jgray@netscape.net
Scott Summers	ssummers@gmail.com
Kitty Pryde	kitkat@gmail.com
Charles Xavier	cxavier@hotmail.com
If we want to combine these two DataFrames, we can use the following command:

pd.concat([df1, df2])
That would result in the following DataFrame:

#--------------------------------------------------------------------

WORKING WITH MULTIPLE DATAFRAMES
Review
This lesson introduced some methods for combining multiple DataFrames:

Creating a DataFrame made by matching the common columns of two DataFrames is called a merge
We can specify which columns should be matches by using the keyword arguments left_on and right_on
We can combine DataFrames whose rows don’t all match using left, right, and outer merges and the how keyword argument
We can stack or concatenate DataFrames with the same columns using pd.concat
Instructions
1.
Cool T-Shirts Inc. just created a website for ordering their products. They want you to analyze two datasets for them:

visits contains information on all visits to their landing page
checkouts contains all users who began to checkout on their website
Use print to inspect each DataFrame.

2.
We want to know the amount of time from a user’s initial visit to the website to when they start to check out.

Use merge to combine visits and checkouts and save it to the variable v_to_c.

3.
In order to calculate the time between visiting and checking out, define a column of v_to_c called time by pasting the following code into script.py:

v_to_c['time'] = v_to_c.checkout_time - \
                 v_to_c.visit_time

print(v_to_c)
4.
To get the average time to checkout, paste the following code into script.py:

print(v_to_c.time.mean())

#-------------------------------------------------------------------------------------
DATA MANIPULATION WITH PANDAS
Page Visits Funnel
Cool T-Shirts Inc. has asked you to analyze data on visits to their website. Your job is to build a funnel, which is a description of how many people continue to the next step of a multi-step process.

In this case, our funnel is going to describe the following process:

A user visits CoolTShirts.com
A user adds a t-shirt to their cart
A user clicks “checkout”
A user actually purchases a t-shirt
If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
11/12Complete
Mark the tasks as complete by checking them off
Funnel for Cool T-Shirts Inc.
1.
Inspect the DataFrames using print and head:

visits lists all of the users who have visited the website
cart lists all of the users who have added a t-shirt to their cart
checkout lists all of the users who have started the checkout
purchase lists all of the users who have purchased a t-shirt
2.
Combine visits and cart using a left merge.

If we want to combine df1 and df2 with a left merge, we use the following code:

pd.merge(df1, df2, how='left')
OR

df1.merge(df2, how='left')
3.
How long is your merged DataFrame?

Use len to find out the number of rows in a DataFrame.

4.
How many of the timestamps are null for the column cart_time?

What do these null rows mean?

You can select null rows from column1 of a DataFrame df using the following code:

df[df.column1.isnull()]
5.
What percent of users who visited Cool T-Shirts Inc. ended up not placing a t-shirt in their cart?

Note: To calculate percentages, it will be helpful to turn either the numerator or the denominator into a float, by using float(), with the number to convert passed in as input. Otherwise, Python will use integer division, which truncates decimal points.

If a row of your merged DataFrame has cart_time equal to null, then that user visited the website, but did not place a t-shirt in their cart.

6.
Repeat the left merge for cart and checkout and count null values. What percentage of users put items in their cart, but did not proceed to checkout?

You can find the percentage of users who put items in their cart but did not proceed to checkout by counting the null values of checkout_time and comparing it to the total number of users who put items in their cart.

7.
Merge all four steps of the funnel, in order, using a series of left merges. Save the results to the variable all_data.

Examine the result using print and head.

Suppose we wanted to merge df1, df2, and df3 using a left merge. We could use the following code:

df1.merge(df2, how='left')\
   .merge(df3, how='left')
8.
What percentage of users proceeded to checkout, but did not purchase a t-shirt?

9.
Which step of the funnel is weakest (i.e., has the highest percentage of users not completing it)?

How might Cool T-Shirts Inc. change their website to fix this problem?

Average Time to Purchase
10.
Using the giant merged DataFrame all_data that you created, let’s calculate the average time from initial visit to final purchase. Start by adding the following column to your DataFrame:

all_data['time_to_purchase'] = \
    all_data.purchase_time - \
    all_data.visit_time
11.
Examine the results using:

print(all_data.time_to_purchase)
12.
Calculate the average time to purchase using the following code:

print(all_data.time_to_purchase.mean())
#-------------------------------------------------------------------------------------

import codecademylib
import pandas as pd

visits = pd.read_csv('visits.csv',
                     parse_dates=[1])
cart = pd.read_csv('cart.csv',
                   parse_dates=[1])
checkout = pd.read_csv('checkout.csv',
                       parse_dates=[1])
purchase = pd.read_csv('purchase.csv',
                       parse_dates=[1])

#print (visits.head())
#print (cart.head())
#print (checkout.head())
#print (purchase.head())

df = pd.merge(visits, cart, how = "left")
df = pd.merge(df, checkout, how = 'left')
df = pd.merge(df, purchase, how = 'left')

user_id_count = df.user_id.count()
cart_count = df.cart_time.count()
checkout_count = df.checkout_time.count()
purchase_count = df.purchase_time.count()

cart_perc = 100 * cart_count / user_id_count
print ("Cart Percentage is " + str(cart_perc) + "%")

checkout_perc = 100 * checkout_count / cart_count
print ("Cart Percentage is " + str(checkout_perc) + "%")

purchase_perc = 100 * purchase_count / checkout_count
print ("Cart Percentage is " + str(purchase_perc) + "%")

df['time_to_purchase'] = df.purchase_time - df.visit_time

print (df.time_to_purchase)

print (df.time_to_purchase.mean())

#print (df[df.cart_time.isnull()].count())

#count_is_null = lambda row: 1 if row.isnull() else 0

#print (df.cart_time.apply(count_is_null).count())

#----------------------------------------------------

import pandas as pd
pd.set_option('display.max_colwidth', -1)

# Loading the data and investigating it
jeopardy_data = pd.read_csv("jeopardy.csv")
#print(jeopardy_data.columns)

# Renaming misformatted columns
jeopardy_data = jeopardy_data.rename(columns = {" Air Date": "Air Date", " Round" : "Round", " Category": "Category", " Value": "Value", " Question":"Question", " Answer": "Answer"})
#print(jeopardy_data.columns)
#print(jeopardy_data["Question"])

# Filtering a dataset by a list of words
def filter_data(data, words):
  # Lowercases all words in the list of words as well as the questions. Returns true is all of the words in the list appear in the question.
  filter = lambda x: all(word.lower() in x.lower() for word in words)
  # Applies the lambda function to the Question column and returns the rows where the function returned True
  return data.loc[data["Question"].apply(filter)]

# Testing the filter function
filtered = filter_data(jeopardy_data, ["King", "England"])
#print(filtered["Question"])

# Adding a new column. If the value of the float column is not "None", then we cut off the first character (which is a dollar sign), and replace all commas with nothing, and then cast that value to a float. If the answer was "None", then we just enter a 0.
jeopardy_data["Float Value"] = jeopardy_data["Value"].apply(lambda x: float(x[1:].replace(',','')) if x != "None" else 0)

# Filtering the dataset and finding the average value of those questions
filtered = filter_data(jeopardy_data, ["King"])
print(filtered["Float Value"].mean())

# A function to find the unique answers of a set of data
def get_answer_counts(data):
    return data["Answer"].value_counts()

# Testing the answer count function
print(get_answer_counts(filtered))


#------------------------ import form hand note -------------------------

df = pd.read_csv('source.csv')
df = pd.to_csv('target.csv')

df.head(10) #print 10 rows
df.head() # print 5 rows

df.tail()

df.info() # basic dataframe info

df = pd.DataFrame({
		'name' : ['John', 'Jame', 'Joe'],
		'address' : ['123 Main St', '456 Maple Ave', '789 Broadway'],
		'age' : [34, 28, 51]
		})

df = pd.DataFrame([
		['John', 'Jame', 'Joe'],
		['123 Main St', '456 Maple Ave', '789 Broadway'],
		[34, 28, 51],
		],
		columns = ['name', 'address', 'column'])
		
#use dictionary to create dataframe
data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}	
df = pd.DataFrame.from_dict(data)

#   col_1 col_2
#0      3     a
#1      2     b
#2      1     c
#3      0     d

data = {'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}
pd.DataFrame.from_dict(data, orient='index')
#       0  1  2  3
#row_1  3  2  1  0
#row_2  a  b  c  d

# select certain column

df.age 		# method 1
df['age']	# method 2

df['name', 'age']

# select certain row

df.iloc[2]
df.iloc[2:]
df.iloc[2:3]
df.iloc[:4]
df.iloc[-2]
df.iloc[[1,3,5]]

#select with logic

df[df.name = 'xxx']
df[df.age > 30]
df[(df.age < 30) & (df.name = 'Jane')] # | for "or"
df[df.name Isin (['James', 'Jane', 'Jack'])

#reset_index 

df.reset_index(drop = True , inplace = True)
#drop = True   delete old index

#add new column
df['new_column'] = ['a', 'b', 'c']
df['is_taxed'] = True # add whole new column with same value
df['proift'] = df['price'] - df['cost'] # or df.price - df.cost
df['l_name'] = df.name.apply(lower)
df['last_name'] = df.name.apply(lambda x: x.split(' ')[-1])

df['price with tax'] = df.apply(lambda row: \n
						row['price']*1.075
						if row['is taxed'] == 'Yes'
						else row['price'],
						axis = 1)

#column raname

df.rename(column = {'name' : 'new_name',
					'namei' : 'new_namei',
					'name3' : 'new_name3'},
					inplace = True)
					
df = pd.DataFrame([
		['John', 'Jame', 'Joe'],
		['123 Main St', '456 Maple Ave', '789 Broadway'],
		[34, 28, 51],
		],
		columns = ['name', 'address', 'column'])
df.columns = ['First Name', 'Age', 'Address'] #batch change column name

#apply command to column

df.column_name.command()

mean max unique
std min unique
median count

print(shipments.state)
['CA', 'CA', 'NY', 'NY', 'NJ', 'NJ']

print(shipments.state.unique())
['CA', 'NY', 'NJ']

print(shipments.state.unique().count())
3

user_id_count = df.user_id.count()
print(df.time_to_purchase.mean())

#groupby

df.groupby('column1').column2.measurment()

grades = df.groupby('students').grade.mean()

tea_counts = teas.groupby('category').id.count().reset_index()
tea_counts = tea_counts.rename(columns = {'id' : 'counts'})

high_earners = df.groupby('category').wage.apply(lambda x: np.percentile(x, 75)).reset_index()
df.groupby(['location', 'Day of week'])['Total cales'].mean().reset_index()

#pivot table
df.pivot(columns = 'column pivot',
		inex = 'column to be row',
		values = 'column to be values')
		
#merge df
new_df = pd.merge(df1, df2)
new_df = df.merge(df1).merge(df3)

#concate
menu = pd.concate([df1], [df2])

#merge left / right
how = left
#only left df item will be rept

#merge inner / outter
df_new = pd.merge(df1, df2, how = 'outer')
#merge all lines without losing data 'nan' and 'None' will be filled

#merge and change column name
pd.merge(
		orders,
		customers,
		left_on = 'customer_id',
		right_on = 'id',
		suffixes = ['_order', '_customer'])
		
#rename at merge to match the df name
pd.merge(orders, customers.rename(columns = {'id' : 'customer'}))




						











#delete column
df = df.drop(columns = 'column name')

#rename column

df.rename(columns = {'old_name' : 'new_name',
					 'old_name1' : 'new_name1'},
					 inplace = True)
					 
					 







