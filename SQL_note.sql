--introduction

/*

What is a Relational Database Management System (RDBMS)?
A relational database management system (RDBMS) is a program that allows you to create, update, and administer a relational database. Most relational database management systems use the SQL language to access the database.

What is SQL?
SQL (Structured Query Language) is a programming language used to communicate with data stored in a relational database management system. SQL syntax is similar to the English language, which makes it relatively easy to write, read, and interpret.

Many RDBMSs use SQL (and variations of SQL) to access the data in tables. For example, SQLite is a relational database management system. SQLite contains a minimal set of SQL commands (which are the same across all RDBMSs). Other RDBMSs may use other variants.

(SQL is often pronounced in one of two ways. You can pronounce it by speaking each letter individually like “S-Q-L”, or pronounce it using the word “sequel”.)

Popular Relational Database Management Systems
SQL syntax may differ slightly depending on which RDBMS you are using. Here is a brief description of popular RDBMSs:

MySQL

MySQL is the most popular open source SQL database. It is typically used for web application development, and often accessed using PHP.

The main advantages of MySQL are that it is easy to use, inexpensive, reliable (has been around since 1995), and has a large community of developers who can help answer questions.

Some of the disadvantages are that it has been known to suffer from poor performance when scaling, open source development has lagged since Oracle has taken control of MySQL, and it does not include some advanced features that developers may be used to.

PostgreSQL

PostgreSQL is an open source SQL database that is not controlled by any corporation. It is typically used for web application development.

PostgreSQL shares many of the same advantages of MySQL. It is easy to use, inexpensive, reliable and has a large community of developers. It also provides some additional features such as foreign key support without requiring complex configuration.

The main disadvantage of PostgreSQL is that it is slower in performance than other databases such as MySQL. It is also less popular than MySQL which makes it harder to come by hosts or service providers that offer managed PostgreSQL instances.

Oracle DB

Oracle Corporation owns Oracle Database, and the code is not open sourced.

Oracle DB is for large applications, particularly in the banking industry. Most of the world’s top banks run Oracle applications because Oracle offers a powerful combination of technology and comprehensive, pre-integrated business applications, including essential functionality built specifically for banks.

The main disadvantage of using Oracle is that it is not free to use like its open source competitors and can be quite expensive.

SQL Server

Microsoft owns SQL Server. Like Oracle DB, the code is close sourced.

Large enterprise applications mostly use SQL Server.

Microsoft offers a free entry-level version called Express but can become very expensive as you scale your application.

SQLite

SQLite is a popular open source SQL database. It can store an entire database in a single file. One of the most significant advantages this provides is that all of the data can be stored locally without having to connect your database to a server.

SQLite is a popular choice for databases in cellphones, PDAs, MP3 players, set-top boxes, and other electronic gadgets. The SQL courses on Codecademy use SQLite.

For more info on SQLite, including installation instructions, read this article.


Conclusion
Relational databases store data in tables. Tables can grow large and have a multitude of columns and records. Relational database management systems (RDBMSs) use SQL (and variants of SQL) to manage the data in these large tables. The RDBMS you use is your choice and depends on the complexity of your application.

*/

--The statements covered in this course use SQLite Relational Database Management System (RDBMS). 

--https://www.codecademy.com/articles/sql-commands

/*
datatype

INTEGER, a positive or negative whole number
TEXT, a text string
DATE, the date formatted as YYYY-MM-DD
REAL, a decimal value
*/

CREATE TABLE table_name (
   column_1 data_type, 
   column_2 data_type, 
   column_3 data_type
);

CREATE TABLE celebs (
   id INTEGER, 
   name TEXT, 
   age INTEGER
);

INSERT INTO celebs (id, name, age) 
VALUES (1, 'Justin Bieber', 22);

INSERT INTO friends (id, name, birthday) 
VALUES (1, 'Jane Doe', '1990-05-30');


SELECT name FROM celebs;

--SELECT statements always return a new table called the result set.

SELECT * FROM celebs;

SELECT column1, column2 
FROM table_name;

SELECT name AS 'Titles'
FROM movies;

--AS is a keyword in SQL that allows you to rename a column or table using an alias.

--Although it’s not always necessary, it’s best practice to surround your aliases with single quotes.

--When using AS, the columns are not being renamed in the table. The aliases only appear in the result.

SELECT DISTINCT tools 
FROM inventory;

--DISTINCT is used to return unique values in the output. It filters out all duplicate values in the specified column(s).

SELECT *
FROM movies
WHERE imdb_rating > 8;

/*
Comparison operators used with the WHERE clause are:

= equal to
!= not equal to
> greater than
< less than
>= greater than or equal to
<= less than or equal to
*/

SELECT *
FROM movies
WHERE year BETWEEN 1990 AND 1999;

--For example, this statement filters the result set to only include movies with years from 1990 up to, and including 1999.

SELECT *
FROM movies
WHERE name BETWEEN 'A' AND 'J';

/*
In this statement, BETWEEN filters the result set to only include movies with names that begin with the letter ‘A’ up to, but not including ones that begin with ‘J’.

However, if a movie has a name of simply ‘J’, it would actually match. This is because BETWEEN goes up to the second value — up to ‘J’. So the movie named ‘J’ would be included in the result set but not ‘Jaws’.

*/

SELECT * 
FROM movies
WHERE name LIKE 'Se_en';

--The _ means you can substitute any individual character here without breaking the pattern. The names Seven and Se7en both match this pattern.

SELECT * 
FROM movies
WHERE name LIKE 'A%';

/*
% is a wildcard character that matches zero or more missing letters in the pattern. For example:

A% matches all movies with names that begin with letter ‘A’
%a matches all movies that end with ‘a’
*/

SELECT name
FROM movies 
WHERE imdb_rating IS NOT NULL;

/*
It is not possible to test for NULL values with comparison operators, such as = and !=.

Instead, we will have to use these operators:

IS NULL
IS NOT NULL
*/

SELECT * 
FROM movies
WHERE year BETWEEN 1990 AND 1999
   AND genre = 'romance';
   
SELECT *
FROM movies
WHERE year > 2014
   OR genre = 'action';
   
   
   
   
   SELECT *
FROM movies
ORDER BY name;

SELECT *
FROM movies
WHERE imdb_rating > 8
ORDER BY year DESC;

/*
DESC is a keyword used in ORDER BY to sort the results in descending order (high to low or Z-A).

ASC is a keyword used in ORDER BY to sort the results in ascending order (low to high or A-Z).

Note: ORDER BY always goes after WHERE (if WHERE is present).
*/

SELECT *
FROM movies
LIMIT 10;

/*
LIMIT is a clause that lets you specify the maximum number of rows the result set will have. This saves space on our screen and makes our queries run faster.

Here, we specify that the result set can’t have more than 10 rows.

LIMIT always goes at the very end of the query. Also, it is not supported in all SQL databases.
*/

SELECT name,
 CASE
  WHEN imdb_rating > 8 THEN 'Fantastic'
  WHEN imdb_rating > 6 THEN 'Poorly Received'
  ELSE 'Avoid at All Costs'
 END AS 'Review'
FROM movies;

/*Each WHEN tests a condition and the following THEN gives us the string if the condition is true.
The ELSE gives us the string if all the above conditions are false.
The CASE statement must end with END.

In the result, you have to scroll right because the column name is very long. To shorten it, we can rename the column to ‘Review’ using AS:*/

-------------------------some samples----------------------------------

SELECT name, gender, number
FROM babies
WHERE year = 1880
ORDER BY number DESC
LIMIT 10;


SELECT *
FROM news
WHERE publisher LIKE "%Wall%" AND category = "t"
ORDER BY id DESC;

SELECT population, year
FROM population_years
WHERE country = "Gabon"
ORDER BY population DESC
LIMIT 5;

SELECT population, country
FROM population_years
WHERE year = 2005
ORDER BY population 
LIMIT 10;

SELECT DISTINCT country, population
FROM population_years
WHERE year = 2010 AND population > 100
ORDER BY population DESC;

SELECT DISTINCT country
FROM population_years
WHERE country LIKE "%Islands%";

SELECT year, population
FROM population_years
WHERE country = "Indonesia"
ORDER BY year;

SELECT COUNT(*)
FROM users
WHERE email LIKE '%.com'; -- count email end with .com

SELECT first_name, COUNT(*) AS 'count'
FROM users
GROUP BY first_name
ORDER BY 2 DESC;          -- count first name

SELECT ROUND(watch_duration_in_minutes) AS 'duration', 
    COUNT(*)
FROM watch_history
GROUP BY 1
ORDER BY 1 ASC;           --round duration and sort with count


SELECT user_id, SUM(amount)
FROM payments
WHERE status = 'paid'
GROUP BY user_id
ORDER BY SUM(amount) DESC;
--you can also rename the SUM(amount) column:

SELECT user_id, SUM(amount) AS 'total'
FROM payments
WHERE status = 'paid'
GROUP BY user_id
ORDER BY total DESC;
--And add column reference numbers:

SELECT user_id, SUM(amount) AS 'total'
FROM payments
WHERE status = 'paid'
GROUP BY 1
ORDER BY 2 DESC;


SELECT user_id, SUM(watch_duration_in_minutes) AS 'Total_time'
FROM watch_history
GROUP BY user_id
HAVING Total_time > 400  --HAVING IS FOLLOWING GROUP BY FOR ADITIONAL FILTER
ORDER BY 2 DESC;

SELECT ROUND(SUM(watch_duration_in_minutes)) AS 'Total min'
FROM watch_history;

SELECT pay_date, SUM(amount) AS 'Total_amount'
FROM payments
WHERE status = 'paid'
GROUP BY 1
ORDER BY Total_amount DESC;

SELECT AVG(amount)
FROM payments
WHERE status = 'paid';


SELECT MAX(watch_duration_in_minutes) AS 'MAX',
    MIN(watch_duration_in_minutes) AS 'MIN'
FROM watch_history;


SELECT CASE
   WHEN url LIKE '%github.com%' THEN 'GitHub'
   WHEN url LIKE '%medium.com%' THEN 'Medium'
   WHEN url LIKE '%nytimes.com%' THEN 'New York Times'
   --WHEN url LIKE '%github.com%' THEN 'GitHub'
   ELSE 'Other'

  END AS 'Source',
  COUNT(*)
FROM hacker_news
GROUP BY Source;


SELECT CASE
   WHEN medium LIKE '%gold%'   THEN 'Gold'
   WHEN medium LIKE '%silver%' THEN 'Silver'
   --ELSE NULL
  END AS 'Bling',
  COUNT(*)
FROM met
WHERE Bling IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC;

------------------------- END OF SAMPLES ------------------------------------------





ALTER TABLE celebs 
ADD COLUMN twitter_handle TEXT;

--The ALTER TABLE statement adds a new column to a table. 

UPDATE celebs 
SET twitter_handle = '@taylorswift13' 
WHERE id = 4; 

DELETE FROM celebs 
WHERE twitter_handle IS NULL;

CREATE TABLE celebs (
   id INTEGER PRIMARY KEY, 
   name TEXT UNIQUE,
   date_of_birth TEXT NOT NULL,
   date_of_death TEXT DEFAULT 'Not Applicable'
);

/*
PRIMARY KEY columns can be used to uniquely identify the row. Attempts to insert a row with an identical value to a row already in the table will result in a constraint violation which will not allow you to insert the new row.

UNIQUE columns have a different value for every row. This is similar to PRIMARY KEY except a table can have many different UNIQUE columns.

NOT NULL columns must have a value. Attempts to insert a row without a value for a NOT NULL column will result in a constraint violation and the new row will not be inserted.
*/


COUNT(): count the number of rows
SUM(): the sum of the values in a column
MAX()/MIN(): the largest/smallest value
AVG(): the average of the values in a column
ROUND(): round the values in the column

SELECT COUNT(*) 
FROM fake_apps; --count all the how many rows

SELECT COUNT(*)
FROM  fake_apps
WHERE price = 0; -- count how many free apps

SELECT SUM(downloads)
FROM fake_apps; -- sum all the downloads

SELECT MIN(downloads)
FROM fake_apps;

SELECT AVG(downloads)
FROM fake_apps;

SELECT name, ROUND(price, 0) -- return price column with 0 decimal
FROM fake_apps;

SELECT ROUND(AVG(price), 2) --Get average price and round it to 2 decimal
FROM fake_apps;

--###############
SELECT AVG(imdb_rating)
FROM movies
WHERE year = 1999;

SELECT AVG(imdb_rating)
FROM movies
WHERE year = 2000;

SELECT AVG(imdb_rating)
FROM movies
WHERE year = 2001;

--instead of above code, we can use below code 
SELECT year,
   AVG(imdb_rating)
FROM movies
GROUP BY year
ORDER BY year;

SELECT price, COUNT(*)
FROM fake_apps
GROUP BY price;

/*
return
price	COUNT(*)
0.0	73
0.99	43
1.99	42
2.99	21
3.99	9
14.99	12
*/

SELECT price, COUNT(*)
FROM fake_apps
WHERE downloads > 20000
GROUP BY price;

/*
return
price	COUNT(*)
0.0	26
0.99	17
1.99	18
2.99	7
3.99	5
14.99	5
*/

SELECT category, SUM(downloads)
FROM fake_apps
GROUP BY category;

/*
category	SUM(downloads)
Books	160864
Business	178726
Catalogs	186158
Education	184724
Entertainment	95168
Finance	178163
Food & Drink	90950
Games	256083
Health & Fitness	165555
Lifestyle	166832
*/

SELECT category, COUNT(downloads)  --Same result as count *
FROM fake_apps
GROUP BY category;

/*
category	COUNT(downloads)
Books	8
Business	10
Catalogs	9
Education	13
Entertainment	8
Finance	9
Food & Drink	5
Games	17
Health & Fitness	9
Lifestyle	10
*/

----------------------

SELECT ROUND(imdb_rating),
   COUNT(name)
FROM movies
GROUP BY ROUND(imdb_rating)
ORDER BY ROUND(imdb_rating);

--code is same above
SELECT ROUND(imdb_rating),
   COUNT(name)
FROM movies
GROUP BY 1
ORDER BY 1;

------------------

SELECT category, 
   price,
   AVG(downloads)
FROM fake_apps
GROUP BY category, price;

SELECT category, 
   price,
   AVG(downloads)
FROM fake_apps
GROUP BY 1, 2;

-------------

SELECT year,
   genre,
   COUNT(name)
FROM movies
GROUP BY 1, 2
HAVING COUNT(name) > 10;

/*
You just learned how to use aggregate functions to perform calculations on your data. What can we generalize so far?

COUNT(): count the number of rows
SUM(): the sum of the values in a column
MAX()/MIN(): the largest/smallest value
AVG(): the average of the values in a column
ROUND(): round the values in the column
Aggregate functions combine multiple rows together to form a single value of more meaningful information.

GROUP BY is a clause used with aggregate functions to combine data from one or more columns.
HAVING limit the results of a query based on an aggregate property.
*/

/*
Column References

The GROUP BY and ORDER BY clauses can reference the selected columns by number in which they appear in the SELECT statement. The example query will count the number of movies per rating, and will:

GROUP BY column 2 (rating)
ORDER BY column 1 (total_movies)
*/


SELECT COUNT(*) AS 'total_movies', 
   rating 
FROM movies 
GROUP BY 2 
ORDER BY 1;

/*
SUM() Aggregate Function

The SUM() aggregate function takes the name of a column as an argument and returns the sum of all the value in that column.
*/


SELECT SUM(salary)
FROM salary_disbursement;

/*
MAX()
Aggregate Function

The MAX() aggregate function in SQL takes the name of a column as an argument and returns the largest value in a column. The given query will return the largest value from the amount column.
*/


SELECT MAX(amount) 
FROM transactions;

/*
COUNT() Aggregate Function

The COUNT() aggregate function in SQL returns the total number of rows that match the specified criteria. For instance, to find the total number of employees who have less than 5 years of experience, the given query can be used.

Note: A column name of the table can also be used instead of *. Unlike COUNT(*), this variation COUNT(column) will not count NULL values in that column.

*/

SELECT COUNT(*)
FROM employees
WHERE experience < 5;


/* What’s the difference between COUNT(1), COUNT(*), and COUNT(column_name)?

It’s important to note that depending on the ‘flavor’ of SQL you are using (MySQL, SQLite, SQL Server, etc.), there may be very slight differences in performance between COUNT(1) and COUNT(*), but generally speaking COUNT(1) and COUNT(*) will both return the number of rows that match the condition specified in your query.

As for COUNT(column_name), this statement will return the number of rows that have a non-null value for the specified column.

Let’s say we have the following table called people:

name color
Austin	Blue
Jenna Green
Reese <NULL>

When we run either of these queries: */

SELECT COUNT(1) FROM people;
SELECT COUNT(*) FROM people;

/* we’re going to get a result of 3 because there are three rows in the table. But If we run this query:*/

SELECT COUNT(favorite_color) FROM people;

/*we will get a result of 2 because the third row contains a value of NULL for favorite_color, therefore that row does not get counted. */


/*
GROUP BY Clause

The GROUP BY clause will group records in a result set by identical values in one or more columns. It is often used in combination with aggregate functions to query information of similar records. The GROUP BY clause can come after FROM or WHERE but must come before any ORDER BY or LIMIT clause.

The given query will count the number of movies per rating.
*/

SELECT rating, 
   COUNT(*) 
FROM movies 
GROUP BY rating;

/*

MIN() Aggregate Function

The MIN() aggregate function in SQL returns the smallest value in a column. For instance, to find the smallest value of the amount column from the table named transactions, the given query can be used.

*/

SELECT MIN(amount) 
FROM transactions;

/*
AVG() Aggregate Function

The AVG() aggregate function returns the average value in a column. For instance, to find the average salary for the employees who have less than 5 years of experience, the given query can be used.

*/

SELECT AVG(salary)
FROM employees
WHERE experience < 5;

/*
HAVING Clause

The HAVING clause is used to further filter the result set groups provided by the GROUP BY clause. HAVING is often used with aggregate functions to filter the result set groups based on an aggregate property. The given query will select only the records (rows) from only years where more than 5 movies were released per year.


*/

SELECT year, 
   COUNT(*) 
FROM movies 
GROUP BY year
HAVING COUNT(*) > 5;

/*
ROUND() Function

The ROUND() function will round a number value to a specified number of places. It takes two arguments: a number, and a number of decimal places. It can be combined with other aggregate functions, as shown in the given query. This query will calculate the average rating of movies from 2015, rounding to 2 decimal places.

*/

SELECT year, 
   ROUND(AVG(rating), 2) 
FROM movies 
WHERE year = 2015

/*

order_id	customer_id	subscription_id	purchase_date
1	3	2	01-10-2017
2	2	4	01-9-2017
3	3	4	01-26-2017
4	9	9	01-4-2017
5	7	5	01-25-2017
subscription_id	description	price_per_month	subscription_length
1	Politics Magazine	10	12 months
2	Politics Magazine	11	6 months
3	Politics Magazine	12	3 months
4	Fashion Magazine	15	12 months
5	Fashion Magazine	17	6 months
customer_id	customer_name	address	
1	Allie Rahaim	123 Broadway	
2	Jacquline Diddle	456 Park Ave.	
3	Lizabeth Letsche	789 Main St.	
4	Jessia Butman	1 Columbus Ave.	
5	Inocencia Goyco	12 Amsterdam Ave.	

*/

--If we want to combine orders and customers, we would type:

SELECT *
FROM orders
JOIN customers
ON orders.customer_id = customers.customer_id;

-------------------------------------------------
  
SELECT *
FROM orders
JOIN subscriptions
ON orders.subscription_id = subscriptions.subscription_id
WHERE subscriptions.description = 'Fashion Magazine';
  
-----------------------

SELECT COUNT(*)
FROM newspaper;

SELECT COUNT(*)
FROM online;

SELECT COUNT(*)
FROM newspaper
JOIN online
ON newspaper.id = online.id;

--above IS INNER JOIN. any records not match will be disrgraeded

SELECT *
FROM table1
LEFT JOIN table2
  ON table1.c2 = table2.c2;
  
SELECT *
FROM newspaper
LEFT JOIN online
  ON newspaper.id = online.id
WHERE online.id IS NULL;

-- above is a left join, left side information will override the right side info if dont match.

/*
------Primary Key vs Foreign Key-----
Let’s return to our example of the magazine subscriptions. Recall that we had three tables: orders, subscriptions, and customers.

Each of these tables has a column that uniquely identifies each row of that table:

order_id for orders
subscription_id for subscriptions
customer_id for customers
These special columns are called primary keys.

Primary keys have a few requirements:

**None of the values can be NULL.
**Each value must be unique (i.e., you can’t have two customers with the same customer_id in the customers table).
**A table can not have more than one primary key column.
**Let’s reexamine the orders table:

order_id	customer_id	subscription_id	purchase_date
1	2	3	2017-01-01
2	2	2	2017-01-01
3	3	1	2017-01-01

Note that customer_id (the primary key for customers) and subscription_id (the primary key for subscriptions) both appear in this.

When the primary key for one table appears in a different table, it is called a foreign key.

So customer_id is a primary key when it appears in customers, but a foreign key when it appears in orders.

In this example, our primary keys all had somewhat descriptive names. Generally, the primary key will just be called id. Foreign keys will have more descriptive names.

Why is this important? The most common types of joins will be joining a foreign key from one table with the primary key from another table. For instance, when we join orders and customers, we join on customer_id, which is a foreign key in orders and the primary key in customers.
*/

--cross join

SELECT shirts.shirt_color,
   pants.pants_color
FROM shirts
CROSS JOIN pants;

--If we have 3 different shirts (white, grey, and olive) and 2 different pants (light denim and black), the results might look like this:

/*

shirt_color	pants_color
white	light denim
white	black
grey	light denim
grey	black
olive	light denim
olive	black

*/

SELECT *
FROM newspaper
CROSS JOIN months
WHERE newspaper.start_month <= months.month AND newspaper.end_month >= months.month;

------

SELECT month,
   COUNT(*)
FROM newspaper
CROSS JOIN months
WHERE start_month <= month 
   AND end_month >= month
GROUP BY month

/* Results
month	COUNT(*)
1	2
2	9
3	13
4	17
5	27
6	30
7	20
8	22
9	21
10	19
11	15
12	10
*/

-- month table only contain number from 1 to 12.

--Union-------------------
/*
table1:

pokemon	type
Bulbasaur	Grass
Charmander	Fire
Squirtle	Water

table2:

pokemon	type
Snorlax	Normal

If we combine these two with UNION:*/

SELECT *
FROM table1
UNION
SELECT *
FROM table2;

/*The result would be:

pokemon	type
Bulbasaur	Grass
Charmander	Fire
Squirtle	Water
Snorlax	Normal

SQL has strict rules for appending data:

Tables must have the same number of columns.
The columns must have the same data types in the same order as the first table.*/

--- WITH -------------------------

WITH previous_results AS (
   SELECT ...
   ...
   ...
   ...
)
SELECT *
FROM previous_results
JOIN customers
  ON _____ = _____;
  
/*The WITH statement allows us to perform a separate query (such as aggregating customer’s subscriptions)
previous_results is the alias that we will use to reference any columns from the query inside of the WITH clause
We can then go on to do whatever we want with this temporary table (such as join the temporary table with another table)*/

WITH previous_query AS (
   SELECT customer_id,
      COUNT(subscription_id) AS 'subscriptions'
   FROM orders
   
   GROUP BY customer_id
)
SELECT customers.customer_name, 
   previous_query.subscriptions
FROM previous_query
JOIN customers
  ON previous_query.customer_id = customers.customer_id;

/*
What is a Funnel?
In the world of marketing analysis, “funnel” is a word you will hear time and time again.

A funnel is a marketing model which illustrates the theoretical customer journey towards the purchase of a product or service. Oftentimes, we want to track how many users complete a series of steps and know which steps have the most number of users giving up.

Some examples include:

Answering each part of a 5 question survey on customer satisfaction
Clicking “Continue” on each step of a set of 5 onboarding modals
Browsing a selection of products → Viewing a shopping cart → Making a purchase
Generally, we want to know the total number of users in each step of the funnel, as well as the percent of users who complete each step.*/  


/* Count the number of distinct user_id who answered each question_text.

You can do this by using a simple GROUP BY command.

What is the number of responses for each question? */

  
SELECT DISTINCT question_text,
      COUNT(DISTINCT user_id)
FROM survey_responses
GROUP BY question_text;

/*
Survey Result
We could use SQL to calculate the percent change between each question, but it’s just as easy to analyze these manually with a calculator or in a spreadsheet program like Microsoft Excel or Google Sheets.

If we divide the number of people completing each step by the number of people completing the previous step:

Question	Percent Completed this Question
1	100%
2	95%
3	82%
4	95%
5	74%

We see that Questions 2 and 4 have high completion rates, but Questions 3 and 5 have lower rates.

This suggests that age and household income are more sensitive questions that people might be reluctant to answer!*/

/* Compare Funnels For A/B Tests
Mattresses and More has an onboarding workflow for new users of their website. It uses modal pop-ups to welcome users and show them important features of the site like:

Welcome to Mattresses and More!
Browse our bedding selection
Select items to add to your cart
View your cart by clicking on the icon
Press ‘Buy Now!’ when you’re ready to checkout
The Product team at Mattresses and More has created a new design for the pop-ups that they believe will lead more users to complete the workflow.

They’ve set up an A/B test where:

50% of users view the original control version of the pop-ups
50% of users view the new variant version of the pop-ups
Eventually, we’ll want to answer the question:

How is the funnel different between the two groups? */


--this 2 code is doing same thing
/*SELECT modal_text,
  COUNT(DISTINCT CASE
      WHEN ab_group = 'control' THEN user_id
      END) AS 'control_clicks'
FROM onboarding_modals
GROUP BY 1
ORDER BY 1 ASC;*/


--this code is doing same thing above
/*SELECT modal_text,
  COUNT(DISTINCT user_id) AS 'control_clicks'
FROM onboarding_modals
WHERE ab_group = 'control'  
GROUP BY 1
ORDER BY 1 ASC;*/


SELECT modal_text,
  COUNT(DISTINCT CASE
      WHEN ab_group = 'control' THEN user_id
      END) AS 'control_clicks',
  COUNT(DISTINCT CASE
      WHEN ab_group = 'variant' THEN user_id
      END) AS 'variant_clicks'
FROM onboarding_modals
GROUP BY 1
ORDER BY 1 ASC;


/* 
Build a Funnel from Multiple Tables 2
First, we want to combine the information from the three tables (browse, checkout, purchase) into one table with the following schema:

browser_date	user_id	is_checkout	is_purchase
2017-12-20	6a7617321513	True	False
2017-12-20	022d871cdcde	False	False
…	…	…	…

Each row will represent a single user:

If the user has any entries in checkout, then is_checkout will be True.
If the user has any entries in purchase, then is_purchase will be True.
If we use an INNER JOIN to create this table, we’ll lose information from any customer who does not have a row in the checkout or purchase table.

Therefore, we’ll need to use a series of LEFT JOIN commands. */

/*
1.
Start by selecting all rows (*) from the LEFT JOIN of:

browse (aliased as b)
checkout (aliased as c)
purchase (aliased as p)
Be sure to use this order to make sure that we get all of the rows.

LIMIT your results to the first 50 so that it loads quickly.*/

--THIS CODE JOIN THE 3 TABLE TOGETHER.
SELECT *
FROM browse AS 'b'
LEFT JOIN checkout AS 'c'
  ON c.user_id = b.user_id
LEFT JOIN purchase AS 'p'
  ON p.user_id = c.user_id



--THIS CODE JOIN THE 3 TABLE TOGETHER. AND CONVERT THE C.USER_ID AND P.USER_ID TO 0 OR 1 ( IS NOT NULL , RETURN 0 / 1)

SELECT DISTINCT b.browse_date, 
    b.user_id, 
    c.user_id  IS NOT NULL AS 'is_checkout', 
    p.user_id  IS NOT NULL AS 'is_purchase'
FROM browse AS 'b'
LEFT JOIN checkout AS 'c'
  ON c.user_id = b.user_id
LEFT JOIN purchase AS 'p'
  ON p.user_id = c.user_id
LIMIT 50;

  
/* browse_date	user_id	is_checkout	is_purchase
2017-12-20	336f9fdc-aaeb-48a1-a773-e3a935442d45	0	0
2017-12-20	4596bb1a-7aa9-4ac9-9896-022d871cdcde	0	0
2017-12-20	2fdb3958-ffc9-4b84-a49d-5f9f40e9469e	1	1
2017-12-20	fc394c75-36f1-4df1-8665-23c32a43591b	0	0  
 */


/* Build a Funnel from Multiple Tables 3
We’ve created a new table that combined all of our data:

browser_date	user_id	is_checkout	is_purchase
2017-12-20	6a7617321513	1	0
2017-12-20	022d871cdcde	0	0
…	…	…	…

Here, 1 represents True and 0 represents False.

Once we have the data in this format, we can analyze it in several ways. */

WITH funnels AS (
  SELECT DISTINCT b.browse_date,
     b.user_id,
     c.user_id IS NOT NULL AS 'is_checkout',
     p.user_id IS NOT NULL AS 'is_purchase'
  FROM browse AS 'b'
  LEFT JOIN checkout AS 'c'
    ON c.user_id = b.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = c.user_id)

SELECT COUNT(*) AS 'num_browse',
  SUM(is_checkout) AS 'num_checkout',
  SUM(is_purchase) AS 'num_purchase',
  1.0 * SUM(is_checkout) / COUNT(user_id) AS 'c_rate',
  1.0 * SUM(is_purchase) / SUM(is_checkout) AS 'p_rate'
FROM funnels;
/* 
The management team suspects that conversion from checkout to purchase changes as the browse_date gets closer to Christmas Day.

We can make a few edits to this code to calculate the funnel for each browse_date using GROUP BY. */

WITH funnels AS (
  SELECT DISTINCT b.browse_date,
     b.user_id,
     c.user_id IS NOT NULL AS 'is_checkout',
     p.user_id IS NOT NULL AS 'is_purchase'
  FROM browse AS 'b'
  LEFT JOIN checkout AS 'c'
    ON c.user_id = b.user_id
  LEFT JOIN purchase AS 'p'
    ON p.user_id = c.user_id)
SELECT browse_date,
   COUNT(*) AS 'num_browse',
   SUM(is_checkout) AS 'num_checkout',
   SUM(is_purchase) AS 'num_purchase',
   1.0 * SUM(is_checkout) / COUNT(user_id) AS 'browse_to_checkout',
   1.0 * SUM(is_purchase) / SUM(is_checkout) AS 'checkout_to_purchase'
FROM funnels
GROUP BY browse_date
ORDER BY browse_date;

/*
CALCULATING CHURN
What is Churn?
A common revenue model for SaaS (Software as a service) companies is to charge a monthly subscription fee for access to their product. Frequently, these companies aim to continually increase the number of users paying for their product. One metric that is helpful for this goal is churn rate.

Churn rate is the percent of subscribers that have canceled within a certain period, usually a month. For a user base to grow, the churn rate must be less than the new subscriber rate for the same period.

To calculate the churn rate, we only will be considering users who are subscribed at the beginning of the month. The churn rate is the number of these users who cancel during the month divided by the total number:

cancellations / total subscribers

Churn
For example, suppose you were analyzing data for a monthly video streaming service called CodeFlix. At the beginning of February, CodeFlix has 1,000 customers. In February, 250 of these customers cancel. The churn rate for February would be:

250 / 1000 = 25% churn rate

CALCULATING CHURN
Single Month I
Now that we’ve gone over what churn is, let’s see how we can calculate it using SQL. In this example, we’ll calculate churn for the month of December 2016.

Typically, there will be data in a subscriptions table available in the following format:

id - the customer id
subscription_start - the subscribe date
subscription_end - the cancel date
When customers have a NULL value for their subscription_end, that’s a good thing. It means they haven’t canceled!

Remember from the previous exercise that churn rate is:

Churn
For the numerator, we only want the portion of the customers who cancelled during December:*/

SELECT COUNT(*)
FROM subscriptions
WHERE subscription_start < '2016-12-01'
  AND (
    subscription_end
    BETWEEN '2016-12-01' AND '2016-12-31'
  );
  

/* For the denominator, we only want to be considering customers who were active at the beginning of December: */

SELECT COUNT(*)
FROM subscriptions
WHERE subscription_start < '2016-12-01'
  AND (
    (subscription_end >= '2016-12-01')
    OR (subscription_end IS NULL)
  );

/*`
You might’ve noticed there are quite a few parentheses in these two queries.

When there are multiple conditions in a WHERE clause using AND and OR, it’s the best practice to always use the parentheses to enforce the order of execution. It reduces confusion and will make the code easier to understand. The condition within the brackets/parenthesis will always be executed first.

Anyways, now that we have the users who canceled during December, and total subscribers, let’s divide the two to get the churn rate.

When dividing, we need to be sure to multiply by 1.0 to cast the result as a float:*/

SELECT 1.0 * 
(
  SELECT COUNT(*)
  FROM subscriptions
  WHERE subscription_start < '2016-12-01'
  AND (
    subscription_end
    BETWEEN '2016-12-01'
    AND '2016-12-31'
  )
) / (
  SELECT COUNT(*) 
  FROM subscriptions 
  WHERE subscription_start < '2016-12-01'
  AND (
    (subscription_end >= '2016-12-01')
    OR (subscription_end IS NULL)
  )
) 
AS result;

/*
Here, we have the numerator divided by the denominator, and then multiplying the answer by 1.0. At the very end, we are renaming the final answer to result using AS. */

/*
Instructions
1.
We’ve imported 4 months of data for a company from when they began selling subscriptions. This company has a minimum commitment of 1 month, so there are no cancellations in the first month.

The subscriptions table contains:

id
subscription_start
subscription_end
Use the methodology provided in the narrative to calculate the churn for January 2017.*/

/*
SELECT DISTINCT subscription_start
FROM subscriptions
ORDER BY subscription_start
LIMIT 200;*/

SELECT 1.0 * 

(

  SELECT COUNT(*)
  FROM subscriptions 
  WHERE subscription_start < '2017-01-01'
  AND ( 
    subscription_end 
    BETWEEN '2017-01-01' 
    AND '2017-01-31'
  )

) / (

  SELECT COUNT(*)
  FROM subscriptions 
  WHERE subscription_start < '2017-01-01'
  AND ( 
    (subscription_end >= '2017-01-01')
    OR 
    (subscription_end IS NULL)
  )
)
AS result;



/* CALCULATING CHURN
Single Month II
The previous method worked, but you may have noticed we selected the same group of customers twice for the same month and repeated a number of conditional statements.

Companies typically look at churn data over a period of many months. We need to modify the calculation a bit to make it easier to mold into a multi-month result. This is done by making use of WITH and CASE.

To start, use WITH to create the group of customers that are active going into December: */

WITH enrollments AS
(SELECT *
FROM subscriptions
WHERE subscription_start < '2016-12-01'
AND (
  (subscription_end >= '2016-12-01')
  OR (subscription_end IS NULL)
)),
/* Let’s create another temporary table that contains an is_canceled status for each of these customers . This will be 1 if they cancel in December and 0 otherwise (their cancellation date is after December or NULL). */

status AS 
(SELECT
CASE
  WHEN (subscription_end > '2016-12-31')
    OR (subscription_end IS NULL) THEN 0
    ELSE 1
  END as is_canceled,
...
/* We could just COUNT() the rows to determine the number of users. However, to support the multiple month calculation, lets add a is_active column to the status temporary table. This uses the same condition we created enrollments with: */

status AS
  ...
  CASE
    WHEN subscription_start < '2016-12-01'
      AND (
        (subscription_end >= '2016-12-01')
        OR (subscription_end IS NULL)
      ) THEN 1
    ELSE 0
  END as is_active
  FROM enrollments
  )
/* This tells us if someone is active at the beginning of the month.

The last step is to do the math on the status table to calculate the month’s churn: */

SELECT 1.0 * SUM(is_canceled) / SUM(is_active)
FROM status;
/* We make sure to multiply by 1.0 to force a float result instead of an integer.
 */

 
 
 
WITH enrollments AS (
  SELECT *
  FROM subscriptions
  WHERE (subscription_start < '2017-01-01')
    AND (
      (subscription_end >= '2017-01-01')
      OR (subscription_end IS NULL)
  )
),

--MY TEST FOR enrollments clause
/*
SELECT *
FROM enrollments
LIMIT 30;*/


status AS (
  SELECT
  CASE
    WHEN (subscription_end > '2017-01-31')
    OR (subscription_end IS NULL) 
      THEN 0
      ELSE 1
  END as is_canceled,

  CASE
    WHEN subscription_start < '2017-01-01'
      AND (
        (subscription_end >= '2017-01-01')
        OR (subscription_end IS NULL)
      ) 
        THEN 1
        ELSE 0
  END as is_active
  FROM enrollments
)

SELECT 1.0 * SUM(is_canceled) / SUM(is_active)
FROM status;

/* CALCULATING CHURN
Multiple Month: Create Months Temporary Table
Our single month calculation is now in a form that we can extend to a multiple month result. But first, we need months!

Some SQL table schemes will contain a prebuilt table of months. Ours doesn’t, so we’ll need to build it using UNION. We’ll need the first and last day of each month.

Our churn calculation uses the first day as a cutoff for subscribers and the last day as a cutoff for cancellations.

This table can be created like: */

SELECT
  '2016-12-01' AS first_day,
  '2016-12-31' AS last_day
UNION
SELECT
  '2017-01-01' AS first_day,
  '2017-01-31' AS last_day;

/* We will be using the months as a temporary table (using WITH) in the churn calculation.

Create the months temporary table using WITH and SELECT everything from it so that you can see the structure.

We need a table for January, February, and March of 2017.   */
  
  
WITH months AS (
  SELECT
    '2017-01-01' AS first_day,
    '2017-01-31' AS last_day
  UNION
  SELECT
    '2017-02-01' AS first_day,
    '2017-02-28' AS last_day
  UNION
  SELECT
    '2017-03-01' AS first_day,
    '2017-03-31' AS last_day;
)

SELECT *
FROM months;

/* CALCULATING CHURN
Multiple Month: Cross Join Months and Users
Now that we have a table of months, we will join it to the subscriptions table. This will result in a table containing every combination of month and subscription.

Ultimately, this table will be used to determine the status of each subscription in each month.

Instructions
1.
The workspace contains the months temporary table from the previous exercise.

Create a cross_join temporary table that is a CROSS JOIN of subscriptions and months.

We’ve added: */

SELECT *
FROM cross_join
LIMIT 100;/* 
at the bottom of this exercise so you can visualize the temporary table you create.

It should SELECT all the columns from the temporary table. */

WITH months AS
(SELECT
  '2017-01-01' as first_day,
  '2017-01-31' as last_day
UNION
SELECT
  '2017-02-01' as first_day,
  '2017-02-28' as last_day
UNION
SELECT
  '2017-03-01' as first_day,
  '2017-03-31' as last_day
),
-- Add temporary cross_join definition here

cross_join AS (
  SELECT *
  FROM subscriptions
  CROSS JOIN months
)

SELECT *
FROM cross_join
LIMIT 100;


/* CALCULATING CHURN
Multiple Month: Determine Active Status
We now have a cross joined table that looks something like:

id	subscription_start	subscription_end	month
1	2016-12-03	2017-02-15	2016-12-01
1	2016-12-03	2017-02-15	2017-01-01
1	2016-12-03	2017-02-15	2017-02-01
1	2016-12-03	2017-02-15	2017-03-01

If you remember our single month example, our ultimate calculation will make use of the status temporary table. The first column of this table was used in the denominator of our churn calculation:

is_active: if the subscription started before the given month and has not been canceled before the start of the given month
For the example above, this column would look like:

month	is_active
2016-12-01	0
2017-01-01	1
2017-02-01	1
2017-03-01	0

Instructions
1.
Add a status temporary table. This table should have the following columns:

id - selected from the cross_join table
month - this is an alias of first_day from the cross_join table. We’re using the first day of the month to represent which month this data is for.
is_active - 0 or 1, derive this column using a CASE WHEN statement
The is_active column should be 1 if the subscription_start is before the month’s first_day and if the subscription_end is either after the month’s first_day or is NULL.

We’ve added:*/

SELECT *
FROM status
LIMIT 100;

--at the bottom of this exercise so you can visualize the temporary table you create. 

WITH months AS
(SELECT
  '2017-01-01' as first_day,
  '2017-01-31' as last_day
UNION
SELECT
  '2017-02-01' as first_day,
  '2017-02-28' as last_day
UNION
SELECT
  '2017-03-01' as first_day,
  '2017-03-31' as last_day
),

cross_join AS
(SELECT *
FROM subscriptions
CROSS JOIN months),

status AS (
  SELECT
  id,
  first_day AS 'month',
  /*CASE
    WHEN (subscription_end > last_day)
    OR (subscription_end IS NULL) 
      THEN 0
      ELSE 1
  END as is_canceled,*/

  CASE
    WHEN subscription_start < first_day
      AND (
        (subscription_end >= first_day)
        OR (subscription_end IS NULL)
      ) 
        THEN 1
        ELSE 0
  END as is_active  
  FROM cross_join
)

SELECT *
FROM status
LIMIT 100;

/* CALCULATING CHURN
Multiple Month: Determine Cancellation Status
For our calculation, we’ll need one more column on the status temporary table: is_canceled

This column will be 1 only during the month that the user cancels.

From the last exercise, the sample user had a subscription_start on 2016-12-03 and their subscription_end was on 2017-02-15. Their complete status table should look like:

month	is_active	is_canceled
2016-12-01	0	0
2017-01-01	1	0
2017-02-01	1	1
2017-03-01	0	0

In our examples, our company has a minimum subscription duration of one month. This means that the subscription_start always falls before the beginning of the month that contains their subscription_end. Outside of our examples, this is not always the case, and you may need to account for customers canceling within the same month that they subscribe.

Instructions
1.
Add an is_canceled column to the status temporary table. Ensure that it is equal to 1 in months containing the subscription_end and 0 otherwise.

Derive this column using a CASE WHEN statement. You can use the BETWEEN function to check if a date falls between two others.

We’ve added: */

SELECT *
FROM status
LIMIT 100;

--at the bottom of this exercise so you can visualize the status table.
  
WITH months AS
(SELECT
  '2017-01-01' as first_day,
  '2017-01-31' as last_day
UNION
SELECT
  '2017-02-01' as first_day,
  '2017-02-28' as last_day
UNION
SELECT
  '2017-03-01' as first_day,
  '2017-03-31' as last_day
),
cross_join AS
(SELECT *
FROM subscriptions
CROSS JOIN months),
status AS
(SELECT id, first_day as month,
CASE
  WHEN (subscription_start < first_day)
    AND (
      subscription_end > first_day
      OR subscription_end IS NULL
    ) THEN 1
  ELSE 0
END as is_active,

CASE
  WHEN (subscription_end >= first_day) 
  AND (subscription_end <= last_day)
    THEN 1
    ELSE 0

  END AS is_canceled
-- add is_canceled here
FROM cross_join)
SELECT *
FROM status
LIMIT 100;

/* CALCULATING CHURN
Multiple Month: Sum Active and Canceled Users
Now that we have an active and canceled status for each subscription for each month, we can aggregate them.

We will GROUP BY month and create a SUM() of the two columns from the status table, is_active and is_canceled.

This provides a list of months, with their corresponding number of active users at the beginning of the month and the number of those users who cancel during the month.

Instructions
1.
Add a status_aggregate temporary table. This table should have the following columns:

month - selected from the status table
active - the SUM() of active users for this month
canceled - the SUM() of canceled users for this month
We’ve added: */

SELECT *
FROM status_aggregate;

--at the bottom of this exercise so you can visualize the temporary table you create.

WITH months AS
(SELECT
  '2017-01-01' as first_day,
  '2017-01-31' as last_day
UNION
SELECT
  '2017-02-01' as first_day,
  '2017-02-28' as last_day
UNION
SELECT
  '2017-03-01' as first_day,
  '2017-03-31' as last_day
),
cross_join AS
(SELECT *
FROM subscriptions
CROSS JOIN months),
status AS
(SELECT id, first_day as month,
CASE
  WHEN (subscription_start < first_day)
    AND (
      subscription_end > first_day
      OR subscription_end IS NULL
    ) THEN 1
  ELSE 0
END as is_active,
CASE 
  WHEN subscription_end BETWEEN first_day AND last_day THEN 1
  ELSE 0
END as is_canceled
FROM cross_join),

status_aggregate AS 
(SELECT month,
SUM(is_active) AS active,
SUM(is_canceled) AS canceled 
FROM status
GROUP BY month
)

SELECT *
FROM status_aggregate;


/* CALCULATING CHURN
Multiple Month: Churn Rate Calculation
Now comes the moment we’ve been waiting for - the actual churn rate.

We use the number of canceled and active subscriptions to calculate churn for each month: churn_rate = canceled / active

Instructions
1.
Add a SELECT statement to calculate the churn rate. The result should contain two columns:

month - selected from status_aggregate
churn_rate - calculated from status_aggregate.canceled and status_aggregate.active.
 */  

WITH months AS
(SELECT
  '2017-01-01' as first_day,
  '2017-01-31' as last_day
UNION
SELECT
  '2017-02-01' as first_day,
  '2017-02-28' as last_day
UNION
SELECT
  '2017-03-01' as first_day,
  '2017-03-31' as last_day
),
cross_join AS
(SELECT *
FROM subscriptions
CROSS JOIN months),
status AS
(SELECT id, first_day as month,
CASE
  WHEN (subscription_start < first_day)
    AND (
      subscription_end > first_day
      OR subscription_end IS NULL
    ) THEN 1
  ELSE 0
END as is_active,
CASE 
  WHEN subscription_end BETWEEN first_day AND last_day THEN 1
  ELSE 0
END as is_canceled
FROM cross_join),
status_aggregate AS
(SELECT
  month,
  SUM(is_active) as active,
  SUM(is_canceled) as canceled
FROM status
GROUP BY month)

SELECT month, 
1.0 * canceled / active AS churn_rate
FROM status_aggregate;

/* CHURN RATES PORJECTS */

/* ANALYZE REAL DATA WITH SQL
Calculating Churn Rates
Four months into launching Codeflix, management asks you to look into subscription churn rates. It’s early on in the business and people are excited to know how the company is doing.

The marketing department is particularly interested in how the churn compares between two segments of users. They provide you with a dataset containing subscription data for users who were acquired through two distinct channels.

The dataset provided to you contains one SQL table, subscriptions. Within the table, there are 4 columns:

id - the subscription id
subscription_start - the start date of the subscription
subscription_end - the end date of the subscription
segment - this identifies which segment the subscription owner belongs to
Codeflix requires a minimum subscription length of 31 days, so a user can never start and end their subscription in the same month.

If you get stuck during this project or would like to see an experienced developer work through it, click “Get Help“ to see a project walkthrough video.

Tasks
9/9Complete
Mark the tasks as complete by checking them off
Get familiar with the data
1.
Take a look at the first 100 rows of data in the subscriptions table. How many different segments do you see?


Hint
Use a SELECT statement and be sure to LIMIT 100.

2.
Determine the range of months of data provided. Which months will you be able to calculate churn for?


Hint
Use MIN and MAX to examine the range of subscription_start.

Calculate churn rate for each segment
3.
You’ll be calculating the churn rate for both segments (87 and 30) over the first 3 months of 2017 (you can’t calculate it for December, since there are no subscription_end values yet). To get started, create a temporary table of months.


Hint
WITH months AS
(SELECT
  '2017-01-01' as first_day,
  '2017-01-31' as last_day
UNION
...
)
4.
Create a temporary table, cross_join, from subscriptions and your months. Be sure to SELECT every column.


Hint
The syntax to join table1 and table2 is:

desired_temp_table AS
(SELECT table1.*, table2.*
FROM table1
CROSS JOIN table2)
5.
Create a temporary table, status, from the cross_join table you created. This table should contain:

id selected from cross_join
month as an alias of first_day
is_active_87 created using a CASE WHEN to find any users from segment 87 who existed prior to the beginning of the month. This is 1 if true and 0 otherwise.
is_active_30 created using a CASE WHEN to find any users from segment 30 who existed prior to the beginning of the month. This is 1 if true and 0 otherwise.

Hint
The is_active from the lesson didn’t account for segments, but might be helpful:

CASE
  WHEN (subscription_start < first_day) 
    AND (
      subscription_end > first_day
      OR subscription_end IS NULL
    ) THEN 1
  ELSE 0
END as is_active
6.
Add an is_canceled_87 and an is_canceled_30 column to the status temporary table. This should be 1 if the subscription is canceled during the month and 0 otherwise.


Hint
You can use BETWEEN to determine if the subscription_end is between the first_day and last_day of the month.

7.
Create a status_aggregate temporary table that is a SUM of the active and canceled subscriptions for each segment, for each month.

The resulting columns should be:

sum_active_87
sum_active_30
sum_canceled_87
sum_canceled_30

Hint
Be sure to GROUP BY month to ensure you get a SUM for each month.

8.
Calculate the churn rates for the two segments over the three month period. Which segment has a lower churn rate?


Hint
Calculate churn by dividing the sum_canceled_ by the sum_active_ for each segment.

Bonus
9.
How would you modify this code to support a large number of segments?


Hint
Avoid hard coding the segment numbers. */

-- START OF THE CODE --


/*
SELECT *
FROM subscriptions
LIMIT 100;

SELECT DISTINCT subscription_start
FROM subscriptions
ORDER BY subscription_start;

SELECT MIN(subscription_start), MAX(subscription_start)
FROM subscriptions
ORDER BY subscription_start;
*/


WITH months AS (
  SELECT 
    '2017-01-01' AS first_day,
    '2017-01-31' AS last_day
  UNION
  SELECT 
    '2017-02-01' AS first_day,
    '2017-02-28' AS last_day
  UNION
  SELECT
    '2017-03-01' AS first_day,
    '2017-03-31' AS last_day 
),

cross_join AS (
  SELECT * 
  FROM subscriptions
  CROSS JOIN monthS
),


status AS (
  SELECT
    id,
    first_day AS month,

    CASE 
      WHEN (segment = 87)
      AND (subscription_end IS NULL 
        OR subscription_end > first_day)
        THEN 1
        ELSE 0
    END AS is_active_87,

    CASE 
      WHEN (segment = 30)
      AND (subscription_end IS NULL 
        OR subscription_end > first_day)
        THEN 1
        ELSE 0
    END AS is_active_30,

    CASE
      WHEN (subscription_end BETWEEN first_day AND last_day)
      AND (segment = 30)
        THEN 1
        ELSE 0
    END AS is_canceled_30,

    CASE
      WHEN (subscription_end BETWEEN first_day AND last_day)
      AND (segment = 87)
        THEN 1
        ELSE 0
    END AS is_canceled_87

  FROM cross_join
),

status_aggregate AS (
  SELECT month,  
  SUM(is_active_87) AS sum_active_87,
  SUM(is_active_30) AS sum_active_30,
  SUM(is_canceled_87) AS sum_canceled_87,
  SUM(is_canceled_30) AS sum_canceled_30,

  1.0 * SUM(is_canceled_87) / SUM(is_active_87) AS c_rate_87,
  1.0 * SUM(is_canceled_30) / SUM(is_active_30) AS c_rate_30

  FROM status
  GROUP BY month
  ORDER BY month
)

SELECT *
FROM status_aggregate
LIMIT 200;


/* FIRST- AND LAST-TOUCH ATTRIBUTION
Introduction
Think of your favorite website: how did you find it? Did you use a search engine? Or click on an ad? Or follow a link in a blog post?

Web developers, marketers, and data analysts use that information to improve their sources (sometimes called channels or touchpoints) online. If an ad campaign drives a lot of visits to their site, then they know that source is working! We say that those visits are attributed to the ad campaign.

But how do websites capture that information? The answer is UTM parameters. These parameters capture when and how a user finds the site. Site owners use special links containing UTM parameters in their ads, blog posts, and other sources. When a user clicks one, a row is added to a database describing their page visit. You can see a common schema for a “page visits” table below and at this link.

user_id - A unique identifier for each visitor to a page
timestamp - The time at which the visitor came to the page
page_name - The title of the section of the page that was visited
utm_source - Identifies which touchpoint sent the traffic (e.g. google, email, or facebook)
utm_medium - Identifies what type of link was used (e.g. cost-per-click or email)
utm_campaign - Identifies the specific ad or email blast (e.g. retargetting-ad or weekly-newsletter)
In this lesson, you will learn how to use SQL, UTM parameters, and touch attribution to draw insights from this data! */

SELECT *
FROM page_visits
LIMIT 10;

/* 
page_name	timestamp	user_id	utm_campaign	utm_source
1 - landing_page	2018-01-24 03:12:16	10006	getting-to-know-cool-tshirts	nytimes
2 - shopping_cart	2018-01-24 04:04:16	10006	getting-to-know-cool-tshirts	nytimes
3 - checkout	2018-01-25 23:10:16	10006	weekly-newsletter	email
1 - landing_page	2018-01-25 20:32:02	10030	ten-crazy-cool-tshirts-facts	buzzfeed
2 - shopping_cart	2018-01-25 23:05:02	10030	ten-crazy-cool-tshirts-facts	buzzfeed
3 - checkout	2018-01-28 13:26:02	10030	retargetting-campaign	email
4 - purchase	2018-01-28 13:38:02	10030	retargetting-campaign	email
1 - landing_page	2018-01-05 18:31:17	10045	getting-to-know-cool-tshirts	nytimes
2 - shopping_cart	2018-01-05 21:16:17	10045	getting-to-know-cool-tshirts	nytimes
3 - checkout	2018-01-09 03:05:17	10045	retargetting-ad	facebook

 */

 
/* Imagine June. She wants to buy a new t-shirt for her mother, who is visiting from out of town. She reads about CoolTShirts.com in a Buzzfeed article, and clicks a link to their landing page. June finds a fabulous Ninja Turtle t-shirt and adds it to her cart. Before she can advance to the checkout page her mom calls, asking for directions. June navigates away from CoolTShirts.com to look up directions.

June’s initial visit is logged in the page_visits table as follows:

user_id	timestamp	page_name	utm_source
10069	2018-01-02 23:14:01	1 - landing_page	buzzfeed
10069	2018-01-02 23:55:01	2 - shopping_cart	buzzfeed

June’s first touch — the first time she was exposed to CoolTShirts.com — is attributed to buzzfeed
June is assigned a user id of 10069
She visited the landing page at 23:14:01 and the shopping cart at 23:55:01 */

SELECT *
FROM page_visits
WHERE user_id = 10069 AND utm_source = 'buzzfeed'
LIMIT 100;

/* page_name	timestamp	user_id	utm_campaign	utm_source
1 - landing_page	2018-01-02 23:14:01	10069	ten-crazy-cool-tshirts-facts	buzzfeed
2 - shopping_cart	2018-01-02 23:55:01	10069	ten-crazy-cool-tshirts-facts	buzzfeed
 */

/*  
 Two days later, CoolTShirts.com runs an ad on June’s Facebook page. June remembers how much she wanted that Ninja Turtles t-shirt, and follows the ad back to CoolTShirts.com.

She now has the following rows in page_visits table:

user_id	timestamp	page_name	utm_source
10069	2018-01-02 23:14:01	1 - landing_page	buzzfeed
10069	2018-01-02 23:14:01	2 - shopping_cart	buzzfeed
10069	2018-01-04 08:12:01	3 - checkout	facebook
10069	2018-01-04 08:13:01	4 - purchase	facebook

June’s last touch — the exposure to CoolTShirts.com that led to a purchase — is attributed to facebook
She visited the checkout page at 08:12:01 and the purchase page at 08:13:01
 */
 
/* First versus Last
If you want to increase sales at CoolTShirts.com, would you count on buzzfeed or increase facebook ads? The real question is: should June’s purchase be attributed to buzzfeed or to facebook?

There are two ways of analyzing this:

First-touch attribution only considers the first utm_source for each customer, which would be buzzfeed in this case. This is a good way of knowing how visitors initially discover a website.
Last-touch attribution only considers the last utm_source for each customer, which would be facebook in this case. This is a good way of knowing how visitors are drawn back to a website, especially for making a final purchase.
The results can be crucial to improving a company’s marketing and online presence. Most companies analyze both first- and last-touch attribution and display the results separately. */

/* The Attribution Query I
We just learned how to attribute a user’s first and last touches. What if we want to attribute the first and last touches for ALL users? This is where SQL comes in handy — with one query we can find all first- or last-touch attributions (the first and last versions are nearly identical). We can save this query to run it later, or modify it for a subset of users. Let’s learn the query…

In order to get first-touch attributions, we need to find the first time that a user interacted with our website. We do this by using a GROUP BY. Let’s call this table first_touch:*/

SELECT user_id,
   MIN(timestamp) AS 'first_touch_at'
FROM page_visits
GROUP BY user_id;

/*
This tells us the first time that each user visited our site, but does not tell us how they got to our site — the query results have no UTM parameters! We’ll see how do get those in the next exercise.
   */
  
SELECT user_id,
  MIN(timestamp) AS 'first_touch_at',
  MAX(timestamp) AS 'last_touch_at'
FROM page_visits
--WHERE user_id = 10069
GROUP BY 1;

/* 
The Attribution Query II
To get the UTM parameters, we’ll need to JOIN these results back with the original table.

We’ll join tables first_touch, akaft, and page_visits, aka pv, on user_id and timestamp. */

ft.user_id = pv.user_id
AND ft.first_touch_at = pv.timestamp

/* 
Remember that first_touch_at is the earliest timestamp for each user. Here’s the simplified query: */

WITH first_touch AS (
      /* ... */
    )
SELECT *
FROM first_touch AS 'ft'
JOIN page_visits AS 'pv'
  ON ft.user_id = pv.user_id
  AND ft.first_touch_at = pv.timestamp;

/* Now fill in the WITH clause using the first_touch query from the previous exercise. We’ll also specify the columns to SELECT.*/

WITH first_touch AS (
   SELECT user_id,
      MIN(timestamp) AS 'first_touch_at'
   FROM page_visits
   GROUP BY user_id)
SELECT ft.user_id,
  ft.first_touch_at,
  pv.utm_source
FROM first_touch AS 'ft'
JOIN page_visits AS 'pv'
  ON ft.user_id = pv.user_id
  AND ft.first_touch_at = pv.timestamp;

/* 
The Attribution Query III
We can easily modify the first-touch attribution query to get last-touch attribution: use MAX(timestamp) instead of MIN(timestamp).

For reference, the first-touch attribution query is shown below. */

WITH first_touch AS (
    SELECT user_id,
       MIN(timestamp) AS 'first_touch_at'
    FROM page_visits
    GROUP BY user_id)
SELECT ft.user_id,
   ft.first_touch_at,
   pv.utm_source
FROM first_touch AS 'ft'
JOIN page_visits AS 'pv'
   ON ft.user_id = pv.user_id
   AND ft.first_touch_at = pv.timestamp;
   
---Now that you’ve seen how it works, it’s time to practice!

--- porjects ---

/* Get familiar with CoolTShirts
1.
How many campaigns and sources does CoolTShirts use? Which source is used for each campaign?

Use three queries:

one for the number of distinct campaigns,
one for the number of distinct sources,
one to find how they are related.

Hint
The first two queries will use SELECT COUNT(DISTINCT column_name).

The third query will use SELECT DISTINCT column1, column2.

2.
What pages are on the CoolTShirts website?

Find the distinct values of the page_name column.


Hint
There are four distinct values in page_name.

What is the user journey?
3.
How many first touches is each campaign responsible for?

You’ll need to use the first-touch query from the lesson (also provided in the hint below). Group by campaign and count the number of first touches for each.


Hint
Here’s the first-touch query: */

WITH first_touch AS (
    SELECT user_id,
        MIN(timestamp) as first_touch_at
    FROM page_visits
    GROUP BY user_id)
SELECT ft.user_id,
    ft.first_touch_at,
    pv.utm_source,
        pv.utm_campaign
FROM first_touch ft
JOIN page_visits pv
    ON ft.user_id = pv.user_id
    AND ft.first_touch_at = pv.timestamp;
	
/* Here’s the query to count first touches per campaign and source. first_touch is the set of all first touches. ft_attr is the same set with source and campaign columns added.

You may group by utm_campaign or both utm_campaignand utm_source. */

WITH first_touch AS (
  /*  ... */),
ft_attr AS (
  SELECT ft.user_id,
         ft.first_touch_at,
         pv.utm_source,
         pv.utm_campaign
  FROM first_touch ft
  JOIN page_visits pv
    ON ft.user_id = pv.user_id
    AND ft.first_touch_at = pv.timestamp
)
SELECT ft_attr.utm_source,
       ft_attr.utm_campaign,
       COUNT(*)
FROM ft_attr
GROUP BY 1, 2
ORDER BY 3 DESC;

/* 4.
How many last touches is each campaign responsible for?

Starting with the last-touch query from the lesson, group by campaign and count the number of last touches for each.


Hint
Here’s the query to count last touches per campaign and source. last_touch is the set of all last touches. lt_attr is the same set with source and campaign columns added.

You may group by utm_campaign or both utm_campaignand utm_source. */

WITH last_touch AS (
  /* ... */),
lt_attr AS (
  SELECT lt.user_id,
         lt.last_touch_at,
         pv.utm_source,
         pv.utm_campaign,
         pv.page_name
  FROM last_touch lt
  JOIN page_visits pv
    ON lt.user_id = pv.user_id
    AND lt.last_touch_at = pv.timestamp
)
SELECT lt_attr.utm_source,
       lt_attr.utm_campaign,
       COUNT(*)
FROM lt_attr
GROUP BY 1, 2
ORDER BY 3 DESC;

/* 5.
How many visitors make a purchase?

Count the distinct users who visited the page named 4 - purchase.


Hint
This can be done using a GROUP BY clause on page_name or a WHERE clause for page_name = '4 - purchase'.

6.
How many last touches on the purchase page is each campaign responsible for?

This query will look similar to your last-touch query, but with an additional WHERE clause.


Hint
In your subquery for last touches (the one using MAX(timestamp)), make sure to use a WHERE clause on page_name: */

WITH last_touch AS (
  SELECT user_id,
         MAX(timestamp) AS last_touch_at
  FROM page_visits
  WHERE page_name = '4 - purchase'
  GROUP BY user_id),

  /* 7.
CoolTShirts can re-invest in 5 campaigns. Given your findings in the project, which should they pick and why?


Hint
This task is open-ended. */

------ code start here -----------------

/*
Here's the first-touch query, in case you need it
*/

WITH first_touch AS (
    SELECT user_id,
        MIN(timestamp) as 'first_touch_at'
    FROM page_visits
    GROUP BY user_id),

last_touch AS (
    SELECT user_id,
        MAX(timestamp) as 'last_touch_at'
    FROM page_visits
    WHERE page_name = '4 - purchase'
    GROUP BY user_id),

join_ft AS(
    SELECT ft.user_id,
        ft.first_touch_at,
        pv.utm_source,
        pv.utm_campaign
    FROM first_touch AS 'ft'
    JOIN page_visits AS 'pv'
        ON ft.user_id = pv.user_id
        AND ft.first_touch_at = pv.timestamp),

join_lt AS (
    SELECT lt.user_id,
        lt.last_touch_at,
        pv.utm_source,
        pv.utm_campaign
    FROM last_touch AS 'lt'
    JOIN page_visits AS 'pv'
        ON lt.user_id = pv.user_id
        AND lt.last_touch_at = pv.timestamp)

-- query to find how they are related.
/*SELECT DISTINCT utm_source, utm_campaign
FROM page_visits;*/

-- find pages on the CoolTShirts website
/*SELECT DISTINCT page_name
FROM page_visits;*/

-- test for join_ft 
/*SELECT *
FROM join_ft
LIMIT 10;*/

-- how many first touch / last touch that campaign responsible for?
SELECT utm_campaign, COUNT(user_id) AS 'lead count'
FROM join_lt --FROM join_ft 
GROUP BY utm_campaign
ORDER BY 2 DESC;

-- How many visitors make a purchase?
--method 1
/*SELECT user_id,
    COUNT (CASE
      WHEN page_name = '4 - purchase' 
      THEN 'user_id' 
      END) AS 'is_purchased',
    timestamp
FROM page_visits;*/

--method 2
/*
SELECT user_id, timestamp, page_name, COUNT(*)
FROM page_visits
GROUP BY page_name
ORDER BY page_name;*/

--method 3
/*SELECT user_id, timestamp, page_name, COUNT(*)
FROM page_visits
WHERE page_name = '4 - purchase';*/



---------- hacker news porjects code----------------------------------

/*SELECT title, score
FROM hacker_news
ORDER BY score DESC
LIMIT 5;*/

WITH grand_total_score AS (
  SELECT SUM(score) AS 'gts'
  FROM hacker_news),

total_score AS (
  SELECT user, SUM(score) AS 'TOT_SCORE'
  FROM hacker_news
  GROUP BY user
  HAVING TOT_SCORE > 200
  ORDER BY TOT_SCORE DESC),

source AS (

  SELECT CASE
    WHEN url LIKE '%github.com%' THEN 'GitHub'
    WHEN url LIKE '%medium.com%' THEN 'Medium'
    WHEN url LIKE '%nytimes.com%' THEN 'New York Times'
    --WHEN url LIKE '%github.com%' THEN 'GitHub'
    ELSE 'Other'

    END AS 'Source',
    COUNT(*)
  FROM hacker_news
  GROUP BY Source),

artical_by_time AS (
  SELECT strftime('%H', timestamp) AS 'Hours',
    ROUND(AVG(score)) AS 'AVG_SCORE',
    COUNT(*) 'NUMBER OF STORY'
  FROM hacker_news
  WHERE timestamp IS NOT NULL
  GROUP BY Hours
  ORDER BY Hours),

percent_top_user AS (
  SELECT (517 + 309 + 304 + 282) / 6366.0)

-- strftime function sample
/*SELECT timestamp,
   strftime('%H', timestamp)
FROM hacker_news
GROUP BY 1
LIMIT 20;*/

SELECT *
FROM artical_by_time;

-- sampel code --------

SELECT 
  SUM(
    CASE WHEN (currency = 'BIT') AND (money_in IS NOT NULL)
    THEN 1
    ELSE 0
    END ) 
FROM transactions;



SELECT MAX(MAX(money_in), MAX(money_out))
FROM transactions;

----------------------------







  
/*############################################################################################################
---------------- SQL CHEATSHEETS SQL CHEATSHEETS SQL CHEATSHEETS SQL CHEATSHEETS SQL CHEATSHEETS-------------
############################################################################################################*/

--------------------Manipulation--------------------

--CREATE TABLE Statment
/* The CREATE TABLE statement creates a new table in a database. It allows one to specify the name of the table and the name of each column in the table. */

CREATE TABLE table_name (
  column1 datatype,
  column2 datatype,
  column3 datatype
);

/* INSERT Statement
The INSERT INTO statement is used to add a new record (row) to a table.

It has two forms as shown:

Insert into columns in order.
Insert into columns by name. */

-- Insert into columns in order:
INSERT INTO table_name
VALUES (value1, value2);

-- Insert into columns by name:
INSERT INTO table_name (column1, column2)
VALUES (value1, value2);

/* UPDATE Statement
The UPDATE statement is used to edit records (rows) in a table. It includes a SET clause that indicates the column to edit and a WHERE clause for specifying the record(s). */

UPDATE table_name
SET column1 = value1, column2 = value2
WHERE some_column = some_value;

/* ALTER TABLE Statement
The ALTER TABLE statement is used to modify the columns of an existing table. When combined with the ADD COLUMN clause, it is used to add a new column. */

ALTER TABLE table_name
ADD column_name datatype;

/* DELETE Statement
The DELETE statement is used to delete records (rows) in a table. The WHERE clause specifies which record or records that should be deleted. If the WHERE clause is omitted, all records will be deleted. */

DELETE FROM table_name
WHERE some_column = some_value;

/* Column Constraints
Column constraints are the rules applied to the values of individual columns: 

PRIMARY KEY constraint can be used to uniquely identify the row.
UNIQUE columns have a different value for every row.
NOT NULL columns must have a value.
DEFAULT assigns a default value for the column when no value is specified.
There can be only one PRIMARY KEY column per table and multiple UNIQUE columns.*/

CREATE TABLE student (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  grade INTEGER NOT NULL,
  age INTEGER DEFAULT 10
);


----------------------Queries-----------------------------------

/* SELECT Statement
The SELECT * statement returns all columns from the provided table in the result set. The given query will fetch all columns and records (rows) from the movies table.
 */
SELECT *
FROM movies;

/* AS Clause
Columns or tables in SQL can be aliased using the AS clause. This allows columns or tables to be specifically renamed in the returned result set. The given query will return a result set with the column for name renamed to movie_title. */

SELECT name AS 'movie_title'
FROM movies;

/* DISTINCT Clause
Unique values of a column can be selected using a DISTINCT query. For a table contact_details having five rows in which the city column contains Chicago, Madison, Boston, Madison, and Denver, the given query would return: 

Chicago
Madison
Boston
Denver*/

SELECT DISTINCT city
FROM contact_details;

-- if dictinct follow by 2 column names. both of the distinct will be displayed.
SELECT DISTINCT utm_source, utm_campaign
FROM page_visits;
 
 /* 
 utm_source	utm_campaign
nytimes	getting-to-know-cool-tshirts
email	weekly-newsletter
buzzfeed	ten-crazy-cool-tshirts-facts
email	retargetting-campaign
facebook	retargetting-ad
medium	interview-with-cool-tshirts-founder
google	paid-search
google	cool-tshirts-search */


/* WHERE Clause
The WHERE clause is used to filter records (rows) that match a certain condition. The given query will select all records where the pub_year equals 2017. */

SELECT title
FROM library
WHERE pub_year = 2017;

/* LIKE Operator
The LIKE operator can be used inside of a WHERE clause to match a specified pattern. The given query will match any movie that begins with Star in its title. */

SELECT name
FROM movies
WHERE name LIKE 'Star%';

/* _ Wildcard
The _ wildcard can be used in a LIKE operator pattern to match any single unspecified character. The given query will match any movie which begins with a single character, followed by ove. */

SELECT name
FROM movies
WHERE name LIKE '_ove';

/* % Wildcard
The % wildcard can be used in a LIKE operator pattern to match zero or more unspecified character(s). The example query will match any movie that begins with The, followed by zero or more of any characters. */

SELECT name
FROM movies
WHERE name LIKE 'The%';

/* NULL Values
Column values in SQL records can be NULL, or have no value. These records can be matched (or not matched) using the IS NULL and IS NOT NULL operators in combination with the WHERE clause. The given query will match all addresses where the address has a value or is not NULL. */

SELECT address
FROM records
WHERE address IS NOT NULL;

/* BETWEEN Operator
The BETWEEN operator can be used to filter by a range of values. The range of values can be text, numbers or date data. The given query will match any movie made between the years 1980 and 1990, inclusive. */

SELECT *
FROM movies
WHERE year BETWEEN 1980 AND 1990;

/* AND Operator
The AND operator allows multiple conditions to be combined. Records must match both conditions that are joined by AND to be included in the result set. The example query will match any car that is blue and made after 2014. */

SELECT model 
FROM cars 
WHERE color = 'blue' 
  AND year > 2014;
  
/* OR Operator
The OR operator allows multiple conditions to be combined. Records matching either condition joined by the OR are included in the result set. The given query will match customers whose state is either ca or ny. */

SELECT name
FROM customers 
WHERE state = "ca" 
   OR state = "ny";
   
/* ORDER BY Clause
The ORDER BY clause can be used to sort the result set by a particular column either alphabetically or numerically. It can be ordered in ascending (default) or descending order with ASC/DESC. In the example, all the rows of the contacts table will be ordered by the birth_date column in descending order. */

SELECT *
FROM contacts
ORDER BY birth_date DESC;

/* LIMIT Clause
The LIMIT clause is used to narrow, or limit, a result set to the specified number of rows. The given query will limit the result set to 5 rows. */

SELECT *
FROM movies
LIMIT 5;

-------------Aggregate Functions----------------------

/* Aggregate Functions in SQL
Aggregate functions perform a calculation on a set of values and return a single value: */

COUNT()
SUM()
MAX()
MIN()
AVG()

/* COUNT() Aggregate Function
The COUNT() aggregate function in SQL returns the total number of rows that match the specified criteria. For instance, to find the total number of employees who have less than 5 years of experience, the given query can be used.

Note: A column name of the table can also be used instead of *. Unlike COUNT(*), this variation COUNT(column) will not count NULL values in that column. */

SELECT COUNT(*)
FROM employees
WHERE experience < 5;

/* SUM() Aggregate Function
The SUM() aggregate function takes the name of a column as an argument and returns the sum of all the value in that column. */

SELECT SUM(salary)
FROM salary_disbursement;\

/* AVG() Aggregate Function
The AVG() aggregate function returns the average value in a column. For instance, to find the average salary for the employees who have less than 5 years of experience, the given query can be used. */

SELECT AVG(salary)
FROM employees
WHERE experience < 5;

/* ROUND() Function
The ROUND() function will round a number value to a specified number of places. It takes two arguments: a number, and a number of decimal places. It can be combined with other aggregate functions, as shown in the given query. This query will calculate the average rating of movies from 2015, rounding to 2 decimal places. */

SELECT year, 
   ROUND(AVG(rating), 2) 
FROM movies 
WHERE year = 2015;

/* GROUP BY Clause
The GROUP BY clause will group records in a result set by identical values in one or more columns. It is often used in combination with aggregate functions to query information of similar records. The GROUP BY clause can come after FROM or WHERE but must come before any ORDER BY or LIMIT clause.

The given query will count the number of movies per rating. */

SELECT rating, 
   COUNT(*) 
FROM movies 
GROUP BY rating;

/* Column References
The GROUP BY and ORDER BY clauses can reference the selected columns by number in which they appear in the SELECT statement. The example query will count the number of movies per rating, and will:

GROUP BY column 2 (rating)
ORDER BY column 1 (total_movies) */

SELECT COUNT(*) AS 'total_movies', 
   rating 
FROM movies 
GROUP BY 2 
ORDER BY 1;

/* HAVING Clause
The HAVING clause is used to further filter the result set groups provided by the GROUP BY clause. HAVING is often used with aggregate functions to filter the result set groups based on an aggregate property. The given query will select only the records (rows) from only years where more than 5 movies were released per year. */

SELECT year, 
   COUNT(*) 
FROM movies 
GROUP BY year
HAVING COUNT(*) > 5;

/* MAX() Aggregate Function
The MAX() aggregate function in SQL takes the name of a column as an argument and returns the largest value in a column. The given query will return the largest value from the amount column. */

SELECT MAX(amount) 
FROM transactions;

/* MIN() Aggregate Function
The MIN() aggregate function in SQL returns the smallest value in a column. For instance, to find the smallest value of the amount column from the table named transactions, the given query can be used. */

SELECT MIN(amount) 
FROM transactions;

----------------------Multiple Tables-------------------------

/* Multiple Tables
Inner Join
The JOIN clause allows for the return of results from more than one table by joining them together with other results based on common column values specified using an ON clause. INNER JOIN is the default JOIN and it will only return results matching the condition specified by ON. */

SELECT * 
FROM books
JOIN authors
  ON books.author_id = authors.id;
  
/* Outer Join
An outer join will combine rows from different tables even if the join condition is not met. In a LEFT JOIN, every row in the left table is returned in the result set, and if the join condition is not met, then NULL values are used to fill in the columns from the right table.*/
 
SELECT column_name(s)
FROM table1
LEFT JOIN table2
  ON table1.column_name = table2.column_name;
  
/* Primary Key
A primary key column in a SQL table is used to uniquely identify each record in that table. A primary key cannot be NULL. In the example, customer_id is the primary key. The same value cannot re-occur in a primary key column. Primary keys are often used in JOIN operations. */

/* Foreign Key
A foreign key is a reference in one table’s records to the primary key of another table. To maintain multiple records for a specific row, the use of foreign key plays a vital role. For instance, to track all the orders of a specific customer, the table order (illustrated at the bottom of the image) can contain a foreign key. */

/* CROSS JOIN Clause
The   clause is used to combine each row from one table with each row from another in the result set. This JOIN is helpful for creating all possible combinations for the records (rows) in two tables.

The given query will select the shirt_color and pants_color columns from the result set, which will contain all combinations of combining the rows in the shirts and pants tables. If there are 3 different shirt colors in the shirts table and 5 different pants colors in the pants table then the result set will contain 3 x 5 = 15 rows. */

SELECT shirts.shirt_color,
   pants.pants_color
FROM shirts
CROSS JOIN pants;

/* UNION Clause
The UNION clause is used to combine results that appear from multiple SELECT statements and filter duplicates.

For example, given a first_names table with a column name containing rows of data “James” and “Hermione”, and a last_names table with a column name containing rows of data “James”, “Hermione” and “Cassidy”, the result of this query would contain three names: “Cassidy”, “James”, and “Hermione”. */

SELECT name
FROM first_names
UNION
SELECT name
FROM last_names

/* WITH Clause
The WITH clause stores the result of a query in a temporary table (temporary_movies) using an alias.

Multiple temporary tables can be defined with one instance of the WITH keyword. */

WITH temporary_movies AS (
   SELECT *
   FROM movies
)
SELECT *
FROM temporary_movies
WHERE year BETWEEN 2000 AND 2020;

/* 
SQLite comes with a strftime() function - a very powerful function that allows you to return a formatted date.

It takes two arguments:

strftime(format, column)

Let’s test this function out:
 */

SELECT timestamp,
   strftime('%H', timestamp)
FROM hacker_news
GROUP BY 1
LIMIT 20;

/* timestamp	strftime('%H', timestamp)
2007-03-16T20:52:19Z	20
2007-04-03T03:04:09Z	03
2007-05-01T03:11:17Z	03
2007-05-05T05:43:58Z	05
2007-05-11T05:48:53Z	05
2007-05-25T22:07:18Z	22 */

/* For strftime(__, timestamp):

%Y returns the year (YYYY)
%m returns the month (01-12)
%d returns the day of the month (1-31)
%H returns 24-hour clock (00-23)
%M returns the minute (00-59)
%S returns the seconds (00-59)
if time format is YYYY-MM-DD HH:MM:SS.
 */