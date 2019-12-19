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
