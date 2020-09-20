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
	
	
import pandas as pd
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print df
#Its output is as follows −
'''
      Name      Age
0     Alex      10
1     Bob       12
2     Clarke    13
'''

flag_to_check = pd.DataFrame ([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], 
columns =  ['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange','Circles','Crosses', 'Saltires', 'Quarters', 'Sunstars', 'Crescent', 'Triangle'])


		
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

# select certain column sample 2


To select columns from a DataFrame:

name = df[['Column1', 'Column2']]

So it should look something like:

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]



# select certain row

df.iloc[2]
df.iloc[2:]
df.iloc[2:3]
df.iloc[:4]
df.iloc[-2]
df.iloc[[1,3,5]]

#return value of a cell

In [3]: sub_df
Out[3]:
          A         B
2 -0.133653 -0.030854

In [4]: sub_df.iloc[0]
Out[4]:
A   -0.133653
B   -0.030854
Name: 2, dtype: float64

In [5]: sub_df.iloc[0]['A']
Out[5]: -0.13365288513107493

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

df.rename(columns = {'name' : 'new_name',
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
menu = pd.concate([df1], [df2]) # zhe ge you cuowu

#concat and re index
pd.concat([s1, s2], ignore_index=True)

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


# rearrange column sequence 

df = df[['x', 'y', 'a', 'b']]



#delete column
df = df.drop(columns = 'column name')

#rename column

df.rename(columns = {'old_name' : 'new_name',
					 'old_name1' : 'new_name1'},
					 inplace = True)
					 
#------------------read excel file---------------------

import io
import requests
import pandas as pd
from zipfile import ZipFile
r = requests.get('http://www.contextures.com/SampleData.zip')
ZipFile(io.BytesIO(r.content)).extractall()

df = pd.read_excel('SampleData.xlsx', sheet_name='SalesOrders')
					 

#----------------- read html file ------------------

dfs = pd.read_html(html_string)

dfs = pd.read_html('http://www.contextures.com/xlSampleData01.html')

df = pd.read_html(html_string, header=0)[0]
#since there could be many tables in one html, dfs stands for many df . and [0] indicate the first one.

#header = 0 only used when header become one row of data 

#--------------- to excel ---------------

df.to_csv('ozito.csv')

df1.to_excel("output.xlsx")

df1.to_excel("output.xlsx", sheet_name='Sheet_name_1')
