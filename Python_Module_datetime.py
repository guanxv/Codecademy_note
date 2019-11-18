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