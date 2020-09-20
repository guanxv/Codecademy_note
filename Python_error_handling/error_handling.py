#---------------------------------------------------------------------------------
#Error Handling-------------------------------------------------------------------
#---------------------------------------------------------------------------------

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
    

#raise your own exception2    

def sum_to_one(n):
  if n < 0:
    SumToOneException("0 or Positive Numbers Only!")
  if n <= 1:
    return n
  return n + sum_to_one(????)

    