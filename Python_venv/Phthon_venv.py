#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-Venv python Virtual enviroment Venv #-#-#-#-#-#-#-#-#-
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# in windows enviroment

'''
Why use Virtual Enviroment ?

in virtual enviroment, the package is installed only for that specific job. 
that prevent things happend like when you update your certain package, and old script not working. 

if you use a single globe enviroment, when you update some package, it may borke some old project. 

How to create:

it come with python , any version higher than 3.3 is good. 

'''

pip list # check what package is installed

python -m venv new_venv # create a new enviroment , -m means run a moudle

new_venv\Scripts\activate.bat #activate enviroment

where python # check if it is activated

pip list 

pip freeze # same as pip list but give in a correct form for txt file

#export this to a txt file. (new_venv.txt)

pip install packagename #anything install from now will be only for this venv

new_venv\Scripts\deactivate.bat #deactivate current enviroment

rmdir new_venv /s #delete the proejct and virtual enviroment

mkdir Another_Project

python -m venv Another_Project\venv #make a new venv for another proejct

Another_Project\venv\Scripts\activate.bat # activate this enviroment

pip install -r new_venv.txt # this will install all the listed package in the txt file

pip list # we should see all the package is installed.  

#now you can start to coding, all the code and resoures stay at root of Another_Project\ . but should not stay under venv.
#the venv is a folder you can totally throw away and rebuild

python -m venv third_venv --system-site-packages # this command will create a new v enviroment that have access to the globe packages 

third_venv\scripts\activate.bat

pip list # now you can see the globe package are included also here

# after this env is activated, any package installed will only be for this project, globle package is not affected

pip list --local # check local list

pip freeze --local 


# to activate an virtual enviroment  in Git Bash, you can use

source /venv/Scripts/activate

#
cd venv
cd Scripts
source activate

#
 
source ./venv/Scripts/activate
