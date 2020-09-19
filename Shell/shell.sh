ls
ls -a
ls -alt
ls ../paint/


#-a - lists all contents, including hidden files and directories
#-l - lists all contents of a directory in long format
#-t - order files and directories by the time they were last modified.


pwd
#print working directory

cd

cd jan/memory

cd ..
cd ../ ..

cd ../feb
cd ../../comedy

mkdir media

touch aa.txt # create a new file called aa.txt
touch bmx/tricks.txt

cp frida.txt lincoln.txt 
cp biopic/ray.txt biopic/notorious.txt historical/ 
cp * satire/ #copy only files not folder
cp m*.txt scifi/ 

mv superman.txt superhero/ 
mv wonderwoman.txt batman.txt superhero/
mv batman.txt spiderman.txt #rename file

rm waterboy.txt
rm -r comedy  #remove folder
#The -r is an option that modifies the behavior of the rm command. The -r stands for “recursive,” and it’s used to delete a directory and all of its child directories.
rm -f #   -f, --force  / ignore nonexistent files, never prompt 

#--------------------

#standard input, abbreviated as stdin, is information inputted into the terminal through the keyboard or input device.

#standard output, abbreviated as stdout, is the information outputted after a process is run.

#standard error, abbreviated as stderr, is an error message outputted by a failed process.

echo "Hello" 
echo "Hello" > hello.txt 

cat hello.txt 
cat oceans.txt > continents.txt 

#> takes the standard output of the command on the left, and redirects it to the file on the right.

cat glaciers.txt >> rivers.txt 
#>> takes the standard output of the command on the left and appends (adds) it to the file on the right. 

cat < lakes.txt
#< takes the standard input from the file on the right and inputs it into the program on the left.

cat volcanoes.txt | wc  
#| is a “pipe”. The | takes the standard output of the command on the left, and pipes it as standard input to the command on the right. You can think of this as “command to command” redirection.

#in turn, the wc command outputs the number of lines, words, and characters in volcanoes.txt, respectively.

cat volcanoes.txt | wc | cat > islands.txt 

wc -l < plants.txt

sort lakes.txt 
cat lakes.txt | sort > sorted-lakes.txt 

uniq deserts.txt 
sort deserts.txt | uniq > uniq-deserts.txt 
#uniq stands for “unique” and filters out adjacent, duplicate lines in a file.

grep Mount mountains.txt 
grep -i Mushroom fungi.txt
#grep stands for “global regular expression print”. It searches files for lines that match a pattern and returns the results. It is also case sensitive. Here, grep searches for “Mount” in mountains.txt

#grep -i enables the command to be case insensitive.

grep -R Arctic /home/ccuser/workspace/geography
# return  /home/ccuser/workspace/geography/aa.txt:Artic Desect

#grep -R searches all files in a directory and outputs filenames and lines containing matched results. -R stands for “recursive”. Here grep -R searches the /home/ccuser/workspace/geography directory for the string “Arctic” and outputs filenames and lines with matched results.

grep -Rl Arctic /home/ccuser/workspace/
#returns /home/ccuser/workspace/Artic_aa.txt

#grep -Rl searches all files in a directory and outputs only filenames with matched results. -R stands for “recursive” and l stands for “files with matches”. Here grep -Rl searches the /home/ccuser/workspace/geography directory for the string “Arctic” and outputs filenames with matched results.

grep -R player . 
# search string 'player' in the current diretory


sed 's/snow/rain/' forests.txt 
sed 's/Dirt/Soils/g' soils.txt

#s: stands for “substitution”. it is always used when using sed for substitution.
#snow: the search string, the text to find.
#rain: the replacement string, the text to add in place.


ls -l | head > list1.txt

ls -la | head >> list1.txt | wc 

nano a.txt

#Ctrl + O saves a file. ‘O’ stands for output.
#Ctrl + X exits the nano program. ‘X’ stands for exit.
#Ctrl + G opens a help menu.

clear 

# clear screen

history

#command line outputs a history of commands that were entered in the current session.

date

#print out current date

less a.txt
#On Linux systems, less is a command that displays file contents or command output one page at a time in your terminal. less is most useful for viewing the content of large files or the results of commands that produce many lines of output. The content displayed by less can be navigated by entering keyboard shortcuts.
#use q for finish




nano ~/.bash_profile

source ~/.bash_profile

#~/.bash_profile is the name of file used to store environment settings. It is commonly called the “bash profile”. When a session starts, it will load the contents of the bash profile before executing commands.

#The ~ represents the user’s home directory.
#The . indicates a hidden file.
#The name ~/.bash_profile is important, since this is how the command line recognizes the bash profile.

#command source ~/.bash_profile activates the changes in ~/.bash_profile for the current session. Instead of closing the terminal and needing to start a new session, source makes the changes available right away in the session we are in.

# ~/.bash_profile ------------

alias pd="pwd"

export USER="Jane Doe" 

# use this command , echo $USER at 

#line USER="Jane Doe" sets the environment variable USER to a name “Jane Doe”. Usually the USER variable is set to the name of the computer’s owner.

#The line export makes the variable to be available to all child sessions initiated from the session you are in. This is a way to make the variable persist across programs.

#At the command line, the command echo $USER returns the value of the variable. Note that $ is always used when returning a variable’s value. Here, the command echo $USER returns the name set for the variable.

export PS1=">> " 

#PS1 is a variable that defines the makeup and style of the command prompt.

#export PS1=">> " sets the command prompt variable and exports the variable. Here we change the default command prompt from $ to >>.
#After using the source command, the command line displays the new command prompt.

HOME

echo $HOME #type in command line. not in bash_profile

#The HOME variable is an environment variable that displays the path of the home directory. Here by typing 

PATH

echo $PATH #type in command line. not in bash_profile

#PATH is an environment variable that stores a list of directories separated by a colon. Looking carefully

#/bin/pwd
#/bin/ls

LESS

export LESS="-N"

#Open the bash profile, and create and export a new environment variable called LESS, setting it equal to the option "-N". The -N option adds line numbers to the file.



# ~/.bash_profile ----- end -------

env

env | grep PATH

#The env command stands for “environment”, and returns a list of the environment variables for the current user.

#----------------------- bash script ---------------------------------

#The beginning of your script file should start with #!/bin/bash on its own line. This tells the computer which type of interpreter to use for the script. 

#When saving the script file, it is good practice to place commonly used scripts in the ~/bin/ directory.

#The script files also need to have the “execute” permission to allow them to be run. To add this permission to a file with filename: script.sh use:

chmod +x script.sh

#Your terminal runs a file every time it is opened to load its configuration.
#On Linux style shells, this is ~/.bashrc and on OSX, this is ~/.bash_profile.

#To ensure that scripts in ~/bin/ are available, you must add this directory to your PATH within your configuration file:

PATH=~/bin:$PATH


#Use ./script.sh to run the script.
./script.sh

greeting="Hello" #Note that there is no space between the variable name, the equals sign, or “Hello”.

#To access the value of a variable, we use the variable name prepended with a dollar sign ($).

echo $greeting

#When bash scripting, you can use conditionals to control which set of commands within the script run. Use if to start the conditional, followed by the condition in square brackets ([ ]). then begins the code that will run if the condition is met. else begins the code that will run if the condition is not met. Lastly, the conditional is closed with a backwards if, fi.


if [ $index -lt 5 ]
then
  echo $index
else
  echo 5
fi

#Equal: -eq
#Not equal: -ne
#Less than or equal: -le
#Less than: -lt
#Greater than or equal: -ge
#Greater than: -gt
#Is null: -z

#When comparing strings, it is best practice to put the variable into quotes ("). This prevents errors if the variable is null or contains spaces. The common operators for comparing strings are:

#Equal: ==
#Not equal: !=

#For example, to compare if the variables foo and bar contain the same string:

if [ "$foo" == "$bar"]

#----sample script.sh-------

#!/bin/bash
first_greeting="Nice to meet you!"
later_greeting="How are you?"
greeting_occasion=1

if [ $greeting_occasion -lt 1 ]
then
  echo $first_greeting
else
  echo $later_greeting
fi

#------- end sample ----------

#There are 3 different ways to loop within a bash script: for, while and until.

#For example, if we had a list of words stored in a variable paragraph, we could use the following syntax to print each one:

for word in $paragraph
do
  echo $word
done

#Note that word is being “defined” at the top of the for loop so there is no $ prepended. Remember that we prepend the $ when accessing the value of the variable. So, when accessing the variable within the do block, we use $word as usual.

#Within bash scripting until and while are very similar. while loops keep looping while the provided condition is true whereas until loops loop until the condition is true. Conditions are established the same way as they are within an if block, between square brackets. If we want to print the index variable as long as it is less than 5, we would use the following while loop:

while [ $index -lt 5 ]
do
  echo $index
  index=$((index + 1))
done
#Note that arithmetic in bash scripting uses the $((...)) syntax and within the brackets the variable name is not prepended with a $.

#The same loop could also be written as an until loop as follows:

until [ $index -eq 5 ]
do
  echo $index
  index=$((index + 1))
done

#

echo "Guess a number"
read number
echo "You guessed $number"

#Another way to access external data is to have the user add input arguments when they run your script. These arguments are entered after the script name and are separated by spaces. For example:

saycolors red green blue

#Within the script, these are accessed using $1, $2, etc, where $1 is the first argument (here, “red”) and so on. Note that these are 1 indexed.

#If your script needs to accept an indefinite number of input arguments, you can iterate over them using the "$@" syntax. For our saycolors example, we could print each color using:

for color in "$@"
do
  echo $color
done

#Lastly, we can access external files to our script. You can assign a set of files to a variable name using standard bash pattern matching using regular expressions. For example, to get all files in a directory, you can use the * character:

files=/some/directory/*

#You can then iterate through each file and do something. Here, lets just print the full path and filename:

for file in $files
do
  echo $file
done

# set up aliases

alias saycolors='./saycolors.sh'

#You can even add standard input arguments to your alias. For example, if we always want “green” to be included as the first input to saycolors, we could modify our alias to:

alias saycolors='./saycolors.sh "green"'

#you can also make alias in command line.

#sample of script.sh
#!/bin/bash
first_greeting="Nice to meet you!"
later_greeting="How are you?"
greeting_occasion=0
greeting_limit=$1
while [ $greeting_occasion -lt $greeting_limit ]
do
  if [ $greeting_occasion -lt 1 ]
  then
    echo $first_greeting
  else
    echo $later_greeting
  fi
  greeting_occasion=$((greeting_occasion + 1))
done

#------------
#!/bin/bash
first_greeting="Nice to meet you!"
later_greeting="How are you?"
greeting_occasion=0

echo "How many times should I greet?"
read greeting_limit
while [ $greeting_occasion -lt $greeting_limit ]
do
  if [ $greeting_occasion -lt 1 ]
  then
    echo $first_greeting
  else
    echo $later_greeting
  fi
  greeting_occasion=$((greeting_occasion + 1))
done
 

#sample code ------------------------------

#One common use of bash scripts is for releasing a “build” of your source code. Sometimes your private source code may contain developer resources or private information that you don’t want to release in the published version.

#In this project, you’ll create a release script to copy certain files from a source directory into a build directory.


#!/bin/bash
echo "🔥🔥🔥Beginning build!! 🔥🔥🔥"

firstline=$(head -n 1 source/changelog.md) # read first line from file changelog.md
read -a splitfirstline <<< $firstline # split string into array

version=${splitfirstline[1]} #get the versioin number from array, the index is 1
echo "You are building version" $version

echo 'Do you want to continue? (enter "1" for yes, "0" for no)' # ask user if want to continue
read versioncontinue

if [ $versioncontinue -eq 1 ]
then 
  echo "OK"
  for filename in source/*
    do
    echo $filename # show user the filenames
    if [ "$filename" == "source/secretinfo.md" ] # not copying the secretinfo
    then
      echo "Not copying" $filename
    else
      echo "Copying" $filename
      cp $filename build/.
    fi
  done
  cd build/
    echo "Build version $version contains:" # show the end result in build folder
ls
  cd ..
else
  echo "Please come back when you are ready" # if user not chose 1 , just stop running.
fi
 
#some more ideas
#Copy secretinfo.md but replace “42” with “XX”.
#Zip the resulting build directory.
#Give the script more character with emojis.
#If you are familiar with git, commit the changes in the build directory.



