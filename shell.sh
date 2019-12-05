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

sort lakes.txt 
cat lakes.txt | sort > sorted-lakes.txt 

uniq deserts.txt 
sort deserts.txt | uniq > uniq-deserts.txt 
#uniq stands for “unique” and filters out adjacent, duplicate lines in a file.

grep Mount mountains.txt 
#grep stands for “global regular expression print”. It searches files for lines that match a pattern and returns the results. It is also case sensitive. Here, grep searches for “Mount” in mountains.txt

#grep -i enables the command to be case insensitive.

grep -R Arctic /home/ccuser/workspace/geography
# return  /home/ccuser/workspace/geography/aa.txt:Artic Desect

#grep -R searches all files in a directory and outputs filenames and lines containing matched results. -R stands for “recursive”. Here grep -R searches the /home/ccuser/workspace/geography directory for the string “Arctic” and outputs filenames and lines with matched results.

grep -Rl Arctic /home/ccuser/workspace/
#returns /home/ccuser/workspace/Artic_aa.txt

#grep -Rl searches all files in a directory and outputs only filenames with matched results. -R stands for “recursive” and l stands for “files with matches”. Here grep -Rl searches the /home/ccuser/workspace/geography directory for the string “Arctic” and outputs filenames with matched results.

sed 's/snow/rain/' forests.txt 

#s: stands for “substitution”. it is always used when using sed for substitution.
#snow: the search string, the text to find.
#rain: the replacement string, the text to add in place.


