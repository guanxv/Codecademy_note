# workflow to push existing project to github

1. set up the empty repository at github
2. git init
3. git add .
4. git commit -m "first commit"
5. git remote add origin remote repository URL
6. git remote -v
7. git push origin master


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT GIT #-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
git init
#initialize git

git status

git add filename [can be multiple filename s]
git add . # add every file in the staging area

#(use "git restore --staged <file>..." to unstage)


git diff filename
#compare working directory and staging area
#can also use "git diff" to check last changes

git commit -m "Complete first line of dialogue"

git log

#----------------

git init creates a new Git repository
git status inspects the contents of the working directory and staging area
git add adds files from the working directory to the staging area
git diff shows the difference between the working directory and the staging area
git commit permanently stores file changes from the staging area in the repository
git log shows a list of all previous commits

#----------------------

git remote add origin https://github.com/guanxv/projectname.git
git remote -v # verify the remote
git push -u origin master 
git pull # bring the change from original

git show HEAD # show the latested changes
git checkout HEAD filename#The command will restore the file in your working directory to look exactly as it did when you last made a commit.
git checkout -- filename # exactly same function as above

git reset HEAD filename
#This command resets the file in the staging area to be the same as the HEAD commit. It does not discard file changes from the working directory, it just removes them from the staging area.

git reset commit_SHA
#This command works by using the first 7 characters of the SHA of a previous commit. For example, if the SHA of the previous commit is 5d692065cf51a2f50ea8e7b19b5a7ae512f633ba , use 5d69206

#HEAD is most recent changes


git checkout HEAD filename: Discards changes in the working directory.
git reset HEAD filename: Unstages file changes in the staging area.
git reset commit_SHA: Resets to a previous commit in your commit history.

git branch
#show current branch

git branch new_branch 
#to create a new branch,  branch names can’t contain whitespaces

git checkout branch_name
#switch to branch

git merge branchF_name
#merging the branch into master

#Your goal is to update master with changes you made to fencing.
#fencing is the giver branch, since it provides the changes.
#master is the receiver branch, since it accepts those changes.

git branch -d branch_name
#delete branch

git branch#: Lists all a Git project’s branches.
git branch branch_name#: Creabites a new branch.
git checkout branch_name#: Used to switch from one branch to another.
git merge branch_name#: Used to join file changes from one branch to another.
git branch -d branch_name#: Deletes the branch specified.

git clone remote_location clone_name
#remote_location tells Git where to go to find the remote. This could be a web address, or a filepath, such as:
#/Users/teachers/Documents/some-remote

#clone_name is the name you give to the directory in which Git will clone the repository.
 
git remote -v
#Git lists the name of the remote, origin, as well as its location.
#Git automatically names this remote origin, because it refers to the remote repository of origin. However, it is possible to safely change its name.
#The remote is listed twice: once for (fetch) and once for (push). We’ll learn about these later in the lesson.

git fetch
#This command will not merge changes from the remote into your local repository. It brings those changes onto what’s called a remote branch.


#Now we’ll use the git merge command to integrate origin/master into your local master branch. The command:

git merge origin/master
#Even though Sally’s new commits have been fetched to your local copy of the Git project, those commits are on the origin/master branch. Your local master branch has not been updated yet, so you can’t view or make changes to any of the work she has added.

#The workflow for Git collaborations typically follows this order:

#Fetch and merge changes from the remote
#Create a branch to work on a new project feature
#Develop the feature on your branch and commit your work
#Fetch and merge from the remote again (in case new commits were made while you were working)
#Push your branch up to the remote for review

git push origin your_branch_name
#will push your branch up to the remote, origin.

git clone#: Creates a local copy of a remote.
git remote -v#: Lists a Git project’s remotes.
git fetch#: Fetches work from the remote into the local copy.
git merge origin/master#: Merges origin/master into your local branch.
git push origin <branch_name>#: Pushes a local branch to the origin remote.


#bash command
cd ../veggie-clone


#git config

git config --global user.name "xxx"
git config --global user.email "xxx@gmail.com"

git config --list


#add a text file called .gitignore at the root of the porject can make git not monitoring certain files
.gitignore

#if the .gitignore added later , you have to run

git rm -rf --cached .  # remove everything from the repositroy
git add . # add everyting back, but under .gitignore control

#to make it work 


# .gitignore rules

target/	#ignore every …folder (due to the trailing /) recursively
target	#ignore every …file or folder named target recursively
/target	#ignore every …file or folder named target in the top-most directory (due to the leading /)
/target/#ignore every …folder named target in the top-most directory (leading and trailing /)
*.class	#ignore every …every file or folder ending with .class recursively


#comment        #…nothing, this is a comment (first character is a #)
\#comment       #…every file or folder with name #comment (\ for escaping)
target/logs/    #…every folder named logs which is a subdirectory of a folder named target
target/*/logs/	#…every folder named logs two levels under a folder named target (* doesn’t include /)
target/**/logs/	#…every folder named logs somewhere under a folder named target (** includes /)
*.py[co]	    #…file or folder ending in .pyc or .pyo. However, it doesn’t match .py!
!README.md	    #Doesn’t ignore any README.md file even if it matches an exclude pattern, e.g. *.md.
                #NOTE This does not work if the file is located within a ignored folder.

#Examples for .ignore

# -----------example 1----- 

#Important Dot Files in Your Home Folder
# ignore everything ...
/*
# ... but the following
!/.profile
!/.bash_rc
!/.bash_profile
!/.curlrc

# .ignore for each sub-directory

# There are several locations where Git looks for ignore files. Besides looking in the root folder of a Git project, 
# Git also checks if there is a .gitignore in every subdirectory. This way you can ignore files on a finer grained 
# level if different folders need different rules.


#global ignore rules for your git account for every repository, This is especially useful for OS-specific files like .DS_Store on MacOS or thumbs.db on Windows.

git config --global core.excludesfile ~/.gitignore_global


#.git/info/exclude

# you can define repository specific rules which are not committed to the Git repository, 
# i.e. these are specific to your local copy. These rules go into the file .git/info/exclude which is created by default 
# in every Git repository with no entries.

# The advantage of .gitignore is that it can be checked into the repository itself, unlike .git/info/exclude. 
# Another advantage is that you can have multiple .gitignore files, one inside each directory/subdirectory for directory specific ignore rules, unlike .git/info/exclude.

# So, .gitignore is available across all clones of the repository. Therefore, in large teams all people are ignoring the same kind of files Example *.db, *.log. 
# And you can have more specific ignore rules because of multiple .gitignore.

# .git/info/exclude is available for individual clones only, hence what one person ignores in his clone is not available in some other person's clone. 
# For example, if someone uses Eclipse for development it may make sense for that developer to add .build folder to .git/info/exclude because other devs may not be using Eclipse.

# In general, files/ignore rules that have to be universally ignored should go in .gitignore, 
# and otherwise files that you want to ignore only on your local clone should go into .git/info/exclude


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#  git clone vs git copy
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# What is the difference between doing (after mkdir repo and cd repo):

git init
git remote add origin git://github.com/cmcculloh/repo.git
git fetch --all
git pull origin master

# and

git clone git://github.com/cmcculloh/repo.git

# # They're basically the same, except clone will setup additional remote tracking branches, not just master.

# # git clone is how you get a local copy of an existing repository to work on. It's usually only used 
# once for a given repository, unless you want to have multiple working copies of it around. 
# (Or want to get a clean copy after messing up your local one...)

# # git pull (or git fetch + git merge) is how you update that local copy with new commits from the
#  remote repository. If you are collaborating with others, it is a command that you will run
#   frequently.

# # As your first example shows, it is possible to emulate git clone with an assortment of other git 
# commands, but it's not really the case that git pull is doing "basically the same thing" as git
#  clone (or vice-versa).

# In laymen language we can say:

# Clone: Get a working copy of the remote repository.
# Pull: I am working on this, please get me the new changes that may be updated by others.
