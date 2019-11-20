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
git push -u origin master
git pull

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

git merge branch_name
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

