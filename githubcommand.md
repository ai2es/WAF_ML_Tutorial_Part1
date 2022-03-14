**Combining the Git Command Line with the Github Workflow**

It is generally good form to create a branch to make changes to files and code within a repository. The following recipe describes how to use git on the command line to create a new branch, use it, make the changes available on GitHub, and then clean up afterwards.  

*Prerequisite:*
Clone a repository from Github. The following line uses this repository, but you would replace that URL with your own repository:  
`git clone https://github.com/ai2es/<specific-repo.git>`

**Workflow**  
Along the way, it may be useful to type in: `git status`. It provides git's current status, and often provides helpful hints on what to do next.

1. Change the active directory to the location where you cloned the repository. Drill down the hierarchy by typing `cd ` and the name of the folder. Use `ls` to get a list of the files within that folder. 

1. From within the repository's directory on your local machine, create a branch:  
`git branch my_new_branch`

2. Checkout the branch:  
`git checkout my_new_branch`

  Tip: You can use `git checkout -b my_new_branch` to create a branch, if it does not already exist, and switch to it in one step.

  This is a good time for `git status` to ensure that you are working on the branch you intended.

3. Make changes inside the branch (e.g., add code, add files, whatever)

4. Stage files and commit changes:  
`git add whatever_file_you_have_changed`  
`git commit -m "the commit message"`  

5. Send your local branch to GitHub:  
`git push -u origin my_new_branch`

6. Go through the pull request process on Github. Be sure to delete the Github (remote) branch manually after your pull request happens.

7. Delete the local branch. The first line of code below checks out the master branch with the intent to keep things straight when you later synchronize the remote Github version. When you delete the local branch in the second line of code below, you might receive a warning message that your local changes have not been merged into master -- that is true. You will add the updated Github  master in the next step:  
`git checkout main`  
`git branch -d my_new_branch`  

8. Update your local repository with the latest updates from Github's version:  
`git pull`  
