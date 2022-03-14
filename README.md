# ai2es-template
This repository is a template for all code repositories in the AI2ES 
organization. The purpose of this template is to ensure that all 
code repositories follow the same general structure and are similarly
laid out. That way new users can easily find files as needed.

## Getting Started
1. Setup a Python installation on the machine you are using. I
   recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) since
   it requires less memory than the full Anaconda Python distribution. Follow
   the instructions on the miniconda page to download and install Miniconda
   for your operating system.
2. Create a new repository within the [Github AI2ES organization](https://github.com/organizations/ai2es/repositories/new).
   Under the **Repository Template** section, select ai2es-template from the dropdown menu. Pick a name for
   your repository and decide whether to make it public or private. Then click
   **Create Repository** to create the repository site.
3. On the repository page, copy the appropriate repository path to your clipboard. The HTTPS option
   works for any setup but requires typing in your username and password whenever you want to pull or push from Github.
   If you set up a [ssh key with Github](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh),
   you can securely access the repo from an authenticated machine without having to type
   your username and password every time.
4. In a terminal window, clone your repository to your local machine with the command
`git clone <repo-address>`.
5. Within the terminal, go to the top-level directory of your repository with cd.
6. If you do not have a prior ai2es python environment within miniconda, create one from the environment.yml file.
   Use the command `conda env create -f environment.yml`. Activate the environment by running `conda activate ai2es` or 
   `source activate ai2es`.
7. If you already have an ai2es environment (check by typing `conda info --envs`), you can update
   the environment with changes to the environment.yml file by running `conda env update -f environment.yml`.
8. Change the name of the `template` directory to the name of your package. Your Python
   code will go there: `git mv template <packagename>`.
9. Modify `setup.py` to edit the project name, description, and any other keyword arguments for setup. 
   
