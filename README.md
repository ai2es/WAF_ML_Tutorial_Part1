# WAF Tutorial Part 1: Traditional ML 

## Introduction and Motivation
This repository is the code associated with the WAF manuscript titled: "A Machine Learning Tutorial for Operational Meteorology, Part I: Traditional Machine Learning" written by Chase, R. J., McGovern, A., Burke, A., Harrison, D. and Lackmann, G. under review. If you have any issues with bugs or other questions please leave an 'issue' associated with this repo.

The goal of this manuscript is to familiarize more meteorologists with the tools of ML and accelerate the use of ML in meteorological workflows. In order to accomplish these goals, it is imperative to supply the code and a sandbox for readers to play around with. 

## Background on data

![SEVIR Sample](https://github.com/MIT-AI-Accelerator/eie-sevir/blob/master/examples/tutorial_img/sevir_sample.gif)

The main dataset used herein is the [The Storm EVent ImagRy (SEVIR) dataset](https://proceedings.neurips.cc/paper/2020/file/fa78a16157fed00d7a80515818432169-Paper.pdf), which consists of over 10,000 matched storm events measured by satellite and radar images. The specific variables are: The red visible channel, the mid-tropospheric water vapor channel, the clean infrared channel, NEXRAD retrieved vertically integrated liquid and GLM measured lightning flashes. The SEVIR dataset github repo can be found [here](https://github.com/MIT-AI-Accelerator/eie-sevir) and a helpful notebook tutorial can be found [here](https://nbviewer.jupyter.org/github/MIT-AI-Accelerator/eie-sevir/blob/master/examples/SEVIR_Tutorial.ipynb). We thank the authors (Mark S. Veillette, Siddharth Samsi and Christopher J. Mattioli) of SEVIR for their efforts and creating a high-quality, open source meteorological dataset primed for machine learning. 

## Getting Started

There are two main ways to interact with the code here. 

### Use Google Colab 

   This is the reconmended and the quickest way to get started and only requires a (free) google account. Google Colab is a cloud instance of python that is run from your favorite web browser (although works best in Chrome). If you wish to use these notebooks, see the directory named colab_notebooks.

### Install python on your local machine and run notebooks there

   This is a bit more intense, especially for people who have never installed python on their machine. This method does allow you to always have the right packages installed and would enable you to actually download all of the SEVIR dataset if you want it (although it is very big... 924G total). 

   1. Setup a Python installation on the machine you are using. I
   recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) since
   it requires less memory than the full Anaconda Python distribution. Follow
   the instructions on the miniconda page to download and install Miniconda
   for your operating system. It is best to do these steps in a terminal (Mac/Linux) or powershell (Windows)

   Once you get it setup, it would be good to have python and jupyter in this base environment.

   ``` $ conda install -c conda-forge python,jupyter ``` 

   2. Now that conda is installed, clone this repository to your local machine with the command:

   ``` $ git clone https://github.com/ai2es/WAF_ML_Tutorial_Part1.git ``` 

   If you dont have git, you can install git ([Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)) or choose the "Download Zip" option, unzip it and then continue with these steps. 

   3. Change into the newly downloaded directory 

   ``` $ cd WAF_ML_Tutroial_Part1.git ``` 

   4. It is good practice to always make a new env for each project you work on. So here we will make a new environment  

   ``` $ conda env create -f environment.yml ``` 

   5. Activate the new environment 

   ``` $ conda activate waf_tutorial_part1 ``` 

   6. Add this new environement to a kernel in jupyter 

   ```$ python -m ipykernel install --user --name waf_tutorial_part1 --display-name "waf_tutorial_part1" ```

   7. Go back to the base environment 

   ```$ conda deactivate ``` 

   8. Start jupyter

   ``` $ jupyter lab ``` 

   9. You should be able to open the notebooks with this repository and you should be able to add the kernel we just installed. 
