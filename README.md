# Classifying Tumor from RNA Microarray using Machine Learning
## Setup the environment
Install pipenv and run pipenv install in pipenv shell
```
pipenv install
```

## Getting Started 
To download the data set
```
bash download_dataset.sh
```
To perform data exploration and generate a 2D visualization, run in command line
```
python data_exploration.py
```
To train machine learning models and parameter tuning, run in command line
```
python model_training.py
``` 
To perform model evaluation, run in command line
```
python model_evaluation.py
``` 

To perform the entire workflow, run in command line
```
main.py
``` 

## Helpful links 
- [[https://cloud.google.com/ml-engine/docs/tensorflow/ml-solutions-overview][Google ML Tutorial]]
## Think about the problem 
- What is the problem that you are trying to solve?
- Is the problem well defined?
- How can you evaluate the outcome of the project? 
- Is machine learning the best solution? 
  - Acess to a sizable set of data
  - Each additional feature requires addtional samples to train model properly  
  - There is no better alternatives

## Measure model success
Knowing when to stop refining the model, and put it into production. 


## Stages of Machine Learning Workflow
### Data Import 
- import csv files, load features and outcomes into dataframes 
- split features and outcomes into train and test dataset
  
### Data Exploration and Feature Engineering
- get the dimension of the data
- if data is high dimensional, use dimension reduction to visualize
- identify features in your data, which is subset of data attributes in your raw data that you use in your model
- clean the data by finding errors or anomalities 

### Data preprossing
- Normalizing numeric data into common scale
- Applying formatting rules to data
- Reducing data redundancy through simplification, eg. converting a text feature into bag of words representation
- Representing text numerically, as when assigning values to each possible value in a categorical feature
- Assigning key values to data instances

### Training Set Creation
- split the data into training and testing set  

### Machine Learning Algorithm
- Cross validation
- Parameter tuning using random search or grid search
- Select and test the models 

### Feature Selection
- Goal: find the most important ten genes associated with each cancer type
- Methods
  1. use SVM to select out the most important feature iteratively, to generate sparsity
  2. use RF to find the most important feature 
- visualization 
  - create a model performance visualization as a function of increasing sparsity 
