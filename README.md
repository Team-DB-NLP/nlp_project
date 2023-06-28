# team_db nlp_project
nlp project for codeup.

## Project Description

* Gathering repositories from GitHub on healthcare, we are creating a model to predict the programming language based on the content of the README file. 

## Project Goals

* Acquire repositories and clean, tokenize, stem, lemmatize, and apply stopwords to the README content.

* Explore the words and statistics of the README files to build visuals for insights into modeling.

* Produce a viable model to predict programming languages based on content of README file.

* Other key drivers:
    * type of 'language'
    * content of 'readme_contents'
    
## Initial Thoughts

* There appears to be key words within the content that will predict the programming language and drive a better predictor than baseline.

## The Plan

* Acquire and join GitHub repositories containing healthcare content.

* Prepare the data using the following columns:
    * target: 
    * features:
        * repo
        * language
        * readme_contents
        * clean
        * stemmed
        * lemmatized

* Explore dataset for predictors of property value ('__')
    * Answer the following questions:
    * Do different programming languages (PL) use a different number of unique words?
    * Which bigram words occur the most in each PL?
    * Which trigram words occur the most in each PL?
    * What is the percentage of each PL for the most common ('all') 20 words?

* Develop a model
    * Using the selected data features develop appropriate predictive models
    * Evaluate the models in action using train and validate splits
    * Choose the most accurate model 
    * Evaluate the most accurate model using the final test data set
    * Draw conclusions

## Data Dictionary
| Feature         | Definition                                                               |
|-----------------|--------------------------------------------------------------------------|
| repo            | name of the repository in GitHub                                         |
| language        | the type of programming language (Python, Java, JavaScript, HTML, other) |
| README_contents | the words contained in the README file for that repository               |
| clean           | words after being coded, lower cased, tokenized, and regexed             |
| stemmed         | words after applying stem or root word                                   |
| lemmatized      | words after applying stem or common word findings                        |

## Steps to Reproduce
1) Clone the repo git@github.com:Team-DB-NLP/nlp_project.git in terminal
2) Download [giant_df.csv](https://drive.google.com/file/d/1ZG1MSOnoF0gER2WHeob_LsuPr5XhnivZ/view?usp=share_link) from public Google Drive and put in personal repo local directory 
3) Run notebook

## Takeaways and Conclusions
Models used:
* 

* 
* 
* 
* 


## Recommendations
* 
* 
* 
* 

## Next Steps

*
