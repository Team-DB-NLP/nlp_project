import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
import os
import time
import prepare
import json

import pandas as pd
import numpy as np

#see the data
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#play with words
import nltk
import re
from pprint import pprint

#split and model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.model_selection import train_test_split

def split_data(df, target):
    '''
    Takes in the titanic dataframe and return train, validate, test subset dataframes
    '''
    
    
    train, test = train_test_split(df,
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[target]
                                   )
    train, validate = train_test_split(train, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[target]
                                       )
    
    return train, validate, test


def get_dataframe():
    new_df = pd.read_csv('giant_df')
    new_df = prepare_data(new_df)
    new_df = do_everything(new_df)
    
    return new_df


def drop_http_words(text):
    words = text.split()
    filtered_words = [word for word in words if 'http' not in word]
    return ' '.join(filtered_words)


def prepare_data(new_df):
    # Drop specific rows from the DataFrame
    new_df = new_df.drop([4, 26, 86, 143])
    # Reset the index of the DataFrame
    new_df = new_df.reset_index(drop=True)
    new_df.drop_duplicates(subset='readme_contents', inplace=True)
    new_df = new_df.dropna()
    new_df = new_df[new_df.language != 'Jupyter Notebook']
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop('Unnamed: 0', axis=1)

    the_list = new_df.language.value_counts()[new_df.language.value_counts() < 7].index.tolist()
    the_dict = {}
    for i in the_list:
        the_dict[i] = 'other'
        

    new_df['language'] = new_df['language'].replace(the_dict)
    new_df['readme_contents'] = new_df['readme_contents'].apply(drop_http_words)
    return new_df


def do_everything(codeup_df):
    cleaned_original = codeup_df['readme_contents'].apply(prepare.basic_clean)
    codeup_df['clean'] = cleaned_original
    tokenized_original = codeup_df['clean'].apply(prepare.tokenize)
    codeup_df['clean'] = tokenized_original

    codeup_df['clean'] = codeup_df['clean'].apply(prepare.remove_stopwords, extra_words = [], exclude_words = [])



    codeup_df['stemmed'] = codeup_df['clean'].apply(prepare.stem)
    codeup_df['lemmatized'] = codeup_df['clean'].apply(prepare.lemmatize)
    return codeup_df


def super_classification_model(new_df, X_train,y_train, X_validate, y_validate, the_c = 1, neighbors = 20):
    '''
    Runs classification models based on our best parameters and returns a pandas dataframe
    '''
    baseline = len(new_df[new_df.language == 'other']) / len(new_df[new_df.language == 'other']) + len(new_df[new_df.language != 'other'])
    the_df = pd.DataFrame(data=[
    {
        'model_train':'baseline',
        'train_predict':round(baseline,3),
        'validate_predict':round(baseline,3)
    }
    ])

    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train, y_train)
    train_predict = knn.score(X_train, y_train)
    validate_predict = knn.score(X_validate, y_validate)
    knn, train_predict, validate_predict
    the_df.loc[1] = ['KNeighborsClassifier', round(train_predict, 3), round(validate_predict, 3)]

    logit = LogisticRegression(random_state= 123,C=the_c)
    logit.fit(X_train, y_train)
    train_predict = logit.score(X_train, y_train)
    validate_predict = logit.score(X_validate, y_validate)
    the_df.loc[2] = ['LogisticRegression', round(train_predict, 3), round(validate_predict, 3)]


    forest = RandomForestClassifier(random_state = 123, max_depth=5)
    forest.fit(X_train, y_train)    
    train_predict = forest.score(X_train, y_train)
    validate_predict = forest.score(X_validate, y_validate)
    the_df.loc[3] = ['RandomForestClassifier', round(train_predict, 3), round(validate_predict, 3)]    


    tree = DecisionTreeClassifier(random_state = 123,max_depth=6)
    tree.fit(X_train, y_train)
    train_predict = tree.score(X_train, y_train)
    validate_predict = tree.score(X_validate, y_validate)
    the_df.loc[4] = ['DecisionTreeClassifier', str(round(train_predict, 3))[:5], str(round(validate_predict, 3))[:5]]    

    return the_df


def get_bows(X_train, X_validate, X_test):
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train)
    X_validate_bow = cv.transform(X_validate)
    X_test_bow = cv.transform(X_test)
    return X_bow, X_validate_bow, X_test_bow