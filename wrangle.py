import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
import os
import time
import prepare
import json
import unicodedata
import pandas as pd
import numpy as np
import seaborn as sns
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


def get_bows(X_train, X_validate, X_test):
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train)
    X_validate_bow = cv.transform(X_validate)
    X_test_bow = cv.transform(X_test)
    return X_bow, X_validate_bow, X_test_bow

    
def get_train(new_df):
    train, validate, test = split_data(new_df, 'language')

    X_train = train['lemmatized']
    y_train = train.language
    X_validate = validate['lemmatized']
    y_validate = validate.language
    X_test = test['lemmatized']
    y_test = test.language
    X_bow, X_validate_bow, X_test_bow = get_bows(X_train, X_validate, X_test)
    the_df = super_classification_model(new_df, X_bow,y_train, X_validate_bow, y_validate)

    # (new_df, X_train,y_train, X_validate, y_validate, the_c = 1, neighbors = 20)

    return the_df, X_test_bow, y_test
    

def get_dataframe():
    new_df = pd.read_csv('giant_df')
    new_df = prepare_data(new_df)
    new_df = do_everything(new_df)
    
    return new_df


def drop_http_words(text):
    words = text.split()
    filtered_words = [word for word in words if 'http' not in word]
    return ' '.join(filtered_words)


def new_column_counts(the_string):
        return len(the_string)


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
    codeup_df['lem_len'] = codeup_df.lemmatized.apply(new_column_counts)
    return codeup_df


def super_classification_model(new_df, X_train,y_train, X_validate, y_validate, the_c = 1, neighbors = 20):
    '''
    Runs classification models based on our best parameters and returns a pandas dataframe
    '''
    baseline = len(new_df[new_df.language == 'other']) / len(new_df.index)
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


def chi2_test(train, columns_list):
    '''
    Runs a chi2 test on all items in a list of lists and returns a pandas dataframe
    '''
    chi_df = pd.DataFrame({'feature': [],
                    'chi2': [],
                    'p': [],
                    'degf':[],
                    'expected':[]})
    
    for iteration, col in enumerate(columns_list):
        
        observed = pd.crosstab(train[col[0]], train[col[1]])
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        chi_df.loc[iteration+1] = [col, chi2, p, degf, expected]

    return chi_df


def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english') 
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words


def all_words_function(new_df):
    df = new_df
    other_words = clean(' '.join(df[df.language=="other"]['lemmatized']))
    Python_words = clean(' '.join(df[df.language=="Python"]['lemmatized']))
    Java_words = clean(' '.join(df[df.language=="Java"]['lemmatized']))
    JavaScript_words = clean(' '.join(df[df.language=="JavaScript"]['lemmatized']))
    HTML_words = clean(' '.join(df[df.language=="HTML"]['lemmatized']))






    other_words_freq = pd.Series(other_words).value_counts()
    Python_words_freq = pd.Series(Python_words).value_counts()
    Java_words_freq = pd.Series(Java_words).value_counts()
    JavaScript_words_freq = pd.Series(JavaScript_words).value_counts()
    HTML_words_freq = pd.Series(HTML_words).value_counts()


    other_words_df = pd.DataFrame(other_words_freq)
    Python_words_df = pd.DataFrame(Python_words_freq)
    Java_words_df = pd.DataFrame(Java_words_freq)
    JavaScript_words_df = pd.DataFrame(JavaScript_words_freq)
    HTML_words_df = pd.DataFrame(HTML_words_freq)

    all_words = clean(' '.join(df['lemmatized']))
    all_freq = pd.Series(all_words).value_counts()
    all_words_df = pd.DataFrame(all_freq)

    word_counts =pd.concat([other_words_freq,Python_words_freq, Java_words_freq, JavaScript_words_freq,HTML_words_freq, all_freq], axis =1).fillna(0).astype(int)

    
    word_counts.columns = ['other', 'Python', 'Java', 'JavaScript', 'HTML', 'all']
    return HTML_words_df, JavaScript_words_df, Java_words_df, Python_words_df, other_words_df, other_words, Python_words, Java_words, JavaScript_words, HTML_words, other_words_freq, Python_words_freq, Java_words_freq, JavaScript_words_freq, HTML_words_freq, word_counts, all_words, all_freq, all_words_df


def visual_one(Python_words_df,other_words_df, Java_words_df, JavaScript_words_df, all_words_df, HTML_words_df):
    one = pd.DataFrame([len(Python_words_df)])
    two = pd.DataFrame([len(other_words_df)])
    three = pd.DataFrame([len(Java_words_df)])
    four = pd.DataFrame([len(JavaScript_words_df)])
    five = pd.DataFrame([len(all_words_df)])

    six = pd.DataFrame([len(HTML_words_df)])

    master_numbers = pd.concat([one,two,three,four, six, five], axis=1)
    master_numbers.columns = ['Python','other',  'Java', 'JavaScript', 'HTML','all',]
    sns.barplot(data = master_numbers)
    plt.title('Do different programming languages use a different number of unique words?')
    plt.show()


def visual_two(word_counts):
    (word_counts.sort_values('all', ascending=False)
    .head(20)
    .apply(lambda row: row/row['all'], axis=1)
    .sort_values(by='all')
    .drop(columns='all')

    .plot.barh(stacked=True, width=1, ec='black')
    )


    plt.title('% of programming language for the most common 20 words')
    plt.legend(bbox_to_anchor=(1.2, 1.0),loc='upper right')
    plt.show()

def get_stats_test(train):
    answer = stats.f_oneway(train['lem_len'][train['language'] == 'other'], 
                     train['lem_len'][train['language'] == 'Python'], 
                     train['lem_len'][train['language'] == 'Java'],
                    train['lem_len'][train['language'] == 'JavaScript'],
                    train['lem_len'][train['language'] == 'HTML'])
    
    return answer