import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords



def basic_clean(the_string):
    the_string = the_string.lower()
    the_string = unicodedata.normalize('NFKD', the_string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8')
    the_string = re.sub(r"[^a-z0-9\s]", '', the_string)
    return the_string


def tokenize(the_string):
    tokenize = nltk.tokenize.ToktokTokenizer()
    the_string = tokenize.tokenize(the_string, return_str = True)
    return the_string


def stem(the_string):
    ps = nltk.porter.PorterStemmer()
    return ps.stem(the_string)


def lemmatize(the_string):
    wnl = nltk.stem.WordNetLemmatizer()
    return wnl.lemmatize(the_string)


def remove_stopwords(the_string, extra_words = [], exclude_words = []):
    stopwords_ls = stopwords.words('english')
    
    stopwords_ls = set(stopwords_ls) - set(exclude_words)
    stopwords_ls = list(stopwords_ls)
    
    if len(extra_words) > 1:
        stopwords_ls.extend(extra_words)
    elif len(extra_words) == 1:
        stopwords_ls.append(extra_words[0])
        
        
    words = the_string.split()
    filtered = [word for word in words if word not in stopwords_ls]
    return ' '.join(filtered)





