import pandas as pd

user = 'pagel_2194'
host = 'data.codeup.com'
password = '7b8GeH6zt3HL1Qw5VT1JZUcie6S31of5'

github_token = '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'
github_username = 'Divante222'

def get_db_url(db, user=user, password=password, host=host):
    return (f"mysql+pymysql://{user}:{password}@{host}/{db}")