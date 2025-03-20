from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import yaml
from bikeshare_model import CONFIG_YML
config=None
with open(CONFIG_YML, "r") as file:
    config = yaml.safe_load(file)

class ExtractMonthYear(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None,dteday='dteday',weekday='weekday'):
        return self 

    def transform(self, df,y=None,dteday='dteday',weekday='weekday'):
        df = df.copy()  
        df['dteday']=pd.to_datetime(df['dteday'])
        df["month"] = df["dteday"].dt.month
        df["year"] = df["dteday"].dt.year
        df=df.drop(['registered','casual'],axis = 1)
        return df

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self,):
        # YOUR CODE HERE
        self.day_names = None

    def fit(self,df,y=None,dteday='dteday',weekday='weekday'):
        # YOUR CODE HERE
        return self

    def transform(self,df,y=None,dteday='dteday',weekday='weekday'):
        # YOUR CODE HERE
        df=df.copy()
        day_names = df[dteday].dt.day_name()[:3]
        df[weekday] = df[weekday].fillna(day_names)
        df = df.drop(columns=[dteday])
        return df

class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self,):
        # YOUR CODE HERE
        self.most_frequent_category = None

    def fit(self,df,y=None,weathersit='weathersit'):
        # YOUR CODE HERE
        return self

    def transform(self,df,y=None,weathersit='weathersit'):
        # YOUR CODE HERE
        df=df.copy()
        self.most_frequent_category = df[weathersit].mode()[0]
        df[weathersit] = df[weathersit].fillna(self.most_frequent_category)
        return df


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self,):
        # YOUR CODE HERE
        self.cat_cols = None
        self.mappings={}

    def fit(self,df,y=None):
        # YOUR CODE HERE

        return self

    def transform(self,df,y=None):
        # YOUR CODE HERE
        df=df.copy()            
        df['season']=df['season'].map(config['season'])
        df['hr'] = df['hr'].map(config['hr'])
        df['weathersit']=df['weathersit'].map(config['weathersit'])
        df['holiday']=df['holiday'].map(config['holiday'])
        df['workingday']=df['workingday'].map(config['workingday'])

        return df


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self,method='iqr', factor=1.5):
        # YOUR CODE HERE
        self.method = method
        self.factor = factor

    def fit(self,df,y=None):
        # YOUR CODE HERE
        return self

    def transform(self,df,y=None):
        # YOUR CODE HERE
        df=df.copy()
        self.columns= df.select_dtypes(include=['int64', 'float64']).columns

        for col in self.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR
            
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        return df


    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self,):
        # YOUR CODE HERE
        self.encoder = None

    def fit(self,df,y=None):
        # YOUR CODE HERE
        return self

    def transform(self, df):
        # YOUR CODE HERE
        df=df.copy()
        weekday=config['weekday']
        for idx,day in enumerate(weekday):
            df['weekday'+'_'+str(idx+1)] = np.where(df['weekday'] == day,1,0)
        
        df = df.drop(columns=['weekday'])

        return df
