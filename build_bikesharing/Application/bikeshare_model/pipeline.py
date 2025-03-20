#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.pipeline  import Pipeline
#from processing.features import ExtractMonthYear, WeathersitImputer, WeekdayImputer, OutlierHandler, WeekdayOneHotEncoder, Mapper
from sklearn.ensemble import RandomForestRegressor
from bikeshare_model.processing.features import  ExtractMonthYear, WeathersitImputer, WeekdayImputer, OutlierHandler, WeekdayOneHotEncoder, Mapper

piperline=Pipeline([
    ('Extract Month Year',ExtractMonthYear()),
    ('weekday_imputer',WeekdayImputer()),
    ('weathersit_imputer',WeathersitImputer()),
    ('mapper',Mapper()),
    ('outlier_handler',OutlierHandler()),
    ('weekday_one_hot_encoder',WeekdayOneHotEncoder()),
])

piperline_fit=Pipeline([('regressor',RandomForestRegressor())])

piperline_predict =  Pipeline([('regressor',RandomForestRegressor())])