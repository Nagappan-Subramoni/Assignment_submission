import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from processing import data_manager
import pipeline

X,y=data_manager.getUserDataPreprocessed()
y_pred=data_manager.loadModelAndPredict(X_test=X)
print(f'Predicted Value {y_pred}')