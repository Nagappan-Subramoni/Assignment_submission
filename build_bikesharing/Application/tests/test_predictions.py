import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from bikeshare_model.processing.data_manager import read_input_data
from bikeshare_model.processing.data_manager import loadModelAndPredict
from bikeshare_model.processing.data_manager import getUserDataPreprocessed 
from bikeshare_model.processing.data_manager import fitAndSave
import bikeshare_model.processing.data_manager as dm

import bikeshare_model.pipeline as pp

def test_valLoadModelAndPredict():
    X,y=getUserDataPreprocessed() 
    y_pred=loadModelAndPredict(X)
    assert y - (10*y)/100 <= y_pred <= y +(10*y)/100 , "Predicted value outside the range"


test_valLoadModelAndPredict()