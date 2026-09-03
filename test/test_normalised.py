from src.data_normalised import normalised
import numpy as np

def testing():
    X_train,X_cv,X_test,_,_,_ = normalised()
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_cv).any()
    assert not np.isnan(X_test).any()
