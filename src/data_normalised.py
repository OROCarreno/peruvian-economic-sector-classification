from split_data import split
import tensorflow as tf
import numpy as np
import pandas as pd

def normalised():
    X_train,X_cv,X_test,y_train,y_cv,y_test = split()
    # we only want to normalised numerical data 
    numerical_data =[
        "age",
        "hours_worked",
        "income",
        "usual_weekly_hours",
        "total_hours_worked",
        "secondary_job_hours"
    ] 
    categorical_data = [
        "sex",
        "area",
        "department",
        "employment_type",
        "company_size",
        "employer_type",          
        "sunat_registration",     
        "education",              
        "health_insurance",       
        "agri_producer",
        "accounting_system",
        "informal_status",
        "payment_frequency",
        "looking_for_another_job",
        "usual_hours_flag",
        "wants_more_hours",
        "available_more_hours",
        "underemployment_flag",
        "lima_area",
        "worked_before"
    ]

    #Fill the nan with median 
    train_medians = X_train[numerical_data].median()

    X_train[numerical_data] = X_train[numerical_data].fillna(train_medians)
    X_cv[numerical_data] = X_cv[numerical_data].fillna(train_medians)
    X_test[numerical_data] = X_test[numerical_data].fillna(train_medians)

    #Filling with mode since its not numerical
    train_modes = X_train[categorical_data].mode().iloc[0]

    X_train[categorical_data] = X_train[categorical_data].fillna(train_modes)
    X_cv[categorical_data] = X_cv[categorical_data].fillna(train_modes)
    X_test[categorical_data] = X_test[categorical_data].fillna(train_modes)


    #Normalisation.
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X_train[numerical_data].to_numpy())

    X_train_norm = norm_l(X_train[numerical_data].to_numpy())
    X_cv_norm = norm_l(X_cv[numerical_data].to_numpy())
    X_test_norm = norm_l(X_test[numerical_data].to_numpy())

    X_train = X_train.drop(columns=numerical_data)
    X_cv = X_cv.drop(columns=numerical_data)
    X_test = X_test.drop(columns=numerical_data)
    
    #changing categorical values to one hot
    X_train_cat = pd.get_dummies(
        X_train[categorical_data].astype(str),
        dtype=float
    )

    X_cv_cat = pd.get_dummies(
        X_cv[categorical_data].astype(str),
        dtype=float
    )

    X_test_cat = pd.get_dummies(
        X_test[categorical_data].astype(str),
        dtype=float
    )

    #making sure they ahve the same colums
    X_cv_cat = X_cv_cat.reindex(
        columns=X_train_cat.columns,
        fill_value=0
    )

    X_test_cat = X_test_cat.reindex(
        columns=X_train_cat.columns,
        fill_value=0
    )

    #concatenating the categorical with numerals
    X_train = np.concatenate(
        [X_train_norm.numpy(), X_train_cat],
        axis=1
    )

    X_cv = np.concatenate(
        [X_cv_norm.numpy(), X_cv_cat],
        axis=1
    )

    X_test = np.concatenate(
        [X_test_norm.numpy(), X_test_cat],
        axis=1
    )

    return X_train,X_cv,X_test,y_train,y_cv,y_test


    