from process_data import create_sectors
from sklearn.model_selection import train_test_split

def split():
    clean_data = create_sectors()
    X = clean_data.drop(columns=[
    "sector",
    "economic_activity",
    "division"
    ])
    y = clean_data["sector"]

    ## we use temp because train_test_split can only split in 2 groups.
    X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
    )
    # test =70%, cv = 15% and test = 15%
    X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
    )

    return X_train,X_cv,X_test,y_train,y_cv,y_test

split()