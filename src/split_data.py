from process_data import create_sectors

def split():
    clean_data = create_sectors()
    X = clean_data.drop(columns=[
    "sector",
    "economic_activity",
    "division"
    ])

    y = clean_data["sector"]

    return X,y

split()