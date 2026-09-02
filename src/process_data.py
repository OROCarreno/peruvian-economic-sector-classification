from load import get_columns

def create_sectors():
    data = get_columns()
    data["division"] = data["economic_activity"].astype(int)//100
    data["sector"] = data["division"].apply(get_sector)
    return data


def get_sector(division):

    if 1 <= division <= 3 or 5 <= division <= 9:
        return 0   # Primary sector

    elif 10 <= division <= 33:
        return 1   # Manufacturing

    elif 41 <= division <= 43:
        return 2   # Construction

    elif 45 <= division <= 47:
        return 3   # Commerce

    elif 35 <= division <= 39 or 49 <= division <= 99:
        return 4   # Services

    else:
        return None

