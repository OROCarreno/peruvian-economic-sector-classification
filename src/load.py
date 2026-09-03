import pandas as pd 

df = pd.read_csv("data/raw/epen_2025.csv")

def get_columns():
    # Get only the colums we want as a feature of the neurons.
    columns = [
        "C208",       # age
        "C207",       # sex
        "AREA",       # urban
        "CCDD",       # department
        "C310",       # employment
        "C317",       # workplace size
        "C318_T",     # hours worked
        "INGTOT",     # income
        "C311",       # employer type
        "C312",       # SUNAT registration
        "C366",       # education
        "SEGURO1",    # health insurance
        "C306A",      # agricultural producer
        "C331",       # usual weekly hours
        "C313",       # accouting system
        "Informal_P", # informal status
        "C338",       # payment frequency 
        "C335",       # looking for job?
        "C330",       # usual_hours_flag
        "C333",       # wants_more_hours
        "C334",       # available_more_hours
        "P209H",      # underemployment_flag
        "whoraT",     # total_hours_worked
        "C328_T",     # secondary job hours
        "DIVISION_LIMA",#lima area
        "C359",      # worked before
        "C309_COD"   # economic activity
    ]


    df_col = df[columns].copy()

    df_col = df_col.rename(columns={
        "C208": "age",
        "C207": "sex",
        "AREA": "area",
        "CCDD": "department",
        "C310": "employment_type",
        "C317": "company_size",
        "C318_T": "hours_worked",
        "INGTOT": "income",
        "C311": "employer_type",
        "C312": "sunat_registration",
        "C366": "education",
        "SEGURO1": "health_insurance",
        "C306A": "agri_producer",
        "C331": "usual_weekly_hours",
        "C313": "accounting_system",
        "Informal_P": "informal_status",
        "C338": "payment_frequency",
        "C335": "looking_for_another_job",
        "C330":  "usual_hours_flag",
        "C333":  "wants_more_hours",
        "C334":  "available_more_hours",
        "P209H": "underemployment_flag",
        "whoraT": "total_hours_worked",
        "C328_T": "secondary_job_hours",
        "DIVISION_LIMA": "lima_area",
        "C359": "worked_before",
        "C309_COD": "economic_activity"

    })

    df_col = df_col.dropna(subset = ["economic_activity"])

    return df_col

