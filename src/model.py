import tensorflow as tf
import numpy as np
import pandas as pd
from split_data import split
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report


X,y = split()
## we use temp because train_test_split can only split in 2 groups.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)
# test =70%, cv = 15% and test = 15%


# classes = np.unique(y_train)

# weights = compute_class_weight(
#     class_weight="balanced",
#     classes=classes,
#     y=y_train
# )
# class_weights = dict(zip(classes, weights))
class_weights = {
    0: 1.0,
    1: 1.9,
    2: 1.5,
    3: 1.1,
    4: 0.9
}

#Lets normalised the data, but onlt the numerical features
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

X_train_num = norm_l(X_train[numerical_data].to_numpy())
X_cv_num = norm_l(X_cv[numerical_data].to_numpy())
X_test_num = norm_l(X_test[numerical_data].to_numpy())

# get rid of the numerical data so we can combine it.
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
X_train_final = np.concatenate(
    [X_train_num.numpy(), X_train_cat],
    axis=1
)

X_cv_final = np.concatenate(
    [X_cv_num.numpy(), X_cv_cat],
    axis=1
)

X_test_final = np.concatenate(
    [X_test_num.numpy(), X_test_cat],
    axis=1
)



##Creating the neural model

#biggest neuron network cv accuracy:65.61
tf.random.set_seed(12345) #same result
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu",name ="l1"),
    tf.keras.layers.Dense(256, activation="relu",name ="l2"),
    tf.keras.layers.Dense(128, activation="relu",name ="l3"),
    tf.keras.layers.Dense(64, activation="relu",name ="l4"),
    tf.keras.layers.Dense(32, activation="relu",name ="l5"),
    tf.keras.layers.Dense(5, activation="linear",name="l6")
])

# big neuron network cv accuracy:65.39
# tf.random.set_seed(12345) #same result
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation="relu",name ="l1"),
#     tf.keras.layers.Dense(64, activation="relu",name ="l2"),
#     tf.keras.layers.Dense(32, activation="relu",name ="l3"),
#     tf.keras.layers.Dense(5, activation="linear",name="l4")
# ])

# mid neuron network cv accuracy:65.07
# tf.random.set_seed(12345) #same result
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation="relu",name ="l1"),
#     tf.keras.layers.Dense(32, activation="relu",name ="l2"),
#     tf.keras.layers.Dense(5, activation="linear",name="l3")

# ])

# small neuron network cv accuracy:65.69
# tf.random.set_seed(12345) #same result
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation="relu",name ="l1"),
#     tf.keras.layers.Dense(5, activation="linear",name="l2")
# ])


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.0003),
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train_final,y_train,
    epochs=20,
    class_weight=class_weights,
    batch_size=64,
    validation_data=(X_cv_final, y_cv),
    callbacks=[early_stop]
)

test_loss, test_accuracy = model.evaluate(
    X_test_final,
    y_test
)
print(f"loss {test_loss},accuracy{test_accuracy*100:.2f}")
test_logits = model.predict(X_cv_final, verbose=0)
test_predictions = np.argmax(test_logits, axis=1)
print(classification_report(y_cv, test_predictions))  

