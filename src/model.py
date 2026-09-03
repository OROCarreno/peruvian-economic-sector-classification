from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from src.data_normalised import normalised
import tensorflow as tf

X_train,X_cv,X_test,y_train,y_cv,y_test = normalised()

# classes = np.unique(y_train)

# weights = compute_class_weight(
#     class_weight="balanced",
#     classes=classes,
#     y=y_train
# )
# class_weights = dict(zip(classes, weights))
# I change the weights based on the clasification report to get better results
class_weights = {
    0: 1.0,
    1: 1.9,
    2: 1.5,
    3: 1.1,
    4: 0.9
}

##Creating the neural model


def biggest():
    tf.random.set_seed(12345) #same result
    name = "biggest"
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu",name ="l1"),
        tf.keras.layers.Dense(256, activation="relu",name ="l2"),
        tf.keras.layers.Dense(128, activation="relu",name ="l3"),
        tf.keras.layers.Dense(64, activation="relu",name ="l4"),
        tf.keras.layers.Dense(32, activation="relu",name ="l5"),
        tf.keras.layers.Dense(5, activation="linear",name="l6")
    ])
    call(model,name)

def mid_big():
    tf.random.set_seed(12345) #same result
    name = "mid big"
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu",name ="l1"),
        tf.keras.layers.Dense(64, activation="relu",name ="l2"),
        tf.keras.layers.Dense(32, activation="relu",name ="l3"),
        tf.keras.layers.Dense(5, activation="linear",name="l4")
    ]
    )
    call(model,name)

   


def mid_small():
    tf.random.set_seed(12345) #same result
    name = "mid small"
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu",name ="l1"),
        tf.keras.layers.Dense(32, activation="relu",name ="l2"),
        tf.keras.layers.Dense(5, activation="linear",name="l3")
    ]
    )
    call(model,name)

    

def small():
    tf.random.set_seed(12345) #same result
    name = "small"
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu",name ="l1"),
        tf.keras.layers.Dense(5, activation="linear",name="l2")
    ]
    )
    call(model,name)


def call(model_size,name):
    model_size.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0003),
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model_size.fit(
        X_train,y_train,
        epochs=20,
        class_weight=class_weights,
        batch_size=64,
        validation_data=(X_cv, y_cv),
        callbacks=[early_stop],
        verbose = 0,
    )
    test_loss, test_accuracy = model_size.evaluate(
    X_test,
    y_test
    )
    
    return f"{name} neural size has loss of:{test_loss}, and accuracy of: {test_accuracy*100:.2f}"
    #to check the clasification report 
    # test_logits = model_size.predict(X_cv, verbose=0)
    # test_predictions = np.argmax(test_logits, axis=1)
    # print(classification_report(y_cv, test_predictions))  


biggest()
mid_big()
mid_small()
small()

