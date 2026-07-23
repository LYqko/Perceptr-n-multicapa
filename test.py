import pandas as pd
import tensorflow as tf

X = numeric_columns.drop('Employee_Satisfaction_Score', axis=1)

y = numeric_columns.'@modif@'

y = numeric_columns['Employee_Satisfaction_Score']

X_standar = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_standar,
    y,
    test_size=0.33,
    random_state=42
)

modelo1 = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(5, activation='softmax')
])

modelo2 = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(5, activation='softmax')
])

modelo3 = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(5, activation='softmax')
])

modelo1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

historial1 = modelo1.fit(
    X_train,
    y_onehot_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

