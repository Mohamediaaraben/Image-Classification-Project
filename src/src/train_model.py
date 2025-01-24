import tensorflow as tf
from data_preprocessing import load_and_preprocess_data

# Charger les données
train_dir = "data/train"
test_dir = "data/test"
train_gen, test_gen = load_and_preprocess_data(train_dir, test_dir)

# Construire le modèle CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(train_gen, epochs=10, validation_data=test_gen)

# Sauvegarder le modèle
model.save("models/saved_model/image_classifier.h5")
