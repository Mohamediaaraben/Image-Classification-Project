import tensorflow as tf
from data_preprocessing import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Charger les données
train_dir = "data/train"
test_dir = "data/test"
_, test_gen = load_and_preprocess_data(train_dir, test_dir)

# Charger le modèle
model = tf.keras.models.load_model("models/saved_model/image_classifier.h5")

# Évaluation
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy:.2f}")

# Prédictions et métriques
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

print(classification_report(y_true, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.savefig('results/confusion_matrix.png')
plt.show()
