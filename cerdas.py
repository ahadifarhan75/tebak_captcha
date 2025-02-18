import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import os

# Ukuran gambar input
IMG_HEIGHT = 50
IMG_WIDTH = 200
CHANNELS = 1

# Label huruf yang mungkin muncul pada CAPTCHA
char_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

# Kernel untuk operasi morfologi
kernel = np.ones((2, 2), np.uint8)

# Pra-pemrosesan untuk Model
def preprocess_image(image):
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

# Fungsi untuk mengonversi karakter ke angka
def char_to_num(char):
    return char_list.index(char)

# Fungsi untuk mengonversi angka ke karakter
def num_to_char(num):
    return char_list[num]

images = []
labels = []

# Loop untuk membaca semua gambar dalam folder
for filename in os.listdir('./gambar'):
    if filename.endswith('.png'):
        img_path = os.path.join('./gambar', filename)
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed_img = preprocess_image(cleaned)
            images.append(processed_img)
            
            # Ambil captcha_text dari nama file
            captcha_text = filename.split('.')[0]
            labels.append([char_to_num(c) for c in captcha_text])

# Konversi ke array numpy
X = np.array(images)
y = np.array(labels)

# Split data menjadi training dan testing terlebih dahulu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Konversi labels menjadi one-hot encoding dengan shape (jumlah_data, panjang_captcha, jumlah_karakter)
y_train = np.array([to_categorical(label, num_classes=len(char_list)) for label in y_train])
y_test = np.array([to_categorical(label, num_classes=len(char_list)) for label in y_test])

# Model Neural Network untuk OCR
inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Flatten dan Dense
# Gunakan Flatten sebelum Dense
x = layers.Flatten()(x)
x = layers.Dense(6 * 256, activation='relu')(x)  # Sesuaikan agar jumlah elemen konsisten
x = layers.Reshape(target_shape=(6, 256))(x)  # 6: panjang maksimal captcha


# Ubah Reshape agar output sesuai dengan urutan karakter dalam CAPTCHA
x = layers.Reshape(target_shape=(6, 256))(x)  # 6: panjang maksimal captcha
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.LSTM(128, return_sequences=True)(x)

# Gunakan TimeDistributed untuk setiap karakter dalam CAPTCHA
outputs = layers.TimeDistributed(layers.Dense(len(char_list), activation='softmax'))(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Callback untuk early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Latih model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=[early_stopping])

# Evaluasi model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Akurasi pada data testing:", test_acc)

# Prediksi
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=2)
predicted_texts = [''.join([num_to_char(num) for num in pred]) for pred in predicted_labels]

print("Hasil Prediksi:", predicted_texts)
