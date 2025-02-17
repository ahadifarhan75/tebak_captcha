import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load gambar
img = cv2.imread('./gambar/429243.png')

# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gunakan thresholding untuk meningkatkan kontras
_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

# Hilangkan noise dengan operasi morfologi
kernel = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Simpan hasil pra-pemrosesan
cv2.imwrite('output.png', cleaned)

# Tampilkan hasil pra-pemrosesan menggunakan Matplotlib
plt.imshow(cleaned, cmap='gray')
plt.title('Hasil Pra-pemrosesan')
plt.axis('off')
plt.show()

# Ukuran gambar input
IMG_HEIGHT = 50
IMG_WIDTH = 200
CHANNELS = 1

# Pra-pemrosesan untuk Model
def preprocess_image(image):
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

# Label huruf yang mungkin muncul pada CAPTCHA
char_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

# Fungsi untuk mengonversi karakter ke angka
def char_to_num(char):
    return char_list.index(char)

# Fungsi untuk mengonversi angka ke karakter
def num_to_char(num):
    return char_list[num]

# Label untuk gambar ini (SESUAIKAN DENGAN CAPTCHA ASLI)
captcha_text = 'fJ6uGY'
labels = [char_to_num(c) for c in captcha_text]
labels = to_categorical(labels, num_classes=len(char_list))

# Pra-pemrosesan gambar
processed_img = preprocess_image(cleaned)
processed_img = np.expand_dims(processed_img, axis=0)

# Split data menjadi training dan testing (sementara pakai gambar yang sama)
X_train, X_test, y_train, y_test = train_test_split([processed_img], [labels], test_size=0.2)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Model Neural Network untuk OCR
inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Perbaiki Reshape: gunakan -1 agar otomatis menyesuaikan
x = layers.Reshape(target_shape=(-1, 256))(x)

# LSTM untuk urutan karakter
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.LSTM(128)(x)

outputs = layers.Dense(len(char_list), activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Latih model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Evaluasi model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Akurasi pada data testing:", test_acc)

# Prediksi
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
predicted_text = ''.join([num_to_char(num) for num in predicted_labels])

print("Hasil Prediksi:", predicted_text)
