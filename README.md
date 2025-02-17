# Tebak Captcha Cerdas dengan Kecerdasan Buatan

Proyek ini menggunakan **Python**, **OpenCV**, dan **TensorFlow/Keras** untuk **mendeteksi dan mengenali teks pada gambar Captcha** secara otomatis dengan **kecerdasan buatan**.  
Ini adalah implementasi **OCR (Optical Character Recognition)** yang cerdas untuk menebak Captcha dengan **pra-pemrosesan gambar** dan **model deep learning**.  

---

## 🎯 **Fitur Utama:**  
- **Pra-pemrosesan Gambar:** Konversi ke grayscale, thresholding, dan penghilangan noise.  
- **Model Deep Learning:** Menggunakan **Convolutional Neural Network (CNN)** untuk mengenali karakter Captcha.  
- **OCR dengan Tesseract:** Memanfaatkan Tesseract untuk mengenali teks dari gambar.  

---

## 🛠️ **Persyaratan Sistem:**  
- **Python**: 3.8 - 3.11  
- **TensorFlow**: 2.x  
- **OpenCV**: Untuk pemrosesan gambar.  
- **Pytesseract**: Untuk OCR.  

---

## 📦 **Instalasi:**  
1. **Pastikan PIP sudah terbaru:**  
    ```bash
    python -m pip install --upgrade pip
    ```  

2. **Install dependensi yang dibutuhkan:**  
    ```bash
    pip install tensorflow keras opencv-python-headless numpy matplotlib pytesseract
    ```  

3. **Jika mengalami kendala pada TensorFlow**, coba dengan:  
    ```bash
    pip install tensorflow==2.13.0
    ```

4. **Pastikan Tesseract sudah terpasang** di sistem operasi:  
   - **Windows**: Unduh dan install dari [https://github.com/ub-mannheim/tesseract/wiki](https://github.com/ub-mannheim/tesseract/wiki)  
   - Tambahkan **path Tesseract** di environment variable, misalnya:  
     ```
     C:\Program Files\Tesseract-OCR
     ```

---

## 🚀 **Cara Menggunakan:**  
1. Masukkan gambar Captcha yang ingin dipecahkan ke dalam folder `./gambar`.  
2. Jalankan skrip:  
    ```bash
    python cerdas.py
    ```  
3. **Hasil OCR** akan ditampilkan di terminal.

---

## 📁 **Struktur Folder:**  
```
tebak-captcha-cerdas/
│   gambar.py        # Skrip utama untuk membaca dan menebak captcha
│   README.md        # Dokumentasi proyek
└───gambar/
        contoh_captcha1.png
        contoh_captcha2.png
```

---

## ⚙️ **Konfigurasi Tambahan:**  
Jika ingin mengubah parameter threshold atau model, sesuaikan pada bagian berikut di `gambar.py`:  
```python
_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
```

---

## 💡 **Catatan:**  
- **Pastikan versi Python kompatibel dengan TensorFlow yang digunakan.**  
- Hasil **OCR** dapat berbeda tergantung pada kualitas gambar Captcha.  

---

## 👨‍💻 **Kontribusi:**  
Pull request dan kontribusi sangat **disambut dengan baik**. Jika menemukan bug atau memiliki saran fitur baru, silakan buat **Issue**.  

---

## 📄 **Lisensi:**  
Proyek ini dilisensikan di bawah **MIT License**. Lihat file **LICENSE** untuk detail lebih lanjut.  




