# Mental Health Text Classification using Deep Learning

Di project ini, kita membandingkan dua arsitektur utama: **LSTM** (sebagai *baseline*) dan **MentalBERT** (model transformer). Tujuannya adalah untuk melihat model mana yang lebih akurat dalam mendeteksi 7 kondisi psikologis dari data teks media sosial.

Selain komparasi model, project ini juga berfokus pada cara terbaik menangani dataset yang sangat tidak seimbang (*extreme class imbalance*) tanpa harus membuang data.

---

## Dataset Info
Dataset yang dipakai adalah **"Sentiment Analysis for Mental Health"** dari Kaggle.
- **Total Data:** ~53.000 baris teks (*statement*).
- **Label Target:** Terdapat 7 kategori kondisi mental (*Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder*).
- **Link Dataset:** [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)


## Fitur & Highlight Project
Beberapa teknik utama yang diimplementasikan di project ini:
1. **Adaptive Preprocessing:** Pembersihan teks dirancang berbeda untuk tiap model. LSTM butuh teks yang sangat bersih (tanpa tanda baca), sedangkan MentalBERT justru butuh tanda baca untuk memahami konteks.
2. **No Data Loss Imbalance Handling:** Alih-alih melakukan *undersampling* yang membuang banyak data, kita menggunakan **Smoothed Class Weights** (pembobotan yang dihaluskan dengan akar kuadrat) agar model tetap stabil.
3. **Custom Focal Loss:** Kita memakai fungsi *Focal Loss* modifikasi di TensorFlow supaya model lebih fokus belajar dari *hard examples* (kalimat yang sulit ditebak) dan mengabaikan kalimat mayoritas yang gampang ditebak.
4. **MentalBERT Fine-Tuning:** Memanfaatkan pretrained model dari HuggingFace yang memang sudah dilatih khusus di forum-forum kesehatan mental (Reddit).

## Download Weights Model 2 (MentalBERT)
Karena file tf_model.h5 milik MentalBERT ini ukurannya lumayan besar (di atas 100 MB) file tf_model.h5 dapat diakses di Google Drive. 

- **Link Google Drive**: https://drive.google.com/drive/folders/1NjqP6rpddkiJJteU8oFGT67JeBnspFw7?usp=share_link
