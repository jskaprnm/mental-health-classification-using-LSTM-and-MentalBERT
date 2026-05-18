# Deep Learning-Based Text Analysis for Mental Health Classification

Repositori ini memuat implementasi *pipeline Natural Language Processing* (NLP) menggunakan pendekatan *Deep Learning* untuk mengklasifikasikan tujuh kondisi spektrum kesehatan mental berdasarkan data tekstual dari media sosial. 

Proyek ini mengevaluasi komparasi arsitektur jaringan saraf berulang (**LSTM**) melawan arsitektur atensi mandiri modern berbasis *Transformer* (**MentalBERT**). Selain itu, proyek ini mengusulkan penggunaan **Smoothed Class Weights** yang dipadukan dengan **Sparse Categorical Focal Loss** untuk mengatasi masalah ketidakseimbangan kelas (*class imbalance*) yang sangat ekstrem pada data medis dunia nyata.


## Informasi Dataset
Dataset yang digunakan dalam penelitian ini adalah **"Sentiment Analysis for Mental Health"** yang bersumber dari Kaggle.
- **Total Sampel:** 53.043 baris data teks informal.
- **Variabel Prediktor:** `statement` (Teks curhatan/ungkapan emosional).
- **Variabel Target:** `status` (7 Kategori: *Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder*).
- **Tautan Unduhan Asli:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

---

## Arsitektur Proyek dan Fitur Utama
1. **Adaptive Text Preprocessing:** Pembersihan data teks yang dirancang khusus; agresif untuk LSTM (menghapus tanda baca & angka) dan moderat untuk MentalBERT (mempertahankan tanda baca dan menyamarkan `@USER` tags).
2. **Imbalance Handling:** Algoritma yang secara matematis menghaluskan penalti kerugian kelas minoritas (Akar Kuadrat dari `compute_class_weight`) tanpa membuang sampel data (*no information loss*).
3. **Custom Focal Loss:** Implementasi kelas `SparseCategoricalFocalLoss` pada TensorFlow untuk memaksa model fokus pada *hard examples* (sampel yang sulit diklasifikasi).
4. **Pretrained MentalBERT:** Proses *fine-tuning* pada model HuggingFace `mental/mental-bert-base-uncased` untuk menyerap leksikon psikologis.

---
