# Model_Rekomendasi_Tempat_WIsata

# Laporan Proyek Machine Learning Terapan 2 - Habib Fabri Arrosyid 

## Domain Proyek
Latar Belakang

Pariwisata merupakan penyumbang signifikan bagi perekonomian Indonesia, dengan berbagai atraksi mulai dari pantai hingga situs budaya. Namun, wisatawan sering kali menghadapi tantangan dalam menemukan destinasi yang sesuai dengan preferensi mereka karena banyaknya pilihan dan keterbatasan rekomendasi yang dipersonalisasi. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi destinasi wisata di Indonesia, khususnya dengan memanfaatkan dataset dari Kaggle untuk memberikan saran yang disesuaikan dengan preferensi pengguna dan penilaian.

Pentingnya Proyek

Proyek ini penting karena mengatasi kebutuhan akan rekomendasi perjalanan yang dipersonalisasi, meningkatkan pengalaman pengguna dengan menyarankan destinasi yang sesuai dengan minat mereka. Dengan menerapkan sistem rekomendasi, kami dapat membantu wisatawan membuat keputusan yang lebih tepat, yang berpotensi meningkatkan pariwisata lokal dan mendukung bisnis di berbagai wilayah. Proyek ini juga menunjukkan penerapan teknik pembelajaran mesin, seperti collaborative filtering dan content-based filtering, dalam menyelesaikan masalah dunia nyata.

Riset dan Referensi

Dataset yang digunakan dalam proyek ini bersumber dari Kaggle: Dataset Destinasi Wisata Indonesia. Referensi tambahan mencakup studi tentang sistem rekomendasi, terutama teknik collaborative filtering menggunakan TensorFlow (seperti yang terlihat di notebook) dan pendekatan content-based filtering untuk rekomendasi yang dipersonalisasi.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang di atas, permasalahan yang akan dibahas dalam proyek ini adalah:

1. Bagaimana cara membangun sistem rekomendasi yang dapat memberikan saran destinasi wisata di Indonesia yang sesuai dengan preferensi pengguna berdasarkan data penilaian dan atribut destinasi?
2. Seberapa efektif pendekatan collaborative filtering dan content-based filtering dalam memberikan rekomendasi yang akurat?
3. Bagaimana cara mengevaluasi performa sistem rekomendasi menggunakan metrik seperti RMSE dan Cosine Similarity?

### Goals
Berdasarkan problem statements, tujuan proyek ini adalah:

1. Membangun sistem rekomendasi destinasi wisata yang akurat menggunakan pendekatan collaborative filtering dan content-based filtering.
2. Mengevaluasi performa sistem rekomendasi dengan metrik evaluasi seperti Root Mean Squared Error (RMSE) untuk collaborative filtering dan Cosine Similarity untuk content-based filtering.
3. Menyediakan rekomendasi yang mendukung wisatawan dalam merencanakan perjalanan yang sesuai dengan minat mereka.

### Solution Statement
Untuk mencapai tujuan, dua pendekatan rekomendasi diusulkan:
- Melakukan Exploratory Data Analysis (EDA) untuk mengidentifikasi pola dan tren dalam data destinasi wisata dan penilaian pengguna.
- Menerapkan collaborative filtering menggunakan TensorFlow untuk merekomendasikan destinasi berdasarkan pola penilaian pengguna.
- Menerapkan content-based filtering untuk merekomendasikan destinasi berdasarkan kesamaan atribut seperti kategori, lokasi, dan harga.
- Menggunakan MinMaxScaler untuk normalisasi data numerik agar sesuai dengan kebutuhan model.
- Mengevaluasi performa model dengan metrik RMSE untuk collaborative filtering dan Cosine Similarity untuk content-based filtering.

## Data Understanding
#### Sumber Data
<img></img>
Link : https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

Informasi Dataset
#### Tipe dan Bentuk Data 
Proyek ini menggunakan tiga dataset dari dataset Destinasi Wisata Indonesia di Kaggle:

- tourism_rating.csv:
Berisi penilaian pengguna untuk destinasi wisata.
Kolom: User_Id, Place_Id, Place_Ratings.
Ukuran: 10.000 entri awal, dengan 79 duplikat dihapus.
Tidak ada nilai yang hilang.
- tourism_with_id.csv:
Detail tentang destinasi wisata.
Kolom: Place_Id, Place_Name, Description, Category, City, Price, Rating, Time_Minutes, Coordinate, Lat, Long.
Ukuran: 437 entri.
Menghapus dua kolom yang tidak relevan (Unnamed: 11, Unnamed: 12).
- user.csv:
Informasi pengguna (tidak banyak digunakan di notebook tetapi tersedia untuk pengembangan lebih lanjut).
Kolom: User_Id, atribut pengguna tambahan.

Deskripsi Variabel
Dataset memiliki 5 variabel dengan keterangan sebagai berikut:

- tourism_rating.csv:
User_Id: Pengidentifikasi unik untuk pengguna.
Place_Id: Pengidentifikasi unik untuk destinasi wisata.
Place_Ratings: Penilaian yang diberikan pengguna untuk destinasi (skala: 1–5).
<img>

- tourism_with_id.csv:
Place_Id: Pengidentifikasi unik untuk destinasi.
Place_Name: Nama destinasi.
Description: Deskripsi singkat destinasi.
Category: Jenis destinasi (misalnya, Bahari untuk pantai).
City: Lokasi destinasi.
Price: Harga tiket masuk (dalam IDR).
Rating: Rata-rata penilaian destinasi.
Time_Minutes: Estimasi durasi kunjungan (beberapa nilai hilang).
Coordinate, Lat, Long: Koordinat geografis.
<img>

- user.csv:
User_Id: Id unik pengguna
Location : domisili pengguna
Age : Usia Pengguna
<img>

### Menangani Missing Value dan Duplicate Data
Pada tahap ini, dataset diperiksa untuk memastikan tidak ada nilai yang hilang (missing values) Berdasarkan analisis awal:
<img><br>
Tidak ada nilai yang hilang pada dataset (dikonfirmasi dengan data.isnull().sum()).
### Menangani Data Duplikat
Pada dataset ini terdapat duplikasi data sebanyak 79 data
<img>
Kemudian akan dilakukan penghapusan data duplikat 
<img>

### Visualisasi Data EDA

Visualisasi data dilakukan untuk menggali insight yang terlihat dari data:
- Tempat destinasi dengan rating terbanyak
<img src="https://github.com/habibarrsyd/Predictive_Analysis_Return_Stock_BMRI_Prediction-/blob/2424506a29e46ba9f56a72f940f261567d855ec5/img/graph_data_predictiveanalysis.jpg"><br>

Interpretasi:

- Sebaran tempat destinasi berdasarkan kategori
<img>
- Sebaran usia pengguna
<img>
- Persebaran harga masuk destinasi
<img>
- Sebaran lokasi pengguna
<img>
- Tempat destinasi di Jogja dengan rating terbanyak
<img>
- Sebaran tempat destinasi di Jogja (setelah filter) berdasarkan kategori
<img>
- Sebaran harga masuk destinasi di Jogja
<img>

## Data Preparation
### Feature Engineering:
#### Dilakukan filtering pada dataset untuk mengambil data pada daerah Jogja saja.
<img>

Data yang terbentuk : 
<img src="https://github.com/habibarrsyd/Predictive_Analysis_Return_Stock_BMRI_Prediction-/blob/2424506a29e46ba9f56a72f940f261567d855ec5/img/data_after_fenginering_predictiveanalysis.jpg"><br>

#### Maping Id
<code>
buat apa dijelasin disini

#### Tambah kolom encoded
<code>
dijelaskan disini

### Normalisasi
<code>
dijelaskan disini untuk apa

### Splitting data
<code>
dijelaskan dsini untuk apa

## Pemodelan
### Collaborative Filtering
<code>
    dijelaskan ngapain kodenya

### Content Based Filtering
<code>
    dijelaskan ngapain kodenya

## Evaluasi
### Collaborative Filtering
<code>
    dijelaskan isinya

### Content Based Filtering
<code>
dijelaskan isinya


## Pengujian  
### Collaborative Filtering
<code>
dijelaskan skemanya ngapain
    hasilnya : 
<pict></pict>
interpretasi

### Content Based Filtering
<code>
dijelaskan skemanya gimana
hasilnya : 
<pict></pict>
interpretasi

## Kesimpulan
Berdasarkan analisis dan pengujian, kesimpulan dari proyek ini adalah:

Dataset destinasi wisata Indonesia dari Kaggle berhasil digunakan untuk membangun sistem rekomendasi dengan dua pendekatan: collaborative filtering dan content-based filtering. Collaborative filtering efektif untuk pengguna dengan riwayat penilaian, dengan RMSE yang menunjukkan akurasi prediksi yang memadai, tetapi rentan terhadap masalah cold-start. Content-based filtering berhasil merekomendasikan destinasi berdasarkan kesamaan atribut, seperti pantai di Yogyakarta, dengan Cosine Similarity sebagai metrik evaluasi. Sistem ini mendukung wisatawan dalam merencanakan perjalanan dengan rekomendasi yang relevan, meskipun performa content-based filtering bergantung pada kualitas atribut destinasi. Untuk perbaikan, disarankan menambahkan data eksternal seperti ulasan pengguna atau sentimen media sosial, serta mengintegrasikan pendekatan hybrid untuk mengatasi keterbatasan masing-masing metode.

## Referensi
- Kaggle. (2025). Indonesia Tourism Destination Dataset. Diakses pada 29 Mei 2025 dari https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination.
- Brownlee, J. (2020). Recommender Systems with Python. Machine Learning Mastery. Diakses pada 29 Mei 2025 dari https://machinelearningmastery.com/recommender-systems-with-python/.
- Dicoding. (2024). Machine Learning Terapan. Diakses pada 29 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.
