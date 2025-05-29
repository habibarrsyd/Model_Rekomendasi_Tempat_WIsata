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
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/datakaggle.jpg"><br>
Link : https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

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
<img src = "https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/bentukratekolom.jpg"><br>

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
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/poinkolom.jpg"><br>

- user.csv: <br>
User_Id: Id unik pengguna

Location : domisili pengguna

Age : Usia Pengguna

<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/userkolom.jpg"><br>

### Menangani Missing Value 
Pada tahap ini, dataset diperiksa untuk memastikan tidak ada nilai yang hilang (missing values) Berdasarkan analisis awal:
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/nullrate.jpg"><br>
Tidak ada nilai yang hilang pada dataset (dikonfirmasi dengan data.isnull().sum()).
### Menangani Data Duplikat
Pada dataset ini terdapat duplikasi data sebanyak 79 data <br>
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/rateduplikat.jpg"><br>
Kemudian akan dilakukan penghapusan data duplikat <br>
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/dropduplikat.jpg"><br>

### Visualisasi Data EDA

Visualisasi data dilakukan untuk menggali insight yang terlihat dari data:
- Tempat destinasi dengan rating terbanyak
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/destinasiratingall.jpg"><br>

Interpretasi:

- Sebaran tempat destinasi berdasarkan kategori
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/destinasikategoriall.jpg"><br>
Interpretasi :

- Sebaran usia pengguna
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/usiaall.jpg"><br>
Interpretasi :

- Persebaran harga masuk destinasi
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/hargaall.jpg"><br>
Interpretasi :

- Sebaran lokasi pengguna
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/sebaranuser.jpg"><br>
Interpretasi :

- Tempat destinasi di Jogja dengan rating terbanyak
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/destinasiratingjogja.jpg"><br>
Interpretasi :

- Sebaran tempat destinasi di Jogja (setelah filter) berdasarkan kategori
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/destinasikategorijogja.jpg"><br>
Interpretasi :

- Sebaran harga masuk destinasi di Jogja
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/hargajogja.jpg"><br>
Interpretasi :


## Data Preparation
### Feature Engineering:
#### Dilakukan filtering pada dataset untuk mengambil data pada daerah Jogja saja.
```
# Filter destinasi hanya dari Yogyakarta
point = point[point['City'] == 'Yogyakarta']
rate = pd.merge(rate, point[['Place_Id']], how='right', on='Place_Id')
```
Bentuk data yang terbentuk : <br>
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/bentukpoinmodif.jpg"><br>

#### Maping Id
```
# Membuat mapping ID
user_ids = rate['User_Id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for x, i in user_to_user_encoded.items()}

place_ids = rate['Place_Id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for x, i in place_to_place_encoded.items()}
```
buat apa dijelasin disini

#### Tambah kolom encoded
```
# Tambahkan kolom encoded
rate['user'] = rate['User_Id'].map(user_to_user_encoded)
rate['place'] = rate['Place_Id'].map(place_to_place_encoded)

num_users = len(user_ids)
num_place = len(place_ids)
```
dijelaskan disini

### Normalisasi
```
# Normalisasi rating ke 0-1
min_rating = min(rate['Place_Ratings'])
max_rating = max(rate['Place_Ratings'])
rate['normalized_rating'] = rate['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
```
dijelaskan disini untuk apa

### Splitting data
```
# Siapkan data training
x = rate[['user', 'place']].values
y = rate['normalized_rating'].values

# Split train/val
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
```
dijelaskan dsini untuk apa

## Pemodelan
### Collaborative Filterin
```
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.places_embedding = layers.Embedding(num_places, embedding_size, embeddings_initializer='he_normal',
                                                 embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.places_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        places_vector = self.places_embedding(inputs[:, 1])
        places_bias = self.places_bias(inputs[:, 1])
        dot_user_places = tf.reduce_sum(user_vector * places_vector, axis=1, keepdims=True)
        x = dot_user_places + user_bias + places_bias
        return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_place, 50)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.Adagrad(learning_rate=0.0003),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_root_mean_squared_error') < 0.25:
            print('\nRoot metrics validasi sudah sesuai harapan')
            self.model.stop_training = True

history = model.fit(
    x=x_train,
    y=y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[myCallback()]
)


```
dijelaskan ngapain kodenya

### Content Based Filtering
#### Ekstraksi Fitur
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Misal kolom fitur text untuk tiap tempat ada di 'Description' atau 'Features'
descriptions = point['Description'].fillna('')  # sesuaikan kolom yang dipakai

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(descriptions)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# buat indices sesuai semua place_id
indices = pd.Series(range(len(point)), index=point['Place_Id'])
```
dijelaskan ngapain kodenya
#### Fungsi Rekomendasi
```
def get_content_recommendations(place_id, cosine_sim=cosine_sim, top_n=5):
    idx = indices[place_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # ambil 5 tempat mirip (selain dirinya sendiri)
    place_indices = [i[0] for i in sim_scores]
    return point.iloc[place_indices]
```
dijelaskan fungsinya

## Evaluasi
### Collaborative Filtering
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/hasilepoch.jpg"><br>
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/grafikrmse.jpg"><br>
dijelaskan isinya

### Content Based Filtering
#### Melihat ukuran dari cosine, baris, dan key sampel
```
print(f"Ukuran cosine_sim: {cosine_sim.shape}")
print(f"Jumlah baris point: {len(point)}")
print(f"Indices keys example: {list(indices.keys())[:10]}")
print(f"indices[{179}]: {indices.get(179)}")
```
Output : <br>
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/ukcosim.jpg"><br>

#### Melihat key yang dapat diujikan
```
print(point['Place_Id'].unique())
```
#### Bentuk key data yang dapat diujikan
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/rawdata.jpg"><br>

#### Kaitannya dengan hasil simulasi
Contoh bentuk hasil simulasi <br>
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/ujicontentbased.jpg"><br>
dijelaskan

## Pengujian  
### Collaborative Filtering
```
from tabulate import tabulate
import numpy as np

# Input user ID secara manual
try:
    user_id = int(input("Masukkan User ID: "))
except ValueError:
    print("User ID harus berupa angka.")
    exit()

# Proses rekomendasi
encoded_user_id = user_to_user_encoded.get(user_id)

# Validasi jika user ID tidak ditemukan
if encoded_user_id is None:
    print(f"User ID {user_id} tidak ditemukan dalam data.")
    exit()

place_visited_by_user = rate[rate['User_Id'] == user_id]
place_not_visited = list(set(place_ids) - set(place_visited_by_user['Place_Id'].values))
place_not_visited_encoded = [place_to_place_encoded.get(x) for x in place_not_visited]

user_place_array = np.array([[encoded_user_id, place] for place in place_not_visited_encoded])
ratings = model.predict(user_place_array).flatten()
top_ratings_indices = ratings.argsort()[-7:][::-1]
recommended_place_ids = [place_encoded_to_place[place_not_visited_encoded[x]] for x in top_ratings_indices]

print('=' * 50)
print(f"Rekomendasi Tempat Wisata untuk User {user_id}")
print('=' * 50)

# Top 5 tempat yang pernah disukai
top_place_user = place_visited_by_user.sort_values(
    by='Place_Ratings', ascending=False).head(5)['Place_Id'].values
place_df_rows = point[point['Place_Id'].isin(top_place_user)]

top_visited_table = [
    [row['Place_Name'], row['Category'], row['Rating'], row['Price']]
    for _, row in place_df_rows.iterrows()
]
print("\nTempat yang Pernah Disukai:")
print(tabulate(top_visited_table, headers=["Nama Tempat", "Kategori", "Rating", "Harga"], tablefmt="fancy_grid"))

# Rekomendasi 7 tempat
recommended_place = point[point['Place_Id'].isin(recommended_place_ids)]
recommended_table = [
    [i + 1, row['Place_Name'], row['Category'], row['Rating'], row['Price']]
    for i, (_, row) in enumerate(recommended_place.iterrows())
]
print("\nRekomendasi Tempat:")
print(tabulate(recommended_table, headers=["#", "Nama Tempat", "Kategori", "Rating", "Harga"], tablefmt="fancy_grid"))
```
dijelaskan skemanya ngapain
hasilnya : 
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/ujicolaboratiffilter.jpg"><br>
interpretasi : 


### Content Based Filtering
```
get_content_recommendations(210)
```
dijelaskan skemanya gimana
hasilnya : 
<img src="https://github.com/habibarrsyd/Model_Rekomendasi_Tempat_WIsata/blob/7943c2f60bf45068be0aecc9d812ce514b360393/img/ujicontentbased.jpg"><br>
interpretasi : 


## Kesimpulan
Berdasarkan analisis dan pengujian, kesimpulan dari proyek ini adalah:

Dataset destinasi wisata Indonesia dari Kaggle berhasil digunakan untuk membangun sistem rekomendasi dengan dua pendekatan: collaborative filtering dan content-based filtering. Collaborative filtering efektif untuk pengguna dengan riwayat penilaian, dengan RMSE yang menunjukkan akurasi prediksi yang memadai, tetapi rentan terhadap masalah cold-start. Content-based filtering berhasil merekomendasikan destinasi berdasarkan kesamaan atribut, seperti pantai di Yogyakarta, dengan Cosine Similarity sebagai metrik evaluasi. Sistem ini mendukung wisatawan dalam merencanakan perjalanan dengan rekomendasi yang relevan, meskipun performa content-based filtering bergantung pada kualitas atribut destinasi. Untuk perbaikan, disarankan menambahkan data eksternal seperti ulasan pengguna atau sentimen media sosial, serta mengintegrasikan pendekatan hybrid untuk mengatasi keterbatasan masing-masing metode.

## Referensi
- Kaggle. (2025). Indonesia Tourism Destination Dataset. Diakses pada 29 Mei 2025 dari https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination.
- Brownlee, J. (2020). Recommender Systems with Python. Machine Learning Mastery. Diakses pada 29 Mei 2025 dari https://machinelearningmastery.com/recommender-systems-with-python/.
- Dicoding. (2024). Machine Learning Terapan. Diakses pada 29 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.
