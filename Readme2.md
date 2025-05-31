Laporan Proyek Machine Learning - Habib Fabri Arrosyid
Project Overview
Pariwisata merupakan sektor penting bagi perekonomian Indonesia, dengan beragam destinasi seperti pantai, situs budaya, dan taman hiburan. Namun, wisatawan sering kesulitan memilih destinasi yang sesuai preferensi mereka karena banyaknya pilihan dan kurangnya rekomendasi yang dipersonalisasi. Proyek ini bertujuan mengembangkan sistem rekomendasi destinasi wisata di Yogyakarta menggunakan dataset dari Kaggle, untuk memberikan saran yang relevan berdasarkan penilaian pengguna dan atribut destinasi.
Mengapa Masalah Ini Penting?Rekomendasi yang dipersonalisasi meningkatkan pengalaman wisatawan, membantu mereka menemukan destinasi sesuai minat, anggaran, dan lokasi. Sistem ini dapat meningkatkan kunjungan ke destinasi lokal, mendukung bisnis pariwisata, dan mempromosikan destinasi kurang dikenal. Proyek ini juga menunjukkan penerapan teknik machine learning seperti collaborative filtering dan content-based filtering untuk masalah dunia nyata.
Riset dan Referensi  

Dataset: Indonesia Tourism Destination [1].  
Studi tentang collaborative filtering menggunakan TensorFlow untuk rekomendasi berbasis penilaian [2].  
Pendekatan content-based filtering untuk rekomendasi berbasis kesamaan atribut [3].

Referensi:[1] Kaggle, “Indonesia Tourism Destination Dataset,” 2025. [Online]. Tersedia: https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination.[2] J. Brownlee, “Recommender Systems with Python,” Machine Learning Mastery, 2020. [Online]. Tersedia: https://machinelearningmastery.com/recommender-systems-with-python/.[3] Dicoding, “Machine Learning Terapan,” 2024. [Online]. Tersedia: https://www.dicoding.com/academies/319-machine-learning-terapan.  
Business Understanding
Problem Statements

Bagaimana cara membangun sistem rekomendasi destinasi wisata di Yogyakarta yang sesuai dengan preferensi pengguna berdasarkan data penilaian dan atribut destinasi?  
Seberapa efektif pendekatan collaborative filtering dan content-based filtering dalam menghasilkan rekomendasi akurat?  
Bagaimana cara mengevaluasi performa sistem rekomendasi menggunakan metrik seperti RMSE dan Cosine Similarity?

Goals

Mengembangkan sistem rekomendasi destinasi wisata di Yogyakarta yang akurat menggunakan collaborative filtering dan content-based filtering.  
Mengevaluasi performa sistem dengan metrik Root Mean Squared Error (RMSE) untuk collaborative filtering dan Cosine Similarity untuk content-based filtering.  
Menyediakan rekomendasi yang membantu wisatawan merencanakan perjalanan sesuai minat mereka.

Solution Approach

Melakukan Exploratory Data Analysis (EDA) untuk mengidentifikasi pola dalam data destinasi dan penilaian pengguna.  
Menerapkan collaborative filtering menggunakan TensorFlow untuk rekomendasi berbasis pola penilaian.  
Menerapkan content-based filtering untuk rekomendasi berbasis kesamaan atribut seperti kategori dan deskripsi.  
Menggunakan MinMaxScaler untuk normalisasi data numerik.  
Mengevaluasi model dengan metrik RMSE dan Cosine Similarity.

Data Understanding
Dataset bersumber dari Indonesia Tourism Destination di Kaggle:  

tourism_rating.csv: 10.000 entri penilaian (79 duplikat dihapus), kolom: User_Id, Place_Id, Place_Ratings. Tidak ada nilai hilang.  
tourism_with_id.csv: 437 entri destinasi, kolom: Place_Id, Place_Name, Description, Category, City, Price, Rating, Time_Minutes, Coordinate, Lat, Long. Kolom tidak relevan (Unnamed: 11, Unnamed: 12) dihapus.  
user.csv: Informasi pengguna, kolom: User_Id, Location, Age.

Variabel pada Dataset:  

tourism_rating.csv:  
User_Id: ID unik pengguna.  
Place_Id: ID unik destinasi.  
Place_Ratings: Penilaian pengguna (skala 1–5).

  

tourism_with_id.csv:  
Place_Id: ID unik destinasi.  
Place_Name: Nama destinasi.  
Description: Deskripsi destinasi.  
Category: Jenis destinasi (e.g., Bahari, Taman Hiburan).  
City: Kota lokasi destinasi.  
Price: Harga tiket masuk (IDR).  
Rating: Rata-rata penilaian destinasi.  
Time_Minutes: Durasi kunjungan (beberapa nilai hilang).  
Coordinate, Lat, Long: Koordinat geografis.

  

user.csv:  
User_Id: ID unik pengguna.  
Location: Domisili pengguna.  
Age: Usia pengguna.




Exploratory Data Analysis (EDA):  

Destinasi dengan Rating Terbanyak:Gereja Perawan Maria Tak Berdosa Surabaya memiliki rating terbanyak (~400), diikuti destinasi alam seperti Pantai Parangtritis dan Taman Sungai Mudal, menunjukkan preferensi kuat terhadap destinasi religi dan alam.  
Distribusi Kategori:Taman Hiburan (130 destinasi) dan Budaya (120) mendominasi, diikuti Cagar Alam (~100). Pusat Perbelanjaan dan Tempat Ibadah memiliki jumlah terendah (<20).  
Distribusi Usia Pengguna:Mayoritas pengguna berusia 25–35 tahun (median ~30 tahun), menunjukkan dominasi dewasa muda.  
Harga Masuk:Mayoritas destinasi gratis (~350), dengan sedikit destinasi berharga >20.000 IDR, menunjukkan destinasi terjangkau mendominasi.  
Lokasi Pengguna:Semarang memiliki pengguna terbanyak (~38), diikuti Cirebon dan Jakarta, menunjukkan konsentrasi di kota-kota besar Jawa.  
Destinasi di Yogyakarta:Taman Sungai Mudal (~175 rating) dan Pantai Parangtritis mendominasi, menunjukkan preferensi terhadap destinasi alam.  
Kategori di Yogyakarta:Taman Hiburan (35) dan Bahari (25) terbanyak, diikuti Budaya (~20).  
Harga di Yogyakarta:Mayoritas destinasi gratis (~100), dengan sedikit destinasi berharga >10.000 IDR.

Menangani Missing Value:Tidak ada nilai hilang pada dataset (data.isnull().sum()).  
Menangani Duplikasi:Sebanyak 79 data duplikat dihapus dari tourism_rating.csv.  
Data Preparation
1. Filtering DataDataset difilter untuk hanya mencakup destinasi di Yogyakarta.  
point = point[point['City'] == 'Yogyakarta']
rate = pd.merge(rate, point[['Place_Id']], how='right', on='Place_Id')

Alasan: Fokus pada Yogyakarta untuk relevansi regional dan mengurangi kompleksitas data.
2. Menangani DuplikasiSebanyak 79 data duplikat dihapus dari tourism_rating.csv.  
rate = rate.drop_duplicates()

Alasan: Menghilangkan duplikasi untuk mencegah bias dalam pelatihan model.
3. Mapping IDUser_Id dan Place_Id diencode menjadi indeks numerik.  
user_ids = rate['User_Id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
place_ids = rate['Place_Id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
rate['user'] = rate['User_Id'].map(user_to_user_encoded)
rate['place'] = rate['Place_Id'].map(place_to_place_encoded)

Alasan: Encoding diperlukan untuk input numerik pada model machine learning.
4. NormalisasiRating dinormalisasi ke rentang 0–1 menggunakan MinMaxScaler.  
min_rating = min(rate['Place_Ratings'])
max_rating = max(rate['Place_Ratings'])
rate['normalized_rating'] = rate['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))

Alasan: Normalisasi memastikan skala data konsisten untuk pelatihan model.
5. Splitting DataData dibagi menjadi 80% data latih dan 20% data uji.  
from sklearn.model_selection import train_test_split
x = rate[['user', 'place']].values
y = rate['normalized_rating'].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

Alasan: Pemisahan data mencegah kebocoran data dan memungkinkan evaluasi generalisasi.
Modeling
Collaborative Filtering
Model neural network berbasis TensorFlow untuk rekomendasi berbasis pola penilaian.  
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

Kelebihan: Efektif untuk pengguna dengan riwayat penilaian, menangkap pola komunitas.Kekurangan: Rentan terhadap cold-start untuk pengguna baru.  
Content-Based Filtering
Menggunakan TfidfVectorizer dan Cosine Similarity untuk rekomendasi berbasis kesamaan deskripsi.  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

descriptions = point['Description'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(descriptions)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(range(len(point)), index=point['Place_Id'])

def get_content_recommendations(place_id, cosine_sim=cosine_sim, top_n=5):
    idx = indices[place_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    place_indices = [i[0] for i in sim_scores]
    return point.iloc[place_indices]

Kelebihan: Efektif untuk rekomendasi berbasis atribut, tidak bergantung pada data pengguna.Kekurangan: Bergantung pada kualitas deskripsi destinasi.  
Top-N Recommendation:  

Collaborative Filtering: 7 rekomendasi destinasi berdasarkan prediksi rating.  
Content-Based Filtering: 5 rekomendasi destinasi berdasarkan kesamaan deskripsi (e.g., Pantai Congot merekomendasikan pantai lain).

Evaluation
Collaborative Filtering
Metrik: Root Mean Squared Error (RMSE).Formula:[RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}]Mengukur rata-rata kesalahan kuadrat antara rating aktual dan prediksi. Nilai lebih rendah menunjukkan akurasi lebih baik.  
Hasil:  

RMSE pelatihan menurun dari 0.3496 ke 0.3492 selama 50 epoch, tetapi RMSE validasi stagnan (~0.3493), menunjukkan potensi overfitting ringan.  
Pengujian untuk User ID 200:Rekomendasi relevan dengan preferensi pengguna (Cagar Alam, Bahari) dan harga terjangkau.

Content-Based Filtering
Metrik: Cosine Similarity.Formula:[\text{Cosine Similarity} = \frac{A \cdot B}{|A| |B|}]Mengukur kesamaan antara vektor fitur (TF-IDF). Nilai mendekati 1 menunjukkan kesamaan tinggi.  
Hasil:  

Pengujian untuk Place_Id=210 (Pantai Congot) menghasilkan 5 rekomendasi pantai di Yogyakarta (e.g., Pantai Timang, rating 4.7), relevan berdasarkan kategori Bahari dan deskripsi.

Kesimpulan
Sistem rekomendasi destinasi wisata di Yogyakarta berhasil dibangun menggunakan dataset dari Kaggle dengan pendekatan collaborative filtering (RMSE ~0.3493) dan content-based filtering (evaluasi Cosine Similarity). Collaborative filtering efektif untuk pengguna dengan riwayat penilaian, tetapi rentan terhadap cold-start. Content-based filtering akurat untuk rekomendasi berbasis atribut, tetapi bergantung pada kualitas deskripsi. Sistem ini mendukung wisatawan dalam merencanakan perjalanan, meskipun dapat ditingkatkan dengan pendekatan hybrid dan data tambahan seperti ulasan pengguna.
Catatan:  

Kode dan visualisasi lengkap tersedia di repository GitHub.  
Gambar dan tabel mendukung analisis EDA dan evaluasi model.

