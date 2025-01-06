import streamlit as st

st.set_page_config(page_title="Metode Analisis", page_icon="📚")

st.title("🧠 Metode Analisis")

st.markdown("""
            Pada Halaman ini akan dijelaskan mengenai metode analisis yang digunakan dalam membangun model prediksi nilai teks keunggulan program studi.

            #### Wilayah dan Jadwal Penelitian
            Penelitian ini dilakukan di lingkungan Universitas Bina Nusantara. Waktu pelaksanaan penelitian ini dimulai sejak November 2024 sampai Januari 2025.

            #### Pengumpulan Data
            Data yang digunakan dalam penelitian ini adalah data penjelasan keunggulan program studi yang diajukan dalam usulan program studi baru dari tanggal 7 Januari 2022 hingga 31 Mei 2024. Data ini diperoleh dari situs web pemerintah yang berisi informasi mengenai program studi baru yang diajukan oleh perguruan tinggi.

            #### Pengolahan Data
            Data yang telah diperoleh kemudian diolah dengan melakukan:
            1. Eliminasi kolom yang tidak digunakan
            2. Ekstraksi data teks keunggulan program studi
            3. Pembagian *dataset* menjadi *training*, *validation*, dan *testing*
            4. Memasangkan tiap teks dengan teks lainnya yang memiliki nilai yang berbeda sebagai bentuk pemenuhan prinsip <a href="https://aclanthology.org/2022.coling-1.240.pdf">contrastive pairwise</a>

            #### Pengembangan Model
            Model prediksi yang digunakan pada aplikasi ini adalah <a href="https://aclanthology.org/2022.coling-1.240.pdf"><em>Neural Pairwise Contrastive Regression</em> (NPCR)</a>. Model ini dikembangkan dengan menggunakan *framework* PyTorch dan dibangun dengan menggunakan *pre-trained model <a href="https://arxiv.org/pdf/1810.04805"><em>Bidirectional Encoder Representations from Transformers</em> (BERT)</a> dari Google.
""", unsafe_allow_html=True)

st.image("NPCR_Architecture.png", caption="Gambar 1: Arsitektur Model NPCR")

st.markdown("""
            Gambar 1 menunjukkan arsitektur model NPCR yang digunakan dalam penelitian ini. Model ini menerima dua teks, yang pertama adalah teks yang ingin diprediksi (teks input) dan yang kedua adalah teks referensi yang akan digunakan sebagai pembanding dari teks input. Model ini terdiri dari dua bagian utama yaitu *BERT Encoder* dan *Contrastive Loss*. *BERT Encoder* bertugas untuk menghasilkan representasi vektor dari teks keunggulan program studi yang diberikan. Sedangkan *Contrastive Loss* bertugas untuk membandingkan dua teks yang diberikan dan menghasilkan nilai prediksi berupa skor relatif. Skor tersebut harus ditambahkan dengan skor dari teks referensi untuk mendapatkan hasil prediksi akhir.
""")
