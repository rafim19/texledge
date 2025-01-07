import streamlit as st

st.set_page_config(page_title="TexLedge", page_icon="📚")

st.title("📚 Selamat Datang di TexLedge!")

st.markdown("""
### Tentang TexLedge

TexLedge adalah sebuah alat bantu untuk website pemerintah yang berfokus pada prediksi nilai penjelasan keunggulan program studi dalam usulan program studi baru. Dengan menggunakan model *Nerual Pairwise Contrastive Regression* (NPCR), TexLedge dapat membantu pengguna dengan memberikan prediksi nilai teks penjelasan keunggulan program studi secara otomatis.

### Fitur Utama:
- Prediksi nilai teks keunggulan program studi

### Cara Penggunaan:
1. Masuk ke halaman Prediksi
2. Masukkan teks keunggulan program studi anda pada form yang disediakan
3. Sistem akan menganalisis dan memberikan prediksi
4. Sistem akan memberikan rekomendasi jika nilai prediksi masih dapat ditingkatkan

Aplikasi ini dikembangkan untuk membantu proses pengusulan program studi baru agar lebih efektif dan efisien.
Berikut adalah video *tutorial* penggunaan aplikasi TexLedge:
""")
st.video("https://www.youtube.com/watch?v=2Ju1ApN0swc")