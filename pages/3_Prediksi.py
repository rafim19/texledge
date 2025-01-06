#%%
import streamlit as st
import torch
from transformers import BertTokenizer
import pandas as pd
from streamlit_quill import st_quill
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import joblib
import torch.nn.functional as F
import inspect

#%%
# print(f"PyTorch version: {torch.__version__}")
st.set_page_config(page_title="Prediksi", page_icon="📚")

#%%

# import os
# os.path.join(os.path.dirname(__file__), 'model')
from networks.reader import text_tokenizer
# from networks.core_networks import npcr_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rescale_tointscore(scaled_scores):
    '''
    rescale scaled scores range[0,1] to original integer scores based on  their set_ids
    :param scaled_scores: list of scaled scores range [0,1] of essays
    :param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
    '''
    # print type(scaled_scores)
    # print scaled_scores[0:100]
    print("scaled_scores:", scaled_scores)
    global maxscore, minscore
    real_scores = np.zeros((scaled_scores.shape[0], 1))
    minscore = -4
    maxscore = 4
    for k, i in enumerate(scaled_scores):
        real_scores[k] = scaled_scores[k]*(maxscore - minscore) + minscore

    return real_scores

@st.cache_resource
def load_model():
    model_path = "./networks/devrank_core_bert.prompt.pt"
    # model_path = "./networks/classification_devrank_core_bert.prompt.pt"
    # model_path = "./networks/cls_devrank_core_bert.prompt.pt"
    model = torch.load(model_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(device)
    model.eval()
    # st.write(model)
    # def forward(self, input_ids, attention_mask=None):
    #     x0_embed = self.embedding(input_ids)[1]
    #     x0_nn1 = self.nn1(x0_embed)
    #     x0_nn1_d = self.dropout(x0_nn1)
    #     x0_relu = self.nn2(x0_nn1_d)
    #     y = self.output(x0_relu)
    #     return y

    # model.forward = forward.__get__(model, type(model))
    # st.write(inspect.getsource(model.forward))

    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    return model, tokenizer

def preprocess_input(text):
    return text_tokenizer(text)

@st.cache_resource
def get_reference_text():
    text = "Pencapaian pembelajaran berdasarkan Kurikulum telah diatur UU No122012 dan diatur pula pada Permendikbud No 3 Tahun 2020 tentang Standar Nasional Pendidikan Tinggi SNDikti dan deskripsi level 8 delapan Kerangka Kualifikasi Nasional Indonesia KKNI sesuai Perpres Nomor 8 Tahun 2012 dan yang terstruktur untuk tercapainya tujuan terlaksananya misi dan terwujudnya visi keilmuan program studiPengembangan KeilmuanProgram Studi Magister Ilmu Pemerintahan Fisip Unmul telah mendesain Visi Misi dan Tujuan Program Studi sesuai kondisi terkini saat ini Salah satu upaya dengan melakukan perbandingan kurikulum Program Studi sejenis dengan 3 Tiga Perguruan Tinggi diantaranya Prodi Magister Ilmu Pemerintahan Universitas Muhamaddiyah UMY Jogjakarta Prodi Magister Ilmu Pemerintahan Universitas Padjajaran Unpad Bandung dan Prodi Magister Ilmu Pemerintahan Universitas Lampung Unila Lampung Kajian ini penting dilakukan untuk menentukan perbandingan keunikan dari Program Studi yang diusulkan Perbandingan ke 3 Tiga Perguruan Tinggi dari Program Studi yang sama diharapkan Program Studi Magister Ilmu Pemerintahan Fisip Unmul memiliki keunikan dengan ciri khas yang berbeda dari Perguruan Tinggi Lain dalam mendukung program pemerintah melalui Kampus Merdeka Tabel 11 Perbandingan pengembangan Keilmuan Magister Ilmu pemerintahan Univeritas Mulawarman dengan Magister Ilmu Pemerintahan di perguruan tinggi NasionalUraianMegister Ilmu Pemerintahan Universitas PadjadajanMagister Ilmu Pemerintahan Universitas LampungMagister Ilmu Pemerintahan Universitas Muhammadiyah YogyakartaMagister Ilmu Pemerintahan Universitas MulawarmanPengembangan keilmuanMagister Ilmu Pemerintahan MIP memiliki pengembangan kajian keilmuan yang khas dengan kajian keilmuan di bidang proses pemerintahan lokal desa dan kebijakan publik Mebagai ilmu yang ekletik MIP juga dibekali dengan kemanjuan pengetahuan konstitusi hubungan pusat dan daerah filsafat ilmu pemerintahan birokrasi dan pelayanan umum isuisu pemerintahn dan dinamika kebijakan metodelogi penelitian sosial sehingga MIP mampu mngembangkan pengetahuan teknologi dan praktik profesional melalui riset yang manghasilkan karya inovatif dan teruji yang dapat memberikan kontribusi nyata dan positif terhadapat kebelangsungan penyelenggaraan pemerintahan dan kebijakan publik di tingkat lokal nasional dan internasionalMagister ilmu Pemerintahan MIP memiliki karateristik kajian keilmuan dalam Pengembangan Manajemen Pemerintahan serta Politik Lokal dan Otonomi Daerah Berbasis NilaiNilai Lokal yang melahirkan MIP yang mampu memahami konsepkonsep mampu menganalisis mengkritisi dan memecahkan masalah dalam lingkup kajian manajemen pemerintahan serta politik lokal dan otonomi daerah Kajian manajemen pemerintahan serta politik lokal dan otonomi daerah merupakan dasar pembangunan sistim politik dan pemerintahan yang sehat dan baik di tengah tingginya tuntutan masyarakat akan adanya reformasi kehidupan politik demokrasi pemerintahan menuju kehidupan politik yang demokratis partisipatif pemerintahan yang melaksanakan prinsipprinsip good governance demi tercapainya citacita nasional Magister Ilmu Pemerintahan MIP berkaitan erat dengan kajian ilmu pemerintahan yang berlandaskan kepada nilainilai keislaman dengan Karakteristik kajian keilmuan ditekankan pada bidang pembangunan kabijakan publik dan dibidang manajementata kelola pemerintahan dengan prinsip tata kelola yang baik termasuk transparansi akuntabilitas profesionalisme dan pendekatan partisipatif diprioritaskan dalam proses pendidikan dalam program MIP MIP dibekali dengan kemajuan pengetahuan sistem pemerintahan reformasi birokrasi kelembagaan pemerintahan budgeting system policy making analysis system kepartaian local government dan dynamic governance dalam mengelola konflik yang terjadi dalam hubungan antar stakeholders menjadi ciri khas kompetensi lulusan MIP Sehingga lulusan MIP mampu Menganalisis memimpin dan mengelola secara teoritis perundangundangan dan teknologi informasi tentang masalahmasalah yang berkaitan dengan eksistensi organisasiorganisasi kelomppokkelompok dan individuindividu dalam masyarakat pascamodern era sekarang ini Memiliki keahlian untuk menganalisis memimpin dan mengelola secara teoritis perundangundangan pemerintahan parpol dan teknologi informasi tentang masalahmasalah yang berkaitan dengan artikulasi dan agregasi kepentingan dan kebutuhan pelayanan publikMemiliki keahlian untuk menganalisis memimpin dan mengelola secara teoritis perundangundangan dan teknologi informasi tentang masalahmasalah yang berkaitan dengan formulasi kebijakan publik atau pembuatan keputusan yang menyangkut masalahmasalah pelayanan publik bagi warga masyarakatMemiliki keahlian kerjasama dengan organisasi politik Ormas dan pemerintah untuk penyelesaian masalah publik dan politikPengembangan kajian keilmuan yang memiliki ciri khas di bidang Tata Kelola Pemerintahan Ibu Kota Negara dan politik dan kebijakan Sumber Daya Alam Magister Ilmu Pemerintahan dibekali dengan kemajuan pengatahuan dan ilmu Tata Kelola Pemerintahan IKN Filsafat dan Teori Ilmu Pemerintahan ICT Politik dan Governance Politik Kebijakan Enterpreneur Government dilandaskan pada fenomena dan pengalaman empirik pemerintahan daerah yang memiliki basis kearifan lokal dan keunggulan berada di wilayah hutan tropis basah dan lingkungannya serta keberadaanya yang berada di pulau kalimantan yang menjadi Ibu Kota Negara Baru Berdasarkan keunikan dan keunggulan Magister Ilmu Pemerintahan Universitas Mulawarman dengan Perguruan Tinggi lain maka ditetapkan visi misi dan tujuan Program Magister Ilmu Pemerintahan Fisip Universitas Mulawarman sebagai berikutVisi Program Magister Ilmu Pemerintahan Universitas Mulawarman adalah Menjadi Program Magister yang terkemuka di Indonesia dalam pengembangan tata Kelola Pemerintahan berbasis Ibu Kota Negara Politik dan Kebijakan Sumber Daya Alam melalui proses akademik penelitian dan pengabdian masyarakat yang diakui secara Nasional Regional dan Internasional di tahun 2030Misi Program Magister Ilmu Pemerintahan Universitas Mulawarman Menyelenggarakan pendidikan yang berkualitas dan efektif secara berkelanjutan yang mampu menghasilkan Magister Ilmu Pemerintahan yang memiliki kompetensi secara kognitif afektif dan psikomotorik Menyelenggarakan Tri Dharma Perguruan Tinggi dengan 25mengembangkan penelitian dasar dan terapan yang inovatif dalam kajian otonomi daerah Tata Kelola Pemerintahan dan Pengembangan Ibu Kota Negara Nusantara IKN Menyelenggarakan pengabdian kepada masyarakat dalam rangka penerapan kompetensi di bidang Ilmu Pemerintahan Mengembangkan institusi dalam meningkatkan kualitas tata kelola Program Magister Ilmu Pemerintahan yang baik sehingga mampu mengantisipasi dan mengakomodasi perubahan lingkungan strategis Membangun jaringan dan kerja sama yang produktif untuk menunjang pendidikan pemerintahan dan dunia usaha di tingkat daerah nasional dan internasional dalam pengembangan Ilmu Pemerintahan Tujuan Program Magister Ilmu Pemerintahan Universitas Mulawarman Terwujudnya sumber daya manusia Ilmu Pemerintahan yang berkualitas bertaqwa kepada Tuhan Yang Maha Esa mampu membelajarkan diri memiliki wawasan yang luas kritis memiliki kepekaan disiplin dan etos kerja dan profesional yang tangguh dan memiliki daya saing dan diakui di tingkat nasional dan internasionalTerwujudnya peningkatan kemampuan dosen dalam melakukan penelitian ilmiah baik dari segi kuantitas maupun kualitas penelitian yang menitikberatkan kepada kajian Proses Tata Kelola Pemerintahan IKN Filsafat dan Teori Ilmu Pemerintahan ICT Politik dan Governance Politik Legislasi Entrepreneur Government yang mempunyai relevansi dan berguna bagi yang berkepentingan dan masyarakat luasTercapainya lulusan Program Magister Ilmu Pemeritahan yang inovatif kritis dan multi paradigmatik dalam kajian pengembangan ilmu dan tata Kelola Pemerintahan berbasis Ibu Kota Negara Baru dengan memecahkan permasalahan bangsa melalui penyelenggaraan programprogram penelitian unggulan strategis yang berguna bagi masyarakat Tercapainya percepatan implementasi hasil penelitian di bidang Ilmu Pemerintahan kepada masyarakat dalam rangka transformasi ilmu pemerintahan kepada masyarakat Terlaksananya mutu pelayanan melalui penyediaan fasilitas sarana prasarana dan teknologi informasi sesuai dengan Standar Nasional Pendidikan Tinggi untuk pengembangan keilmuan di bidang pengembangan ilmu dan tata Kelola Pemerintahan berbasis Ibu Kota Negara Baru melalui kegiatan pendidikan penelitian pengabdian kepada masyarakat dan kegiatan non akademik lain yang mendukung Terbentuknya jaringan dan kerjasama dengan berbagai lembaga pendidikan tinggi pemerintah organisasi swasta dan NGO di tingkat nasional dan internasional Dalam mewujudkan visi misi dan tujuan tersebut maka disusunlah analisis komponen yang menjadi tolak ukur indikator dari misi dan tujuan tersebut Selanjutnya analisis komponen dilakukan analisis SWOT terhadap Visi Misi dan Tujuan tersebut yakni Kekuatan Strenghts Visi Program Magister Ilmu Pemerintahan mencerminkan tekad dan komitmen yang kuat untuk menghasilkan pemikiran konsep dan aplikasi yang spesifik yang berfokus pada kajian ilmu pemerintahan dan tata Kelola pengembangan IKN Politik dan Kebijakan Sumber Daya Alam Filsafat Ilmu Pemerintahan ICT Politik dan Governance Entrepreneur Government sesuai dengan perkembangan otonomi daerah dan politik kebijakan baik ditingkat lokal maupun nasional Rumusan visi misi dan tujuan Program Magister Ilmu Pemerintahan telah selaras dengan visi misi tujuan dan sasaran Universitas Mulawarman dan Fisip Unmul yang mencerminkan Tri Dharma Perguruan Tinggi secara terpaduMagister Ilmu Pemerintahan memiliki posisi yang strategis di wilayah Ibu Kota Negara di Provinsi Kalimantan Timur dengan kondisi lingkungan yang berkarakteristik hutan tropis basah dan lingkungannya yang menjadi basis utama landasan dalam pengembangan keilmuan di bidang pemerintahanKelemahan Weaknesses Dukungan dari stakeholders dan lembagalembaga terkait belum dapat digarap secara serius dan optimal Ke depan perlu dibuat banyak kerja sama yang lebih konkrit yang saling menguntungkan dan memberikan nilai tambah yang signifikan Terbatasnya fasilitas dan anggaran dalam mengoptimalkan pengembangan Tri Dharma Perguruan TinggiProgram Magister Ilmu Pemerintahan Fisip Unmul baru berdiri sehingga belum bisa menyamai Program Magister Ilmu Pemerintahan di Universitas lain dalam membangun jaringan dalam pengembangan keilmuan3 Peluang Opportunities Program Magister Ilmu Pemerintahan Fisip Unmul merupakan Program Magister Ilmu Pemerintahan pertama dan satusatunya di Provinsi Kalimantan Timur Sebagai daerah penyangga IKN Nusantara di Kalimantan Timur Program Magister Ilmu Pemerintahan memiliki peluang besar untuk menyelenggarakan kegiatan yang berkaitan dengan pendidikan penelitian dan pengabdian kepada masyarakat di bidang kajian Pemerintahan dan tata Kelola Pengembangan Ibu Kota Negara Baru Terbukanya peluang kerja sama kegiatan pendidikan penelitian dan pengabdian kepada masyarakat dengan berbagai instansi dan perguruan tinggi lainnya Dukungan berbagai beasiswa dari pemerintah melalui Direktorat Perguruan Tinggi dan Pemerintah Provinsi Kalimantan Timur4 Ancaman Threats Terdapat beberapa lembaga perguruan tinggi lain yang menyelenggarakan Program Magister Ilmu Pemeritahan sejenis yang sudah memiliki reputasi nasional dan internasional yang menawarkan kerjasama dengan pemerintah daerah serta tawaran beasiswa kepada sarjana strata satu yang berprestasi Dukungan berbagai beasiswa dari Direktorat Perguruan Tinggi dan Pemerintah Daerah yang belum dapat dimanfaatkan oleh Program Magister Ilmu Pemerintahan karena baru berdiri dan sesuai ketentuan mendapat Nilai Akreditasi Baik Permendikbud No 7 tahun 2020 Pasal 25 ayat 2 dan Pasal 35 Setelah dipetakan melalui analisis komponen SWOT maka didapat keunggulan Adapun keunggulan Program Magister Ilmu Pemerintahan FISIP Unmul dibandingkan 3 tiga program studi sejenis pada tingkat Perguruan Tinggi nasional danatau internasional yang mencakup dari beberapa aspek yaituProgram Magister Ilmu Pemerintahan FISIP Unmul merupakan satusatunya program Studi di bidang Ilmu Pemerintahan di Provinsi Kalimantan Timur Tenaga Pengajar Program Magister Ilmu Pemerintahan FISIP Unmul telah bergelar S3 dan telah memiliki 2 Orang Guru Besar Kurikulum Program Magister Ilmu Pemerintahan FISIP Unmul telah mengacu kepada Kerangka Kurikulum Nasional Indonesia KKNI yang menekankan Perolehan Pengetahuan Faktual Program Magister Ilmu Pemerintahan FISIP Unmul telah memiliki gedung perkantoran sendiri dan fasilitas yang memadai untuk menunjang proses belajar mengajar seperti ruang perkuliahan ruang baca offline perpustakaan online integrated digital library Fisip Unmul Smart Library Fisip Unmul laboratorium ruang seminar musholla dan internet Dibandingkan dengan kurikulum program studi sejenis Program Magister Ilmu Pemerintahan Fisip Unmul berada dalam Wilayah penyangga Ibu Kota Negara Nusantara yang tentunya akan membawa dampak dan berpengaruh terhadap perubahan pola mindset dan perilaku birokrasi dalam manajemen pemerintahan sehingga kehadiran Program Magister Ilmu Pemerintahan FISIP Unmul sangat tepat"
    score = 2
    return text, score
    #%%
    # X_train = joblib.load('./data/X_train.pkl')
    # Y_train = joblib.load('./data/Y_train.pkl')

    # index_of_2 = np.where(Y_train == 2)[0][0]
    # return X_train[index_of_2], Y_train[index_of_2]
    #%%
    df = pd.read_json('cleaned_dataset.json')
    sample = df[df['nilai'] == 2].sample(1)
    return sample['isiPengusul'].values[0], sample['nilai'].values[0]

def main():
    model, tokenizer = load_model()
    # scaler = joblib.load("target_scaler.pkl")

    rekomendasi = {
        0: """
            Tulisan tentang keunggulan program studi yang Anda susun belum memenuhi kriteria yang diharapkan. Keunikan atau keunggulan program studi perlu disusun berdasarkan perbandingan dengan program studi lain pada tingkat nasional dan mencakup tiga aspek penting: pengembangan keilmuan, kajian capaian pembelajaran, dan kurikulum.
            
            Rekomendasi perbaikan:

            1. Identifikasi setidaknya tiga program studi sejenis pada tingkat nasional sebagai pembanding.
            2. Lakukan analisis perbandingan antara program studi Anda dengan tiga program studi pembanding tersebut dalam hal pengembangan keilmuan. Uraikan bagaimana program studi Anda memiliki fokus atau pendekatan yang berbeda dalam mengembangkan ilmu pengetahuan di bidang terkait.
            3. Bandingkan capaian pembelajaran (learning outcomes) yang diharapkan dari lulusan program studi Anda dengan program studi pembanding. Jelaskan bagaimana program studi Anda membekali lulusan dengan kompetensi atau keterampilan yang unik dan relevan.
            4. Analisis perbandingan kurikulum antara program studi Anda dengan program studi pembanding. Sorot keunikan atau kekuatan kurikulum program studi Anda, seperti mata kuliah khusus, pendekatan pembelajaran inovatif, atau kerja sama industri yang mendukung pengembangan kompetensi mahasiswa.
        """,
        1: """
            Tulisan tentang keunggulan program studi yang Anda susun belum memenuhi kriteria yang diharapkan. Keunikan atau keunggulan program studi perlu disusun berdasarkan perbandingan dengan program studi lain pada tingkat nasional dan mencakup tiga aspek penting: pengembangan keilmuan, kajian capaian pembelajaran, dan kurikulum.
            
            Rekomendasi perbaikan:

            1. Identifikasi setidaknya tiga program studi sejenis pada tingkat nasional sebagai pembanding.
            2. Lakukan analisis perbandingan antara program studi Anda dengan tiga program studi pembanding tersebut dalam hal pengembangan keilmuan. Uraikan bagaimana program studi Anda memiliki fokus atau pendekatan yang berbeda dalam mengembangkan ilmu pengetahuan di bidang terkait.
            3. Bandingkan capaian pembelajaran (learning outcomes) yang diharapkan dari lulusan program studi Anda dengan program studi pembanding. Jelaskan bagaimana program studi Anda membekali lulusan dengan kompetensi atau keterampilan yang unik dan relevan.
            4. Analisis perbandingan kurikulum antara program studi Anda dengan program studi pembanding. Sorot keunikan atau kekuatan kurikulum program studi Anda, seperti mata kuliah khusus, pendekatan pembelajaran inovatif, atau kerja sama industri yang mendukung pengembangan kompetensi mahasiswa.
        """,
        2: """
            Tulisan Anda tentang keunggulan program studi sudah cukup baik karena telah disusun berdasarkan perbandingan dengan tiga program studi pada tingkat nasional dan mencakup tiga aspek: pengembangan keilmuan, kajian capaian pembelajaran, dan kurikulum. Namun, untuk lebih meningkatkan kualitas, Anda dapat memperluas perbandingan pada tingkat internasional.
            
            Rekomendasi peningkatan:

            1. Selain tiga program studi pada tingkat nasional, tambahkan juga perbandingan dengan setidaknya tiga program studi sejenis pada tingkat internasional.
            2. Lakukan analisis perbandingan yang lebih mendalam dalam aspek pengembangan keilmuan. Eksplorasi bagaimana program studi Anda dan program studi pembanding internasional berkontribusi pada kemajuan ilmu pengetahuan di bidang terkait melalui penelitian, publikasi, atau inovasi.
            3. Bandingkan capaian pembelajaran program studi Anda dengan standar internasional atau praktik terbaik global. Identifikasi keunikan atau keunggulan dalam mempersiapkan lulusan untuk karir atau studi lanjut di tingkat internasional.
            4. Analisis kurikulum program studi Anda dengan program studi pembanding internasional. Cari tahu apakah ada fitur unik atau inovatif dalam kurikulum Anda yang belum diadopsi secara luas di tingkat global.
        """,
        3: """
            Tulisan Anda tentang keunggulan program studi sudah sangat baik karena telah disusun berdasarkan perbandingan dengan tiga program studi pada tingkat nasional dan internasional, serta mencakup tiga aspek: pengembangan keilmuan, kajian capaian pembelajaran, dan kurikulum. Untuk mencapai tingkat keunggulan tertinggi, Anda dapat mempertimbangkan apakah program studi Anda merupakan satu-satunya di dunia dalam hal pengembangan keilmuan, capaian pembelajaran, atau kurikulum yang unik.
            
            Rekomendasi peningkatan:

            1. Lakukan riset yang lebih ekstensif untuk mengetahui apakah program studi Anda memiliki pendekatan pengembangan keilmuan yang benar-benar unik dan tidak ditemukan pada program studi lain di dunia. Misalnya, apakah ada fokus penelitian atau metodologi khusus yang hanya diterapkan di program studi Anda.
            2. Periksa apakah capaian pembelajaran program studi Anda mencakup kompetensi atau keterampilan yang sangat spesifik dan tidak ditawarkan oleh program studi lain secara global. Ini bisa berupa kombinasi unik dari pengetahuan, keterampilan, dan sikap yang diharapkan dari lulusan Anda.
            3. Analisis keunikan kurikulum program studi Anda dibandingkan dengan program studi sejenis di seluruh dunia. Cari tahu apakah ada mata kuliah, pendekatan pembelajaran, atau kolaborasi industri yang benar-benar inovatif dan belum diterapkan di tempat lain.
            4. Jika program studi Anda terbukti memiliki keunikan global dalam hal pengembangan keilmuan, capaian pembelajaran, atau kurikulum, fokuskan uraian pada aspek-aspek tersebut untuk menunjukkan keunggulan yang tak tertandingi.
        """
    }

    st.title("Prediksi Kelayakan Butir Keunggulan Program Studi")

    with st.expander("**Panduan Pengisian**", expanded=False):
        st.write("""
            Bagian ini berisi keunggulan atau keunikan program studi yang diusulkan berdasarkan perbandingan 3 (tiga) program studi sejenis pada tingkat nasional dan/atau internasional yang mencakup aspek:
            1. Pengembangan keilmuan
            2. Kajian capaian pembelajaran
            3. Kurikulum program studi sejenis
            ---
            **Catatan:** 
            1. Formulir Pengisian hanya menerima teks untuk memaksimalkan hasil prediksi.
            2. Fitur ini menggunakan teknologi AI yang bisa memberikan hasil yang tidak akurat. Oleh karena itu, hasil yang diberikan tidak dijamin kebenarannya dan sebaiknya selalu dicek kembali dengan keadaan aslinya.
        """)

    text1 = st.text_area("Masukkan teks keunggulan program studi Anda di sini:", height=300, key="text_area", help="Formulir Pengisian hanya menerima teks untuk memaksimalkan hasil prediksi.")
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # text1 = st_quill(html=True, key="quill")
    text2, ref_score = get_reference_text()

    # with open("./data/X_test_cls.pkl", "rb") as f:
    #     X_test = joblib.load(f)[0]
    # with open("./data/Y_test_cls.pkl", "rb") as f:
    #     Y_test = joblib.load(f)[0]

    # print(X_test)
    # print(Y_test)

    if st.button("Prediksi"):
        clean_text1 = preprocess_input(text1)
        # if len(clean_text1.split()) < 100:
        #     st.warning("Teks harus memiliki minimal 100 kata.")
        #     return
        
        clean_text2 = text2

        inputs1 = tokenizer.tokenize(clean_text1)
        # st.write(inputs1[:50])
        indexed_tokens1 = tokenizer.convert_tokens_to_ids(inputs1)
        # st.write(indexed_tokens1)
    
        X = np.empty([1, 512], dtype=np.int32)

        if len(indexed_tokens1) > 512:
            X[0, :512] = indexed_tokens1[:512]
        else:
            X[0, :len(indexed_tokens1)] = indexed_tokens1
            X[0, len(indexed_tokens1):] = 0
        
        x0 = [j for j in X]
        # st.write(len(x0))
        x = torch.LongTensor(x0).to(device)
        # st.write(x)

        inputs2 = tokenizer.tokenize(clean_text2)
        indexed_tokens2 = tokenizer.convert_tokens_to_ids(inputs2)
        X = np.empty([1, 512], dtype=np.int32)

        if len(indexed_tokens2) > 512:
            X[0, :512] = indexed_tokens2[:512]
        else:
            X[0, :len(indexed_tokens2)] = indexed_tokens2
            X[0, len(indexed_tokens2):] = 0
        
        x0 = [j for j in X]
        x2 = torch.LongTensor(x0).to(device)
        
        # print(x)
        
        # inputs1 = tokenizer.encode_plus(
        #     clean_text1,
        #     max_length=512,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt"
        # )

        # st.write(inputs1)
        # st.write(len(inputs1['input_ids']))
        # return


        # inputs2 = tokenizer.encode_plus(
        #     clean_text2,
        #     max_length=512,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt"
        # )

        # with torch.no_grad():
        #     # x0 = [j for j in X_test]
        #     # print(len(x0))
        #     # x = torch.LongTensor([x0]).to(device)
        #     logits = model(x.to(device))
        #     probs = F.softmax(logits, dim=1)
        #     print(probs)
        #     final_pred_score = torch.argmax(probs, dim=1)
        #     final_pred_score = final_pred_score.cpu().numpy().flatten()[0]

        # if final_pred_score < 2:
        #     st.warning("Tidak Memenuhi")
        #     st.warning(f"Skor teks: {final_pred_score}")
        #     with st.expander("Rekomendasi Perbaikan", expanded=False):
        #         st.markdown(rekomendasi[0])
        # else:
        #     st.success("Memenuhi")
        #     st.success(f"Skor teks: {final_pred_score}")
        #     with st.expander("Rekomendasi Peningkatan", expanded=False):
        #         st.markdown(rekomendasi[math.floor(final_pred_score)])

        with torch.no_grad():
            prediction = model(x.to(device), x2.to(device))
            # print("prediction:", prediction)
            relative_score = rescale_tointscore(prediction.cpu().numpy())
            relative_score = relative_score.flatten()[0]
            # print("relative_score:", relative_score)
            # print("reference_score:", ref_score)
            final_pred_score = relative_score + ref_score
            final_pred_score = np.clip(final_pred_score, 0, 4)
        
        final_pred_score = round(final_pred_score, 1)
        if final_pred_score < 2:
            st.warning("Tidak Memenuhi")
            st.warning(f"Skor teks: {final_pred_score}")
            
            with st.expander("Rekomendasi Perbaikan", expanded=False):
                st.write("*Tulisan di dbawah ini hanyalah rekomendasi saja. Jika anda merasa sudah memenuhi seluruh kriteria, anda dapat mengabaikan rekomendasi ini.*")
                st.markdown(rekomendasi[0])
        else:
            st.success("Memenuhi")
            st.success(f"Skor teks: {final_pred_score}")
            with st.expander("Rekomendasi Peningkatan", expanded=False):
                st.write("*Tulisan di dbawah ini hanyalah rekomendasi saja. Jika anda merasa sudah memenuhi seluruh kriteria, anda dapat mengabaikan rekomendasi ini.*")
                st.markdown(rekomendasi[math.floor(final_pred_score)])

if __name__ == "__main__":
    main()