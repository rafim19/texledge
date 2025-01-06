#%%
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import html
import re
import emoji
import nltk
import altair as alt
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import plotly.express as px
import pickle

#%%
st.set_page_config(page_title="Evaluasi Model", page_icon="📚")

st.title("📊 Evaluasi Model")
st.markdown("""

""", unsafe_allow_html=True)

st.markdown("""
Halaman ini memuat hasil evaluasi dari model <em>Neural Pairwise Contrastive Learning</em> (NPCR) yang sudah di-<em>tuning</em> melalui beberapa proses, yaitu:

<style>
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: center;
}
.selected {
    background-color: #8ed973;
    color: #006100;
}
</style>

<ol>
        <li>
            Pencarian <em>loss function</em> yang terbaik
            <table>
                <tr>
                    <th rowspan="2"><em>Loss Function</em></th>
                    <th rowspan="2"><em>Batch Size</em></th>
                    <th rowspan="2"><em>Dropout</em></th>
                    <th rowspan="2">Token Maksimum</th>
                    <th colspan="4">Metrik Evaluasi</th>
                </tr>
                <tr>
                    <th>MAE</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th><em>R-Squared</em></th>
                </tr>
                <tr>
                    <td>MSE</td>
                    <td rowspan="3">8</td>
                    <td rowspan="3">0.5</td>
                    <td rowspan="3">512</td>
                    <td>0.36</td>
                    <td>0.27</td>
                    <td>0.52</td>
                    <td>0.78</td>
                </tr>
                <tr>
                    <td>Weighted MSE</td>
                    <td>0.38</td>
                    <td>0.29</td>
                    <td>0.54</td>
                    <td>0.77</td>
                </tr>
                <tr class="selected">
                    <td style="font-weight: bold">MAE</td>
                    <td>0.33</td>
                    <td>0.25</td>
                    <td>0.50</td>
                    <td>0.80</td>
                </tr>
            </table>
        </li>
        <li>
            Pencarian nilai <em>batch size</em> yang terbaik
            <table>
                <tr>
                    <th rowspan="2">Loss Function</em></th>
                    <th rowspan="2">Batch Size</em></th>
                    <th rowspan="2">Dropout</em></th>
                    <th rowspan="2">Token Maksimum</th>
                    <th colspan="4">Metrik Evaluasi</th>
                </tr>
                <tr>
                    <th>MAE</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th><em>R-Squared</em></th>
                </tr>
                <tr>
                    <td rowspan="5">MAE</td>
                    <td>12</td>
                    <td rowspan="5">0.5</td>
                    <td rowspan="5">512</td>
                    <td>0.34</td>
                    <td>0.25</td>
                    <td>0.50</td>
                    <td>0.79</td>
                </tr>
                <tr>
                    <td>10</td>
                    <td>0.34</td>
                    <td>0.25</td>
                    <td>0.50</td>
                    <td>0.79</td>
                </tr>
                <tr class="selected">
                    <td style="font-weight: bold">8</td>
                    <td>0.33</td>
                    <td>0.25</td>
                    <td>0.50</td>
                    <td>0.80</td>
                </tr>
                <tr>
                    <td>6</td>
                    <td>0.35</td>
                    <td>0.27</td>
                    <td>0.52</td>
                    <td>0.79</td>
                </tr>
                <tr>
                    <td>4</td>
                    <td>0.36</td>
                    <td>0.27</td>
                    <td>0.52</td>
                    <td>0.78</td>
                </tr>
            </table>
        </li>
        <li>
            Pencarian nilai <em>dropout</em> yang terbaik
            <table>
                <tr>
                    <th rowspan="2"><em>Loss Function</em></th>
                    <th rowspan="2"><em>Batch Size</em></th>
                    <th rowspan="2"><em>Dropout</em></th>
                    <th rowspan="2">Token Maksimum</th>
                    <th colspan="4">Metrik Evaluasi</th>
                </tr>
                <tr>
                    <th>MAE</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th><em>R-Squared</em></th>
                </tr>
                <tr>
                    <td rowspan="3">MAE</td>
                    <td rowspan="3">8</td>
                    <td>0.7</td>
                    <td rowspan="3">512</td>
                    <td>0.35</td>
                    <td>0.26</td>
                    <td>0.51</td>
                    <td>0.79</td>
                </tr>
                <tr class="selected">
                    <td style="font-weight: bold">0.5</td>
                    <td>0.33</td>
                    <td>0.25</td>
                    <td>0.50</td>
                    <td>0.80</td>
                </tr>
                <tr>
                    <td>0.3</td>
                    <td>0.35</td>
                    <td>0.27</td>
                    <td>0.52</td>
                    <td>0.79</td>
                </tr>
            </table>
        </li>
        <li>
            Pencarian nilai token maksimum yang terbaik
            <table>
                <tr>
                    <th rowspan="2"><em>Loss Function</em></th>
                    <th rowspan="2"><em>Batch Size</em></th>
                    <th rowspan="2"><em>Dropout</em></th>
                    <th rowspan="2">Token Maksimum</th>
                    <th colspan="4">Metrik Evaluasi</th>
                </tr>
                <tr>
                    <th>MAE</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th><em>R-Squared</em></th>
                </tr>
                <tr>
                    <td rowspan="3">MAE</td>
                    <td rowspan="3">8</td>
                    <td rowspan="3">0.5</td>
                    <td class="selected" style="font-weight: bold">512</td>
                    <td class="selected">0.33</td>
                    <td class="selected">0.25</td>
                    <td class="selected">0.50</td>
                    <td class="selected">0.80</td>
                </tr>
                <tr>
                    <td>256</td>
                    <td>0.34</td>
                    <td>0.26</td>
                    <td>0.51</td>
                    <td>0.79</td>
                </tr>
                <tr>
                    <td>128</td>
                    <td>0.35</td>
                    <td>0.26</td>
                    <td>0.51</td>
                    <td>0.79</td>
                </tr>
            </table>
        </li>
</ol>
            
Dari proses tersebut didapatkan bahwa model dengan <em>loss function</em> MAE, <em>batch size</em> 8, <em>dropout</em> 0.5, dan token maksimum 512 memberikan hasil evaluasi terbaik. Model ini memiliki nilai MAE sebesar 0.33, MSE sebesar 0.25, RMSE sebesar 0.50, dan <em>R-Squared</em> sebesar 0.80.
""", unsafe_allow_html=True)

st.markdown("""
### Performa Model Terbaik

<table>
    <tr>
        <th style="text-align: center;">Parameter</th>
        <th style="text-align: center;">Nilai</th>
    </tr>
    <tr>
        <td><em>Loss Function</em></td>
        <td>MAE</td>
    </tr>
    <tr>
        <td><em>Batch Size</em></td>
        <td>8</td>
    </tr>
    <tr>
        <td><em>Dropout</em></td>
        <td>0.5</td>
    </tr>
    <tr>
        <td>Token Maksimum</td>
        <td>512</td>
    </tr>
    <tr>
        <td>MAE</td>
        <td>0.33</td>
    </tr>
    <tr>
        <td>MSE</td>
        <td>0.25</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>0.50</td>
    </tr>
    <tr>
        <td><em>R-Squared</em></td>
        <td>0.80</td>
    </tr>
</table>
""", unsafe_allow_html=True)

#%%
with open('./evaluation/test_predictions_history.pickle', 'rb') as file:
    test_loader = pickle.load(file)[0]
    test_loader = np.array(test_loader)
    test_loader = np.clip(test_loader, 0, 4).flatten()
with open('./evaluation/test_ground_truth.pickle', 'rb') as file:
    ground_truth = pickle.load(file)
    ground_truth = np.array(ground_truth).flatten()

# Create a DataFrame for the predictions and ground truth
df = pd.DataFrame({
    'Predictions': test_loader,
    'Ground Truth': ground_truth
})

# Create tabs for scatter plot and density heatmap
tab1, tab2 = st.tabs(["Scatter Plot", "Density Heatmap"])

with tab1:
    st.write("Sumbu x (Horizontal) menunjukkan nilai aktual, sedangkan sumbu y (Vertikal) menunjukkan nilai yang diprediksi oleh model")
    scatter_fig = px.scatter(df, x='Ground Truth', y='Predictions', title='Nilai Prediksi vs Nilai Aktual',
                             labels={'Ground Truth': 'Nilai Aktual', 'Predictions': 'Nilai Prediksi'},
                             opacity=0.6)
    st.plotly_chart(scatter_fig)

with tab2:
    st.write("Sumbu x (Horizontal) menunjukkan nilai aktual, sedangkan sumbu y (Vertikal) menunjukkan nilai yang diprediksi oleh model")
    st.write("Hasil prediksi yang memiliki frekuensi lebih tinggi akan memiliki warna yang lebih terang")
    fig = px.density_heatmap(df, x='Ground Truth', y='Predictions', title='Nilai Prediksi vs Nilai Aktual',
                             labels={'Nilai Aktual': 'Ground Truth', 'Predictions': 'Nilai Prediksi'},
                             nbinsx=50, nbinsy=50, color_continuous_scale='Cividis')
    st.plotly_chart(fig)

# # Create a scatter plot for comparing predictions and ground truth
# scatter_fig = px.scatter(df, x='Ground Truth', y='Predictions', title='Scatter Plot of Predictions vs Ground Truth',
#                          labels={'Ground Truth': 'Nilai Aktual', 'Predictions': 'Nilai Prediksi'},
#                          opacity=0.6)

# # Show the scatter plot in Streamlit
# st.plotly_chart(scatter_fig)

# # Create an interactive scatter plot using Plotly
# fig = px.density_heatmap(df, x='Ground Truth', y='Predictions', title='Nilai Prediksi vs Nilai Aktual',
#                          labels={'Nilai Aktual': 'Ground Truth', 'Predictions': 'Nilai Prediksi'},
#                          nbinsx=50, nbinsy=50, color_continuous_scale='Cividis')

# # Show the plot in Streamlit
# st.plotly_chart(fig)