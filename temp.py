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
# from nltk.corpus import wordnet as wn
import altair as alt
from wordcloud import WordCloud
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd

#%%
# Load the cleaned JSON data into a pandas DataFrame
# df = pd.read_json('cleaned_html_dataset_all.json')
df = pd.read_json('cleaned_augmented_dataset_all.json')
df = df.sort_values(by='usulanId').reset_index(drop=True)
print(len(df))
print('selesai')
exit()
print('out of bounds')

df = pd.read_json('cleaned_html_dataset_all.json')
mutable_df = df.copy()
print(len(df))
txt_df = pd.read_csv('eda_label_0.txt', sep='\t')
txt_df.columns = ['usulanId', 'isiPengusul']
txt_df = txt_df.sort_values(by='usulanId').reset_index(drop=True)
# print(txt_df[txt_df['usulanId'] == 1].reset_index(drop=True).head(20))
# exit()
# print(txt_df.columns)
# print(txt_df.head(20))

# Group by 'usulanId' and remove the first occurrence in each group
# txt_df = txt_df.groupby('usulanId').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
# print(txt_df.head(20))

count = 0
last_usulan_id = -1
# Loop through each row
for index, row in txt_df.iterrows():
    if row['usulanId'] != last_usulan_id:
        count = 0

    total = len(txt_df[txt_df['usulanId'] == row['usulanId']])
    threshold = total // 2
    print('Threshold:', threshold)
    # exit()
    data = df[df['usulanId'] == row['usulanId']]

    if not data.empty:
        # print('isOK:', len(data) > 1 and count > threshold)
        # exit()
        loc = 1 if len(data) > 1 and count > threshold else 0
        new_row = data.iloc[loc].copy()
        # print('new_row:', new_row['evaluatorId'])
        new_row['isiPengusul'] = row['isiPengusul']
        # print(new_row)
        mutable_df = pd.concat([mutable_df, pd.DataFrame([new_row])], ignore_index=True)
        count += 1
        last_usulan_id = row['usulanId']

# Read the txt file with tab separator
txt_df = pd.read_csv('eda_label_4.txt', sep='\t')
# Assign column names to the dataframe
txt_df.columns = ['usulanId', 'isiPengusul']  # Replace with actual column names
# print(txt_df.columns)
# print(txt_df.head(20))

# Group by 'usulanId' and remove the first occurrence in each group
# txt_df = txt_df.groupby('usulanId').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
# print(txt_df.head(20))

count = 0
last_usulan_id = -1
# Loop through each row
for index, row in txt_df.iterrows():
    if row['usulanId'] != last_usulan_id:
        count = 0

    total = len(txt_df[txt_df['usulanId'] == row['usulanId']])
    threshold = total // 2
    data = df[df['usulanId'] == row['usulanId']]
    
    if not data.empty:
        loc = 1 if len(data) > 1 and count > threshold else 0
        new_row = data.iloc[loc].copy()
        new_row['isiPengusul'] = row['isiPengusul']
        # print(new_row)
        mutable_df = pd.concat([mutable_df, pd.DataFrame([new_row])], ignore_index=True)
        count += 1
        last_usulan_id = row['usulanId']

# Export the updated dataframe to a new JSON file
mutable_df.to_json('cleaned_augmented_dataset_all.json', orient='records')
print(len(mutable_df))
print('exported')
# print(df[df['usulanId'] == 2].iloc[0])
# print(mutable_df[mutable_df['usulanId'] == 2].head(20))

#%%
df = pd.read_json('cleaned_dataset_all.json')

#%%
df.info()

#%%
extracted = df[['usulanId', 'nilai', 'isiPengusul']]

#%%
# Add a new column 'word_count' to the dataframe which contains the number of words in 'isiPengusul'
extracted['word_count'] = extracted['isiPengusul'].apply(lambda x: len(str(x).split()))

#%%
extracted = extracted[extracted['word_count'] > 3]

# #%%
# extracted[extracted['nilai'] == 2]['word_count'].quantile(0.5)

#%%
extracted[(extracted['nilai'] == 2) & (extracted['word_count'] == 6)]['isiPengusul'][7783]

#%%
extracted = 

#%%
extracted.head()

#%%
# Group by 'usulanId' and calculate the mean of 'nilai', also take the first 'isiPengusul' in each group
grouped_df = extracted.groupby('usulanId').agg({'nilai': 'mean', 'isiPengusul': 'first'}).reset_index()

# Display the grouped dataframe
grouped_df.head()

#%%
grouped_df.info()

#%%
export = grouped_df[['nilai', 'isiPengusul']]

# Export the 'export' dataframe to a JSON file
export.to_json('grouped_mean_dataset.json', orient='records')