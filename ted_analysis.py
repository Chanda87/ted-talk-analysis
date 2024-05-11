import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
ted_mn = pd.read_csv('ted_main.csv')
transcripts = pd.read_csv('transcripts.csv')

# Combine data
merged_data = pd.merge(ted_mn, transcripts, on='url', how='inner')

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['transcript'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    matching_rows = merged_data[merged_data['title'] == title]
    
    if matching_rows.empty:
        return []  # return an empty list if no match is found
    
    idx = matching_rows.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations
    talk_indices = [i[0] for i in sim_scores]
    return merged_data['title'].iloc[talk_indices]

# Streamlit UI
st.title('TED Talk Recommender')

# Select TED talk title from dropdown
selected_title = st.selectbox('Select TED Talk Title', ted_mn['title'])

# Display selected TED talk
st.subheader('Selected TED Talk:')
st.write(selected_title)

# Get recommendations for selected TED talk
recommendations = get_recommendations(selected_title)

# Display recommendations
st.subheader('Recommended TED Talks:')
if recommendations:
    st.write(recommendations)
else:
    st.write('No recommendations found for this talk.')
