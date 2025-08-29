# import pandas as pd
# import streamlit as st
# import pickle
# import pandas
# import requests


# def fetch_poster(movie_id):
#     response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=17b369f937261ad182053800a18d17d6&language=en-US)'.format(movie_id))
#     data = response.json()
#     print(data)
#     return "http://image.tmdb.org/t/p/w500" + data['poster_path']


# def recommend(movie):
#     movie_index = movies[movies['title'] == movie].index[0]
#     distances = similarity[movie_index]
#     movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

#     recommended_movies = []
#     recommended_movies_posters = []
#     for i in movie_list:
#         df_index = i[0]
#         movie_id = movies.iloc[df_index]['movie_id']
#         recommended_movies.append(movies.iloc[i[0]].title)
#         # fetch poster from API
#        # recommended_movies.append(movies.iloc[df_index].title)
#         recommended_movies_posters.append(fetch_poster(movie_id))
#     return recommended_movies,recommended_movies_posters


# movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
# movies = pd.DataFrame(movies_dict)

# similarity = pickle.load(open('similarity.pkl', 'rb'))

# st.title("Movie Recommending System")

# selected_movie_name = st.selectbox(
# "Enter the Movie",
# movies['title'].values)

# if st.button("Recommend"):
#     names, posters = recommend(selected_movie_name)

#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.text(names[0])
#         st.image(posters[0])

#     with col2:
#         st.text(names[1])
#         st.image(posters[1])

#     with col3:
#         st.text(names[2])
#         st.image(posters[2])

#     with col4:
#         st.text(names[3])
#         st.image(posters[3])

#     with col5:
#         st.text(names[4])
#         st.image(posters[4])
    #     st.text(recommended_movie_names[0])
    #     st.image(recommended_movie_posters[0])
    # with col2:
    #     st.text(recommended_movie_names[1])
    #     st.image(recommended_movie_posters[1])
    #
    # with col3:
    #     st.text(recommended_movie_names[2])
    #     st.image(recommended_movie_posters[2])
    # with col4:
    #     st.text(recommended_movie_names[3])
    #     st.image(recommended_movie_posters[3])
    # with col5:
    #     st.text(recommended_movie_names[4])
    #     st.image(recommended_movie_posters[4])

import pandas as pd
import streamlit as st
import pickle
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math


# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Load your preprocessed movie data (make sure it includes 'tag', 'movie_id', 'title', and 'sentiment' columns)
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)


# Function to clean and normalize text
def clean_text(text):
    return ' '.join(str(text).lower().split())


# Combine tags, genres, and keywords if available for richer context
if 'genres' in movies.columns and 'keywords' in movies.columns:
    movies['combined_tag'] = movies.apply(
        lambda row: ' '.join(row['tag']) + ' ' + ' '.join(row['genres']) + ' ' + ' '.join(row['keywords']), axis=1)
else:
    movies['combined_tag'] = movies['tag']


# Apply cleaning function
movies['combined_tag'] = movies['combined_tag'].apply(clean_text)


# If sentiment scores are not computed yet, compute them on combined_tag
def get_sentiment_scores(text):
    if not isinstance(text, str) or text.strip() == '':
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    return analyzer.polarity_scores(text)


if 'sentiment' not in movies.columns:
    movies['sentiment'] = movies['combined_tag'].apply(get_sentiment_scores)


# Load similarity matrix if you want similarity-based recommendations as well
similarity = pickle.load(open('similarity.pkl', 'rb'))


def fetch_poster(movie_id):
    # Use your valid TMDb API key here
    api_key = '17b369f937261ad182053800a18d17d6'
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
    response = requests.get(url)
    data = response.json()
    if 'poster_path' in data and data['poster_path']:
        return "http://image.tmdb.org/t/p/w500" + data['poster_path']
    else:
        # Return a placeholder image URL if not found
        return "https://via.placeholder.com/500x750?text=No+Image"


def recommend_by_mood(df, mood, top_n=5):
    mood_filters = {
        'bittersweet': {
            'genres': ['drama', 'romance', 'comedy'],
            'keywords': ['heartwarming', 'emotion', 'family', 'relationships'],
            'sentiment_range': (0, 0.3)
        },
        'adrenaline rush': {
            'genres': ['action', 'adventure', 'thriller'],
            'keywords': ['fast-paced', 'exciting', 'intense', 'chase'],
            'sentiment_range': (0.5, 1.0)
        },
        'existential': {
            'genres': ['drama', 'mystery', 'sci-fi'],
            'keywords': ['philosophy', 'isolated', 'identity', 'existence'],
            'sentiment_range': (-1.0, 0)
        },
        'cozy nostalgia': {
            'genres': ['family', 'comedy', 'animation', 'fantasy'],
            'keywords': ['nostalgic', 'warm', 'comforting', 'classic'],
            'sentiment_range': (0.3, 1.0)
        }
    }

    if mood not in mood_filters:
        return df[['title', 'movie_id']].head(top_n)

    filters = mood_filters[mood]
    low, high = filters['sentiment_range']

    # Filter by sentiment compound score range
    filtered = df[df['sentiment'].apply(lambda x: low <= x['compound'] <= high)]

    # Filter by genres/keywords in combined_tag (case-insensitive)
    genre_mask = filtered['combined_tag'].apply(
        lambda text: any(genre in text for genre in filters['genres'])
    )
    keyword_mask = filtered['combined_tag'].apply(
        lambda text: any(keyword in text for keyword in filters['keywords'])
    )

    filtered = filtered[genre_mask | keyword_mask]

    return filtered[['title', 'movie_id']].head(top_n)


st.title("Personalized Emotion Based Movie Picks")


selected_mood = st.selectbox(
    "Select Your Mood",
    ["bittersweet", "adrenaline rush", "existential", "cozy nostalgia"],
)


num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)


if st.button("Recommend"):
    recommendations = recommend_by_mood(movies, selected_mood, top_n=num_recommendations)

    recommended_movies = recommendations['title'].values
    recommended_movie_ids = recommendations['movie_id'].values
    recommended_posters = [fetch_poster(m_id) for m_id in recommended_movie_ids]

    # Number of movies per row
    movies_per_row = 3
    total_movies = len(recommended_movies)
    num_rows = math.ceil(total_movies / movies_per_row)

    for row in range(num_rows):
        start_idx = row * movies_per_row
        end_idx = start_idx + movies_per_row
        cols = st.columns(movies_per_row)

        for idx, col in enumerate(cols):
            movie_idx = start_idx + idx
            if movie_idx < total_movies:
                col.image(
                    recommended_posters[movie_idx],
                    use_container_width=True
                )
                col.markdown(
                    f"<div style='text-align:center; font-weight:bold; font-size:18px; max-width:120px; word-break:break-word; margin:auto;'>{recommended_movies[movie_idx]}</div>",
                    unsafe_allow_html=True
                )


    





