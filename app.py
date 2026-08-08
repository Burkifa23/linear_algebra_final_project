import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recommender import get_k_nearest_neighbours, generate_recommendations

# --- 1. Page Config & Caching ---
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Use st.cache_data so the matrix isn't reloaded on every button click
@st.cache_data
def load_data():
    return pd.read_csv('data_cleaned.csv', index_col=0)

df = load_data()

# --- 2. Sidebar Controls ---
st.sidebar.header("Recommendation Parameters")

# Create a dropdown for the Target User ID based on the matrix index
target_user = st.sidebar.selectbox("Select Target User ID:", df.index)

# Slider to adjust the 'k' nearest neighbours
k_neighbours = st.sidebar.slider("Number of Neighbours (k):", min_value=1, max_value=10, value=5)

# Slider to limit the number of recommendations returned
num_recs = st.sidebar.slider("Max Recommendations:", min_value=1, max_value=10, value=5)

# --- 3. Main Dashboard Layout ---
st.title("Collaborative Filtering Recommendation Engine")
st.markdown("This system uses **Euclidean distance** across a matrix to find peer users with similar geometric preference patterns.")

# Split the layout into two columns for results
col1, col2 = st.columns([2, 1])

if st.sidebar.button("Generate Recommendations"):
    
    # Execute the linear algebra engine
    neighbours = get_k_nearest_neighbours(target_user, df, k=k_neighbours)
    recs = generate_recommendations(target_user, neighbours, df)
    
    with col1:
        st.subheader(f"Top {num_recs} Movies for User {target_user}")
        
        if not recs:
            st.warning("No recommendations available for this user.")
        else:
            # Format the output into a clean DataFrame for the UI
            rec_df = pd.DataFrame(recs[:num_recs], columns=["Movie", "Predicted Rating"])
            st.dataframe(rec_df, use_container_width=True)
            
    with col2:
        st.subheader("Trusted Peers")
        st.markdown("Nearest Neighbours by Euclidean Distance:")
        for n_id, dist in neighbours:
            st.write(f"**User {n_id}** (Distance: {dist:.2f})")

# --- 4. Global Analysis Section ---
st.divider()
st.subheader("Global Average Ratings")

# Compute column means across the matrix
item_averages = df.replace(0, pd.NA).mean().sort_values(ascending=False).head(15)

# Render the bar chart directly in Streamlit using Matplotlib
fig, ax = plt.subplots(figsize=(10, 4))
item_averages.plot(kind='bar', color='#4A90E2', edgecolor='black', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Rating')
st.pyplot(fig)