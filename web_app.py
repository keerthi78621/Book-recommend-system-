# Streamlit is required for web interface
# Install: pip install streamlit
# Run: streamlit run web_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os

# Page configuration
st.set_page_config(page_title="Book Recommender System", page_icon="📚", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .book-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .author {
        font-size: 16px;
        color: #666;
    }
    .rating {
        font-size: 18px;
        color: #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Load or generate data
@st.cache_data
def load_data():
    """Load or generate sample data."""
    # Always use sample data for reliable demo
    books = pd.DataFrame({
        'book_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'The Hobbit',
            'The Fellowship of the Ring',
            'The Two Towers',
            'The Return of the King',
            "Harry Potter and the Sorcerer's Stone",
            "Harry Potter and the Chamber of Secrets",
            'The Lightning Thief',
            'The Sea of Monsters',
            'The Lion, the Witch and the Wardrobe',
            "The Magician's Nephew"
        ],
        'authors': [
            'J.R.R. Tolkien', 'J.R.R. Tolkien', 'J.R.R. Tolkien', 'J.R.R. Tolkien',
            'J.K. Rowling', 'J.K. Rowling', 'Rick Riordan', 'Rick Riordan',
            'C.S. Lewis', 'C.S. Lewis'
        ],
        'average_rating': [4.26, 4.45, 4.43, 4.45, 4.47, 4.42, 4.28, 4.25, 4.21, 4.22],
        'ratings_count': [2100000, 1900000, 1800000, 1750000, 2000000, 1950000, 900000, 850000, 1200000, 800000]
    })
    
    ratings = pd.DataFrame({
        'user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 
                   6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10],
        'book_id': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 
                   1, 2, 5, 6, 3, 4, 7, 8, 1, 5, 7, 9, 2, 6, 8, 10, 1, 2, 5, 10],
        'rating': [5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 5, 4, 4, 
                  4, 5, 4, 4, 5, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 5, 5, 4, 5]
    })
    
    # Convert to float to avoid dtype issues
    ratings['rating'] = ratings['rating'].astype(float)
    
    return books, ratings

# Build model
@st.cache_resource
def build_model(books, ratings):
    """Build the recommendation model."""
    # Create user-item matrix
    unique_books = ratings['book_id'].unique()
    unique_users = ratings['user_id'].unique()
    
    book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
    idx_to_book_id = {idx: book_id for book_id, idx in book_id_to_idx.items()}
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    rows = ratings['book_id'].map(book_id_to_idx).astype(float)
    cols = ratings['user_id'].map(user_id_to_idx).astype(float)
    values = ratings['rating'].values.astype(float)
    
    user_item_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(unique_books), len(unique_users))
    )
    
    # Compute similarity
    book_vectors = user_item_matrix.T
    similarity_matrix = cosine_similarity(book_vectors)
    
    return similarity_matrix, book_id_to_idx, idx_to_book_id, books

# Get recommendations
def get_recommendations(book_title, books, similarity_matrix, book_id_to_idx, idx_to_book_id, top_n=5):
    """Get book recommendations."""
    book_match = books[books['title'].str.contains(book_title, case=False, na=False)]
    
    if book_match.empty:
        return []
    
    book = book_match.iloc[0]
    book_id = book['book_id']
    
    if book_id not in book_id_to_idx:
        return []
    
    idx = book_id_to_idx[book_id]
    sim_scores = similarity_matrix[idx]
    
    top_indices = np.argsort(sim_scores)[::-1][:top_n + 1]
    
    recommendations = []
    for i in top_indices:
        if i != idx:
            other_book_id = idx_to_book_id[i]
            book_info = books[books['book_id'] == other_book_id].iloc[0]
            
            recommendations.append({
                'title': book_info['title'],
                'authors': book_info['authors'],
                'average_rating': book_info['average_rating'],
                'similarity_score': float(sim_scores[i])
            })
    
    return recommendations[:top_n]

# Main app
def main():
    """Main Streamlit app."""
    st.title("📚 Book Recommender System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This is a **Book Recommender System** using:
    - Item-Based Collaborative Filtering
    - Cosine Similarity Algorithm
    
    Enter a book you liked, and get personalized recommendations!
    """)
    
    st.sidebar.title("Dataset")
    books, ratings = load_data()
    st.sidebar.write(f"📚 Books: {len(books)}")
    st.sidebar.write(f"⭐ Ratings: {len(ratings)}")
    st.sidebar.write(f"👥 Users: {ratings['user_id'].nunique()}")
    
    # Build model
    similarity_matrix, book_id_to_idx, idx_to_book_id, books = build_model(books, ratings)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Find Similar Books")
        book_title = st.text_input("Enter a book you liked:", placeholder="e.g., The Hobbit")
        
        if st.button("Get Recommendations", type="primary"):
            if book_title:
                recommendations = get_recommendations(
                    book_title, books, similarity_matrix, 
                    book_id_to_idx, idx_to_book_id, top_n=5
                )
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations for '{book_title}'")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"""
                        <div class="book-card">
                            <div class="title">{i}. {rec['title']}</div>
                            <div class="author">✍️ {rec['authors']}</div>
                            <div class="rating">⭐ {rec['average_rating']}/5 | Similarity: {rec['similarity_score']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Book not found: '{book_title}'")
                    st.write("Try these books:")
                    for _, book in books.head(5).iterrows():
                        st.write(f"  - {book['title']}")
    
    with col2:
        st.subheader("📖 Popular Books")
        popular = books.nlargest(5, 'ratings_count')
        for _, book in popular.iterrows():
            st.markdown(f"""
            <div class="book-card">
                <div class="title" style="font-size: 16px;">{book['title'][:25]}...</div>
                <div class="rating">⭐ {book['average_rating']}/5</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show sample recommendations
    st.markdown("---")
    st.subheader("🎯 Sample Recommendations")
    
    sample_books = ['The Hobbit', 'Harry Potter', 'The Lightning Thief']
    
    for sample in sample_books:
        with st.expander(f"Recommendations for '{sample}'"):
            recs = get_recommendations(sample, books, similarity_matrix, 
                                      book_id_to_idx, idx_to_book_id, top_n=3)
            for rec in recs:
                st.write(f"📖 **{rec['title']}** by {rec['authors']} (⭐{rec['average_rating']})")

if __name__ == "__main__":
    main()
