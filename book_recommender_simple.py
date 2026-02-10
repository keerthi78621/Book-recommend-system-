#!/usr/bin/env python3
"""
Book Recommender System - Complete Implementation
=================================================
A machine learning-based book recommendation system using 
Item-Based Collaborative Filtering with Cosine Similarity.

This is a simplified, single-file version for easy execution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("BOOK RECOMMENDER SYSTEM")
    print("=" * 70)
    print("Item-Based Collaborative Filtering with Cosine Similarity")
    print("=" * 70)


def load_or_generate_data():
    """Load data or generate sample data if files not found."""
    DATA_DIR = 'data'
    BOOKS_FILE = f'{DATA_DIR}/books.csv'
    RATINGS_FILE = f'{DATA_DIR}/ratings.csv'
    
    if not os.path.exists(BOOKS_FILE) or not os.path.exists(RATINGS_FILE):
        print("\nGenerating sample data...")
        
        # Create sample books
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
        
        # Create sample ratings with structure
        ratings = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 
                       6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10],
            'book_id': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 
                       1, 2, 5, 6, 3, 4, 7, 8, 1, 5, 7, 9, 2, 6, 8, 10, 1, 2, 5, 10],
            'rating': [5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 5, 4, 4, 
                      4, 5, 4, 4, 5, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 5, 5, 4, 5]
        })
        
        print("Sample data created!")
        return books, ratings
    
    # Load actual dataset
    books = pd.read_csv(BOOKS_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    print("Dataset loaded!")
    return books, ratings


def preprocess_data(books, ratings):
    """Clean and preprocess data."""
    print("\n" + "-" * 40)
    print("Preprocessing Data...")
    print("-" * 40)
    
    # Clean books
    books_clean = books.copy()
    books_clean['title'] = books_clean['title'].fillna('Unknown Title')
    books_clean['authors'] = books_clean['authors'].fillna('Unknown Author')
    books_clean['average_rating'] = books_clean['average_rating'].fillna(
        books_clean['average_rating'].median()
    )
    
    # Remove duplicates
    original = len(books_clean)
    books_clean = books_clean.drop_duplicates(subset=['title', 'authors'], keep='first')
    print(f"Books: {original} -> {len(books_clean)} (removed {original - len(books_clean)} duplicates)")
    
    # Clean ratings
    ratings_clean = ratings.copy()
    ratings_clean = ratings_clean[
        (ratings_clean['rating'] >= 1) & 
        (ratings_clean['rating'] <= 5)
    ]
    original = len(ratings_clean)
    ratings_clean = ratings_clean.drop_duplicates(subset=['user_id', 'book_id'], keep='first')
    print(f"Ratings: {original:,} -> {len(ratings_clean):,} (cleaned)")
    
    return books_clean, ratings_clean


def create_user_item_matrix(ratings):
    """Create user-item rating matrix."""
    print("\n" + "-" * 40)
    print("Creating User-Item Matrix...")
    print("-" * 40)
    
    unique_books = ratings['book_id'].unique()
    unique_users = ratings['user_id'].unique()
    
    print(f"Unique Books: {len(unique_books)}")
    print(f"Unique Users: {len(unique_users)}")
    
    # Create mappings
    book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
    idx_to_book_id = {idx: book_id for book_id, idx in book_id_to_idx.items()}
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    # Map book_id and user_id to consecutive indices
    rows = ratings['book_id'].map(book_id_to_idx)
    cols = ratings['user_id'].map(user_id_to_idx)
    values = ratings['rating'].values
    
    user_item_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(unique_books), len(unique_users))
    )
    
    print(f"Matrix Shape: {user_item_matrix.shape}")
    sparsity = (1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100
    print(f"Sparsity: {sparsity:.2f}%")
    
    return user_item_matrix, book_id_to_idx, idx_to_book_id


def compute_similarity(user_item_matrix):
    """Compute cosine similarity between books."""
    print("\n" + "-" * 40)
    print("Computing Cosine Similarity...")
    print("-" * 40)
    
    book_vectors = user_item_matrix.T
    similarity_matrix = cosine_similarity(book_vectors)
    
    print(f"Similarity Matrix: {similarity_matrix.shape}")
    
    return similarity_matrix


def get_recommendations(book_title, books_df, similarity_matrix, 
                       book_id_to_idx, idx_to_book_id, top_n=5):
    """Get book recommendations based on a given book."""
    
    # Find the book
    book_match = books_df[
        books_df['title'].str.contains(book_title, case=False, na=False)
    ]
    
    if book_match.empty:
        print(f"Book not found: '{book_title}'")
        return []
    
    book = book_match.iloc[0]
    book_id = book['book_id']
    
    if book_id not in book_id_to_idx:
        print(f"Book '{book_title}' has no ratings")
        return []
    
    idx = book_id_to_idx[book_id]
    sim_scores = similarity_matrix[idx]
    
    # Get top N+1 similar books
    top_indices = np.argsort(sim_scores)[::-1][:top_n + 1]
    
    recommendations = []
    for i in top_indices:
        if i != idx:
            other_book_id = idx_to_book_id[i]
            book_info = books_df[books_df['book_id'] == other_book_id].iloc[0]
            
            recommendations.append({
                'title': book_info['title'],
                'authors': book_info['authors'],
                'average_rating': book_info['average_rating'],
                'similarity_score': float(sim_scores[i])
            })
    
    return recommendations


def display_recommendations(recommendations, book_title):
    """Display recommendations in a formatted way."""
    print("\n" + "=" * 70)
    print(f"BOOK RECOMMENDATIONS FOR: '{book_title}'")
    print("=" * 70)
    
    if not recommendations:
        print("No recommendations found.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Book: {rec['title']}")
        print(f"   Author: {rec['authors']}")
        print(f"   Rating: {rec['average_rating']}/5")
        print(f"   Similarity: {rec['similarity_score']:.3f}")


def create_visualizations(books, ratings):
    """Create data visualizations."""
    print("\n" + "-" * 40)
    print("Creating Visualizations...")
    print("-" * 40)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Rating Distribution
    ax1 = axes[0, 0]
    rating_counts = ratings['rating'].value_counts().sort_index()
    ax1.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='white')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Ratings', fontweight='bold')
    for bar, count in zip(ax1.patches, rating_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    # 2. Average Ratings per Book
    ax2 = axes[0, 1]
    book_avg = ratings.groupby('book_id')['rating'].mean()
    ax2.hist(book_avg, bins=10, color='coral', edgecolor='white')
    ax2.set_xlabel('Average Rating')
    ax2.set_ylabel('Number of Books')
    ax2.set_title('Average Book Ratings', fontweight='bold')
    
    # 3. Top Books by Ratings
    ax3 = axes[1, 0]
    book_counts = ratings.groupby('book_id').agg({'rating': ['mean', 'count']}).reset_index()
    book_counts.columns = ['book_id', 'avg_rating', 'num_ratings']
    popular = book_counts.merge(books, on='book_id').nlargest(10, 'num_ratings')
    y_pos = np.arange(len(popular))
    ax3.barh(y_pos, popular['num_ratings'], color='seagreen')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([t[:20] + '...' if len(t) > 20 else t for t in popular['title']])
    ax3.invert_yaxis()
    ax3.set_xlabel('Number of Ratings')
    ax3.set_title('Top 10 Books by Ratings', fontweight='bold')
    
    # 4. User Activity
    ax4 = axes[1, 1]
    user_counts = ratings.groupby('user_id').size()
    ax4.hist(user_counts, bins=10, color='purple', alpha=0.7, edgecolor='white')
    ax4.axvline(user_counts.mean(), color='red', linestyle='--', label=f'Mean: {user_counts.mean():.1f}')
    ax4.set_xlabel('Ratings per User')
    ax4.set_ylabel('Number of Users')
    ax4.set_title('User Activity Distribution', fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('output/visualizations.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    print("Saved: output/visualizations.png")


def main():
    """Main execution function."""
    print_header()
    
    try:
        # Step 1: Load Data
        books, ratings = load_or_generate_data()
        
        print("\nDataset Summary:")
        print(f"  Books: {len(books):,}")
        print(f"  Ratings: {len(ratings):,}")
        print(f"  Users: {ratings['user_id'].nunique():,}")
        print(f"  Average Rating: {ratings['rating'].mean():.2f}")
        
        # Step 2: Preprocess Data
        books_clean, ratings_clean = preprocess_data(books, ratings)
        
        # Step 3: Create User-Item Matrix
        user_item_matrix, book_id_to_idx, idx_to_book_id = create_user_item_matrix(ratings_clean)
        
        # Step 4: Compute Similarity
        similarity_matrix = compute_similarity(user_item_matrix)
        
        # Step 5: Create Visualizations
        create_visualizations(books_clean, ratings_clean)
        
        # Step 6: Get Recommendations
        print("\n" + "=" * 70)
        print("TESTING RECOMMENDATIONS")
        print("=" * 70)
        
        # Test with The Hobbit
        recommendations = get_recommendations(
            book_title="The Hobbit",
            books_df=books_clean,
            similarity_matrix=similarity_matrix,
            book_id_to_idx=book_id_to_idx,
            idx_to_book_id=idx_to_book_id,
            top_n=5
        )
        display_recommendations(recommendations, "The Hobbit")
        
        # Test with Harry Potter
        recommendations = get_recommendations(
            book_title="Harry Potter",
            books_df=books_clean,
            similarity_matrix=similarity_matrix,
            book_id_to_idx=book_id_to_idx,
            idx_to_book_id=idx_to_book_id,
            top_n=5
        )
        display_recommendations(recommendations, "Harry Potter")
        
        # Interactive mode
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("Available books:")
        for i, (_, book) in enumerate(books_clean.head(10).iterrows(), 1):
            print(f"  {i}. {book['title']}")
        
        while True:
            print("\n" + "-" * 40)
            user_input = input("Enter a book title (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the Book Recommender System!")
                break
            
            if not user_input:
                continue
            
            recommendations = get_recommendations(
                book_title=user_input,
                books_df=books_clean,
                similarity_matrix=similarity_matrix,
                book_id_to_idx=book_id_to_idx,
                idx_to_book_id=idx_to_book_id,
                top_n=10
            )
            
            if recommendations:
                display_recommendations(recommendations, user_input)
            else:
                print("No recommendations found. Try another book.")
        
        print("\n" + "=" * 70)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download dataset or use sample data")
        print("3. Run: python book_recommender_simple.py")


if __name__ == "__main__":
    main()
