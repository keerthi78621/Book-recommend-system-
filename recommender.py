"""
Recommender Engine Module
=========================
Core recommendation system using item-based collaborative filtering
with cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
import os
from pathlib import Path


class BookRecommender:
    """
    Item-based Collaborative Filtering Book Recommender.
    
    Uses cosine similarity to find similar books based on user ratings.
    """
    
    def __init__(self, books_path='output/cleaned_books.csv', 
                 ratings_path='output/clean_ratings.csv'):
        """
        Initialize the recommender system.
        
        Args:
            books_path (str): Path to cleaned books CSV
            ratings_path (str): Path to cleaned ratings CSV
        """
        self.books = None
        self.ratings = None
        self.user_item_matrix = None
        self.item_similarity = None
        self.book_id_to_idx = None
        self.idx_to_book_id = None
        self.books_path = books_path
        self.ratings_path = ratings_path
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
    
    def load_data(self, books_path=None, ratings_path=None):
        """
        Load preprocessed data.
        
        Args:
            books_path (str): Custom path to books CSV
            ratings_path (str): Custom path to ratings CSV
        """
        if books_path:
            self.books_path = books_path
        if ratings_path:
            self.ratings_path = ratings_path
        
        print("=" * 60)
        print("LOADING PREPROCESSED DATA")
        print("=" * 60)
        
        if not os.path.exists(self.books_path):
            raise FileNotFoundError(
                f"Cleaned books file not found: {self.books_path}\n"
                "Please run data_preprocessor.py first."
            )
        
        if not os.path.exists(self.ratings_path):
            raise FileNotFoundError(
                f"Cleaned ratings file not found: {self.ratings_path}\n"
                "Please run data_preprocessor.py first."
            )
        
        self.books = pd.read_csv(self.books_path)
        self.ratings = pd.read_csv(self.ratings_path)
        
        print(f"✅ Loaded {len(self.books)} books")
        print(f"✅ Loaded {len(self.ratings):,} ratings")
    
    def create_user_item_matrix(self):
        """
        Create user-item rating matrix.
        
        Returns:
            csr_matrix: Sparse user-item matrix
        """
        print("\n" + "=" * 60)
        print("CREATING USER-ITEM MATRIX")
        print("=" * 60)
        
        # Create mappings between book_id and matrix index
        unique_books = self.ratings['book_id'].unique()
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        
        # Create sparse matrix
        rows = self.ratings['book_id'].map(self.book_id_to_idx)
        cols = self.ratings['user_id']
        values = self.ratings['rating'].values
        
        # Create user-item matrix (items x users for item-based similarity)
        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(unique_books), self.ratings['user_id'].nunique())
        )
        
        print(f"✅ Created matrix: {self.user_item_matrix.shape}")
        print(f"   Items (books): {self.user_item_matrix.shape[0]:,}")
        print(f"   Users: {self.user_item_matrix.shape[1]:,}")
        print(f"   Sparsity: {(1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100:.2f}%")
        
        return self.user_item_matrix
    
    def compute_similarity(self, method='cosine'):
        """
        Compute item-item similarity matrix.
        
        Args:
            method (str): Similarity method ('cosine', 'correlation')
        """
        print("\n" + "=" * 60)
        print(f"COMPUTING SIMILARITY MATRIX ({method.upper()})")
        print("=" * 60)
        
        if self.user_item_matrix is None:
            raise ValueError("Please create user-item matrix first.")
        
        print("⏳ Computing similarity matrix... (this may take a moment)")
        
        if method == 'cosine':
            # Transpose to get users x items for cosine similarity
            # Each item becomes a vector of user ratings
            item_vectors = self.user_item_matrix.T
            
            # Compute cosine similarity between items
            self.item_similarity = cosine_similarity(item_vectors)
        
        elif method == 'correlation':
            from scipy.stats import pearsonr
            
            # Manual Pearson correlation (slower but accurate)
            item_vectors = self.user_item_matrix.T.toarray()
            n_items = item_vectors.shape[1]
            self.item_similarity = np.zeros((n_items, n_items))
            
            for i in range(n_items):
                for j in range(n_items):
                    if i != j:
                        mask = (item_vectors[:, i] != 0) & (item_vectors[:, j] != 0)
                        if mask.sum() > 0:
                            corr, _ = pearsonr(
                                item_vectors[mask, i], 
                                item_vectors[mask, j]
                            )
                            self.item_similarity[i, j] = corr if not np.isnan(corr) else 0
        
        print(f"✅ Similarity matrix shape: {self.item_similarity.shape}")
        
        # Save similarity matrix
        self.save_similarity_matrix()
        
        return self.item_similarity
    
    def save_similarity_matrix(self, filepath=None):
        """
        Save similarity matrix to file.
        
        Args:
            filepath (str): Path to save the matrix
        """
        if filepath is None:
            filepath = self.output_dir / 'similarity_matrix.npy'
        
        np.save(filepath, self.item_similarity)
        print(f"💾 Saved similarity matrix to: {filepath}")
    
    def load_similarity_matrix(self, filepath=None):
        """
        Load precomputed similarity matrix.
        
        Args:
            filepath (str): Path to the saved matrix
        """
        if filepath is None:
            filepath = self.output_dir / 'similarity_matrix.npy'
        
        if not os.path.exists(filepath):
            print(f"⚠️ Similarity matrix not found at: {filepath}")
            return False
        
        self.item_similarity = np.load(filepath)
        print(f"✅ Loaded similarity matrix: {self.item_similarity.shape}")
        return True
    
    def get_book_recommendations(self, book_title, top_n=10):
        """
        Get book recommendations based on a book title.
        
        Args:
            book_title (str): Title of the book
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended books with details
        """
        if self.books is None or self.item_similarity is None:
            raise ValueError("Please load data and compute similarity first.")
        
        # Find the book in our dataset
        book_match = self.books[self.books['title'].str.contains(book_title, case=False, na=False)]
        
        if book_match.empty:
            print(f"❌ Book not found: '{book_title}'")
            # Try fuzzy matching or partial match
            print("\n💡 Try searching for a book like:")
            popular = self.get_popular_books(top_n=5)
            for _, book in popular.iterrows():
                print(f"   - {book['title']}")
            return []
        
        # Use the first match
        book = book_match.iloc[0]
        book_id = book['book_id']
        
        # Get the book index
        if book_id not in self.book_id_to_idx:
            print(f"⚠️ Book '{book_title}' has no ratings. Try another book.")
            return []
        
        idx = self.book_id_to_idx[book_id]
        
        # Get similarity scores for this book
        sim_scores = self.item_similarity[idx]
        
        # Get top N+1 similar items (including itself)
        top_indices = np.argsort(sim_scores)[::-1][:(top_n + 1)]
        
        # Remove the book itself
        recommendations = []
        for i in top_indices:
            if i != idx:
                book_idx = self.idx_to_book_id[i]
                book_info = self.books[self.books['book_id'] == book_idx].iloc[0]
                
                recommendations.append({
                    'book_id': book_idx,
                    'title': book_info['title'],
                    'authors': book_info['authors'],
                    'average_rating': book_info['average_rating'],
                    'similarity_score': float(sim_scores[i]),
                    'ratings_count': book_info.get('ratings_count', 'N/A')
                })
        
        return recommendations[:top_n]
    
    def get_user_recommendations(self, user_id, top_n=10):
        """
        Get recommendations for a specific user.
        
        Args:
            user_id (int): User ID
            top_n (int): Number of recommendations
            
        Returns:
            list: Recommended books
        """
        if self.books is None or self.item_similarity is None:
            raise ValueError("Please load data and compute similarity first.")
        
        # Check if user exists
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if user_ratings.empty:
            print(f"❌ User {user_id} not found.")
            return []
        
        # Get books the user has rated highly (4 or 5 stars)
        liked_books = user_ratings[user_ratings['rating'] >= 4]['book_id'].tolist()
        
        if not liked_books:
            print(f"⚠️ User {user_id} has no highly-rated books.")
            return []
        
        # Calculate scores based on similar books to liked books
        book_scores = {}
        
        for book_id in liked_books:
            if book_id not in self.book_id_to_idx:
                continue
            
            idx = self.book_id_to_idx[book_id]
            sim_scores = self.item_similarity[idx]
            
            # Add scores for all books
            for i, score in enumerate(sim_scores):
                other_book_id = self.idx_to_book_id[i]
                
                # Skip books user has already rated
                if other_book_id in user_ratings['book_id'].values:
                    continue
                
                if other_book_id not in book_scores:
                    book_scores[other_book_id] = 0
                
                book_scores[other_book_id] += score
        
        # Sort by score and return top N
        sorted_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for book_id, score in sorted_books[:top_n]:
            book_info = self.books[self.books['book_id'] == book_id].iloc[0]
            
            recommendations.append({
                'book_id': book_id,
                'title': book_info['title'],
                'authors': book_info['authors'],
                'average_rating': book_info['average_rating'],
                'recommendation_score': float(score),
                'ratings_count': book_info.get('ratings_count', 'N/A')
            })
        
        return recommendations
    
    def get_popular_books(self, top_n=10):
        """
        Get the most popular books.
        
        Args:
            top_n (int): Number of books to return
            
        Returns:
            pd.DataFrame: Popular books
        """
        if self.ratings is None:
            raise ValueError("Please load data first.")
        
        book_counts = self.ratings.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        book_counts.columns = ['book_id', 'avg_rating', 'num_ratings']
        popular = book_counts.merge(self.books, on='book_id')
        popular = popular.sort_values('num_ratings', ascending=False)
        
        return popular.head(top_n)
    
    def display_recommendations(self, recommendations, title="Recommendations"):
        """
        Display recommendations in a formatted way.
        
        Args:
            recommendations (list): List of recommendation dicts
            title (str): Title to display
        """
        print("\n" + "=" * 70)
        print(f"📚 {title}")
        print("=" * 70)
        
        if not recommendations:
            print("No recommendations found.")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. 📖 {rec['title']}")
            print(f"   Author: {rec['authors']}")
            print(f"   Rating: {rec['average_rating']}/5")
            if 'similarity_score' in rec:
                print(f"   Similarity: {rec['similarity_score']:.3f}")
            elif 'recommendation_score' in rec:
                print(f"   Score: {rec['recommendation_score']:.3f}")
            if rec.get('ratings_count') != 'N/A':
                print(f"   Ratings: {rec['ratings_count']:,}")
        
        print()
    
    def search_books(self, query, top_n=5):
        """
        Search for books by title.
        
        Args:
            query (str): Search query
            top_n (int): Number of results
            
        Returns:
            pd.DataFrame: Matching books
        """
        if self.books is None:
            raise ValueError("Please load data first.")
        
        matches = self.books[
            self.books['title'].str.contains(query, case=False, na=False)
        ].head(top_n)
        
        return matches
    
    def get_book_info(self, book_title):
        """
        Get information about a specific book.
        
        Args:
            book_title (str): Title of the book
            
        Returns:
            dict: Book information
        """
        if self.books is None:
            raise ValueError("Please load data first.")
        
        match = self.books[self.books['title'].str.contains(book_title, case=False, na=False)]
        
        if match.empty:
            return None
        
        return match.iloc[0].to_dict()


# Quick test and demonstration
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 BOOK RECOMMENDER SYSTEM - DEMO")
    print("=" * 70)
    
    try:
        # Initialize recommender
        recommender = BookRecommender()
        
        # Load data
        recommender.load_data()
        
        # Create user-item matrix
        recommender.create_user_item_matrix()
        
        # Compute similarity
        recommender.compute_similarity(method='cosine')
        
        # Show popular books
        print("\n" + "=" * 70)
        print("📖 TOP 10 POPULAR BOOKS")
        print("=" * 70)
        popular = recommender.get_popular_books(top_n=10)
        for i, (_, book) in enumerate(popular.iterrows(), 1):
            print(f"{i}. {book['title'][:50]:<50} | Rating: {book['avg_rating']:.2f} | Ratings: {book['num_ratings']:,}")
        
        # Get recommendations for a book
        print("\n" + "=" * 70)
        print("📚 RECOMMENDATIONS FOR 'THE HOBBIT'")
        print("=" * 70)
        
        recommendations = recommender.get_book_recommendations("The Hobbit", top_n=5)
        recommender.display_recommendations(recommendations, "Books Similar to 'The Hobbit'")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📥 Please run the preprocessing first:")
        print("   1. Download the dataset from Kaggle")
        print("   2. Place files in 'data' folder")
        print("   3. Run: python data_loader.py")
        print("   4. Run: python data_preprocessor.py")
