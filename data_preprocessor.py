"""
Data Preprocessor Module
========================
Handles data cleaning and preprocessing for the Book Recommender System.
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path


class DataPreprocessor:
    """
    Preprocesses books and ratings data for the recommender system.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
    
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_books(self, books_df):
        """
        Preprocess books dataframe.
        
        Args:
            books_df (pd.DataFrame): Raw books dataframe
            
        Returns:
            pd.DataFrame: Cleaned books dataframe
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING BOOKS DATA")
        print("=" * 60)
        
        books = books_df.copy()
        
        # Display original info
        print(f"📚 Original books: {len(books)}")
        print(f"   Missing values:\n{books.isnull().sum()}")
        
        # 1. Handle missing values
        print("\n🔧 Step 1: Handling Missing Values")
        books['title'] = books['title'].fillna('Unknown Title')
        books['authors'] = books['authors'].fillna('Unknown Author')
        books['average_rating'] = books['average_rating'].fillna(books['average_rating'].median())
        
        # 2. Clean text columns
        print("\n🔧 Step 2: Cleaning Text Data")
        books['clean_title'] = books['title'].apply(self.clean_text)
        books['clean_authors'] = books['authors'].apply(self.clean_text)
        
        # 3. Create combined features for content-based filtering
        print("\n🔧 Step 3: Creating Combined Features")
        
        # Fill missing genres and description (handle column existence)
        if 'genres' in books.columns:
            books['genres'] = books['genres'].fillna('')
        else:
            books['genres'] = ''
        
        if 'description' in books.columns:
            books['description'] = books['description'].fillna('')
        else:
            books['description'] = ''
            
        books['clean_description'] = books['description'].astype(str).apply(self.clean_text)
        
        # Combine features (ensure all are strings)
        books['combined_features'] = (
            books['clean_title'].astype(str) + ' ' +
            books['clean_authors'].astype(str) + ' ' +
            books['clean_description'].astype(str)
        )
        
        # 4. Remove duplicates
        print("\n🔧 Step 4: Removing Duplicates")
        original_count = len(books)
        books = books.drop_duplicates(subset=['title', 'authors'], keep='first')
        duplicates_removed = original_count - len(books)
        print(f"   Removed {duplicates_removed} duplicate books")
        
        # 5. Filter by popularity (optional but recommended)
        print("\n🔧 Step 5: Filtering Popular Books")
        original_count = len(books)
        if 'ratings_count' in books.columns:
            # Keep books with at least 10 ratings
            books = books[books['ratings_count'] >= 10]
            print(f"   Kept {len(books)} books (filtered out low-rating books)")
        
        # 6. Reset index
        books = books.reset_index(drop=True)
        
        print(f"\n✅ Final books count: {len(books)}")
        
        return books
    
    def preprocess_ratings(self, ratings_df, books_df=None):
        """
        Preprocess ratings dataframe.
        
        Args:
            ratings_df (pd.DataFrame): Raw ratings dataframe
            books_df (pd.DataFrame): Books dataframe (for filtering)
            
        Returns:
            pd.DataFrame: Cleaned ratings dataframe
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING RATINGS DATA")
        print("=" * 60)
        
        ratings = ratings_df.copy()
        
        # Display original info
        print(f"⭐ Original ratings: {len(ratings)}")
        
        # 1. Handle missing values
        print("\n🔧 Step 1: Handling Missing Values")
        ratings = ratings.dropna()
        print(f"   After removing nulls: {len(ratings):,} ratings")
        
        # 2. Ensure valid ratings
        print("\n🔧 Step 2: Validating Ratings")
        ratings = ratings[ratings['rating'] >= 0]
        ratings = ratings[ratings['rating'] <= 5]
        print(f"   After validating ratings (0-5 scale): {len(ratings):,} ratings")
        
        # 3. Filter ratings to only include books in our books dataframe
        if books_df is not None:
            print("\n🔧 Step 3: Filtering Ratings to Match Books")
            valid_book_ids = set(books_df['book_id'].unique())
            original_count = len(ratings)
            ratings = ratings[ratings['book_id'].isin(valid_book_ids)]
            print(f"   Removed {original_count - len(ratings):,} ratings for books not in dataset")
        
        # 4. Remove duplicate ratings
        print("\n🔧 Step 4: Removing Duplicate Ratings")
        original_count = len(ratings)
        ratings = ratings.drop_duplicates(subset=['user_id', 'book_id'], keep='first')
        print(f"   Removed {original_count - len(ratings):,} duplicate ratings")
        
        # 5. Reset index
        ratings = ratings.reset_index(drop=True)
        
        print(f"\n✅ Final ratings count: {len(ratings):,}")
        
        # Rating distribution
        print("\n📊 Rating Distribution:")
        rating_dist = ratings['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            pct = (count / len(ratings)) * 100
            print(f"   {rating}⭐: {count:,} ({pct:.1f}%)")
        
        return ratings
    
    def filter_active_users(self, ratings_df, min_ratings=5):
        """
        Filter to keep only active users.
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            min_ratings (int): Minimum number of ratings per user
            
        Returns:
            pd.DataFrame: Filtered ratings
        """
        print(f"\n🔧 Filtering users with < {min_ratings} ratings")
        
        user_counts = ratings_df.groupby('user_id').size()
        active_users = user_counts[user_counts >= min_ratings].index
        
        filtered_ratings = ratings_df[ratings_df['user_id'].isin(active_users)]
        
        print(f"   Kept {len(filtered_ratings):,} ratings from {len(active_users):,} active users")
        
        return filtered_ratings
    
    def filter_popular_books(self, ratings_df, books_df, min_ratings=10):
        """
        Filter to keep only popular books.
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            books_df (pd.DataFrame): Books dataframe
            min_ratings (int): Minimum number of ratings per book
            
        Returns:
            tuple: (filtered_ratings, filtered_books)
        """
        print(f"\n🔧 Filtering books with < {min_ratings} ratings")
        
        book_counts = ratings_df.groupby('book_id').size()
        popular_books = book_counts[book_counts >= min_ratings].index
        
        filtered_ratings = ratings_df[ratings_df['book_id'].isin(popular_books)]
        filtered_books = books_df[books_df['book_id'].isin(popular_books)]
        
        print(f"   Kept {len(filtered_books):,} books with {len(filtered_ratings):,} ratings")
        
        return filtered_ratings, filtered_books
    
    def save_data(self, books, ratings):
        """
        Save preprocessed data to files.
        
        Args:
            books (pd.DataFrame): Preprocessed books
            ratings (pd.DataFrame): Preprocessed ratings
        """
        print("\n" + "=" * 60)
        print("SAVING PREPROCESSED DATA")
        print("=" * 60)
        
        books.to_csv(self.output_dir / 'cleaned_books.csv', index=False)
        ratings.to_csv(self.output_dir / 'clean_ratings.csv', index=False)
        
        print(f"✅ Saved cleaned books to: {self.output_dir / 'cleaned_books.csv'}")
        print(f"✅ Saved clean ratings to: {self.output_dir / 'clean_ratings.csv'}")
    
    def get_popular_books(self, books_df, ratings_df, top_n=10):
        """
        Get the most popular books.
        
        Args:
            books_df (pd.DataFrame): Books dataframe
            ratings_df (pd.DataFrame): Ratings dataframe
            top_n (int): Number of books to return
            
        Returns:
            pd.DataFrame: Popular books
        """
        # Calculate average rating and number of ratings per book
        book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        book_stats.columns = ['book_id', 'avg_rating', 'num_ratings']
        
        # Merge with books
        popular = book_stats.merge(books_df, on='book_id')
        
        # Sort by number of ratings (popularity)
        popular = popular.sort_values('num_ratings', ascending=False)
        
        return popular.head(top_n)
    
    def get_top_rated_books(self, books_df, ratings_df, min_ratings=50, top_n=10):
        """
        Get the highest-rated books (with minimum ratings threshold).
        
        Args:
            books_df (pd.DataFrame): Books dataframe
            ratings_df (pd.DataFrame): Ratings dataframe
            min_ratings (int): Minimum ratings to be considered
            top_n (int): Number of books to return
            
        Returns:
            pd.DataFrame: Top-rated books
        """
        # Calculate average rating and number of ratings per book
        book_stats = ratings_df.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        book_stats.columns = ['book_id', 'avg_rating', 'num_ratings']
        
        # Filter by minimum ratings
        book_stats = book_stats[book_stats['num_ratings'] >= min_ratings]
        
        # Merge with books
        top_rated = book_stats.merge(books_df, on='book_id')
        
        # Sort by average rating
        top_rated = top_rated.sort_values('avg_rating', ascending=False)
        
        return top_rated.head(top_n)


# Quick test
if __name__ == "__main__":
    from data_loader import DataLoader
    
    try:
        # Load data
        loader = DataLoader()
        books, ratings, _ = loader.load_all()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        clean_books = preprocessor.preprocess_books(books)
        clean_ratings = preprocessor.preprocess_ratings(ratings, clean_books)
        
        # Filter active users and popular books
        clean_ratings = preprocessor.filter_active_users(clean_ratings, min_ratings=5)
        clean_ratings, clean_books = preprocessor.filter_popular_books(
            clean_ratings, clean_books, min_ratings=10
        )
        
        # Save data
        preprocessor.save_data(clean_books, clean_ratings)
        
        # Show popular books
        print("\n" + "=" * 60)
        print("TOP 10 POPULAR BOOKS")
        print("=" * 60)
        popular = preprocessor.get_popular_books(clean_books, clean_ratings, top_n=10)
        print(popular[['title', 'authors', 'avg_rating', 'num_ratings']].to_string())
        
    except FileNotFoundError as e:
        print(f"\n{e}")
