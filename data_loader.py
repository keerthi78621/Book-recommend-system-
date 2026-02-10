"""
Data Loader Module
==================
Handles loading datasets for the Book Recommender System.
"""

import pandas as pd
import os
from pathlib import Path


class DataLoader:
    """
    Loads and validates datasets for the recommender system.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.books = None
        self.ratings = None
        self.users = None
    
    def load_books(self, filename='books.csv'):
        """
        Load books dataset.
        
        Args:
            filename (str): Name of the books CSV file
            
        Returns:
            pd.DataFrame: Books dataframe
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Books file not found at: {filepath}\n"
                f"Please download the dataset and place it in the 'data' folder."
            )
        
        try:
            self.books = pd.read_csv(filepath, on_bad_lines='skip')
            print(f"✅ Loaded {len(self.books)} books")
            return self.books
        except Exception as e:
            print(f"❌ Error loading books: {e}")
            raise
    
    def load_ratings(self, filename='ratings.csv'):
        """
        Load ratings dataset.
        
        Args:
            filename (str): Name of the ratings CSV file
            
        Returns:
            pd.DataFrame: Ratings dataframe
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Ratings file not found at: {filepath}\n"
                f"Please download the dataset and place it in the 'data' folder."
            )
        
        try:
            self.ratings = pd.read_csv(filepath, on_bad_lines='skip')
            print(f"✅ Loaded {len(self.ratings)} ratings")
            return self.ratings
        except Exception as e:
            print(f"❌ Error loading ratings: {e}")
            raise
    
    def load_users(self, filename='users.csv'):
        """
        Load users dataset (optional).
        
        Args:
            filename (str): Name of the users CSV file
            
        Returns:
            pd.DataFrame: Users dataframe or None if not available
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"⚠️ Users file not found at: {filepath} (optional)")
            return None
        
        try:
            self.users = pd.read_csv(filepath, on_bad_lines='skip')
            print(f"✅ Loaded {len(self.users)} users")
            return self.users
        except Exception as e:
            print(f"⚠️ Error loading users: {e}")
            return None
    
    def load_all(self, books_file='books.csv', ratings_file='ratings.csv', 
                 users_file='users.csv'):
        """
        Load all datasets.
        
        Args:
            books_file (str): Books filename
            ratings_file (str): Ratings filename
            users_file (str): Users filename
            
        Returns:
            tuple: (books, ratings, users) dataframes
        """
        print("=" * 60)
        print("LOADING DATASETS")
        print("=" * 60)
        
        books = self.load_books(books_file)
        ratings = self.load_ratings(ratings_file)
        users = self.load_users(users_file)
        
        print("\n📊 Dataset Summary:")
        print(f"   Books: {len(books):,}")
        print(f"   Ratings: {len(ratings):,}")
        if users is not None:
            print(f"   Users: {len(users):,}")
        else:
            print(f"   Users: N/A (optional)")
        
        return books, ratings, users
    
    def get_statistics(self):
        """
        Display basic statistics about loaded datasets.
        
        Returns:
            dict: Statistics dictionary
        """
        if self.books is None or self.ratings is None:
            raise ValueError("Please load the datasets first.")
        
        stats = {
            'total_books': len(self.books),
            'total_ratings': len(self.ratings),
            'total_users': self.ratings['user_id'].nunique(),
            'unique_books': self.ratings['book_id'].nunique(),
            'avg_rating': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std(),
            'ratings_per_user': len(self.ratings) / self.ratings['user_id'].nunique(),
            'ratings_per_book': len(self.ratings) / self.ratings['book_id'].nunique()
        }
        
        print("\n📈 Dataset Statistics:")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value:,}")
        
        return stats


# Quick test
if __name__ == "__main__":
    try:
        loader = DataLoader()
        books, ratings, users = loader.load_all()
        loader.get_statistics()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n📥 Please download the dataset first!")
        print("   Visit: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k")
