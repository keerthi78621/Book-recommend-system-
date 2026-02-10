#!/usr/bin/env python3
"""
Sample Data Generator
=====================
Generates sample data for testing the Book Recommender System
when the real dataset is not available.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def generate_sample_data(output_dir='data', n_books=100, n_users=50, n_ratings=1000):
    """
    Generate sample books, users, and ratings data.
    
    Args:
        output_dir (str): Directory to save data
        n_books (int): Number of books to generate
        n_users (int): Number of users to generate
        n_ratings (int): Number of ratings to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("GENERATING SAMPLE DATA")
    print("=" * 60)
    
    # Sample book data
    book_titles = [
        "The Great Gatsby", "To Kill a Mockingbird", "1984", "Pride and Prejudice",
        "The Catcher in the Rye", "The Hobbit", "Harry Potter", "The Lord of the Rings",
        "Fahrenheit 451", "Brave New World", "Animal Farm", "The Odyssey",
        "The Iliad", "Moby Dick", "War and Peace", "Crime and Punishment",
        "The Brothers Karamazov", "Anna Karenina", "Wuthering Heights", "Jane Eyre",
        "Les Misérables", "The Count of Monte Cristo", "Don Quixote", "The Divine Comedy",
        "Frankenstein", "Dracula", "The Picture of Dorian Gray", "The Turn of the Screw",
        "Great Expectations", "Oliver Twist", "David Copperfield", "A Tale of Two Cities",
        "The Bell Jar", "The Outsiders", "The Giver", "Hunger Games",
        "The Fault in Our Stars", "Paper Towns", "Looking for Alaska", "Percy Jackson",
        "The Lightning Thief", "Sea of Monsters", "The Hunger Games", "Catching Fire",
        "Mockingjay", "Divergent", "Insurgent", "Allegiant", "The Maze Runner",
        "The Scorch Trials", "The Death Cure", "The Knife of Never Letting Go",
        "Ender's Game", "Speaker for the Dead", "Xenocide", "Children of the Mind",
        "The Martian", "Artemis", "Project Hail Mary", "Dune", "Messiah",
        "Neuromancer", "Snow Crash", "Count Zero", "Mona Lisa Overdrive",
        "Foundation", "Foundation and Empire", "Second Foundation", "Foundation's Edge",
        "I Robot", "Bicentennial Man", "The Caves of Steel", "The Naked Sun",
        "The Rest of the Robots", "Asimov's Complete Stories", "Nightfall", "The Stars",
        "Twelve Stories", "The Gods Themselves", "The End of Eternity",
        "Jurassic Park", "The Lost World", "Sphere", "Congo", "Sphere",
        "Timeline", "State of Fear", "The Andromeda Strain", "Next",
        "Prey", "Cell", "Travels", "Terminal", "Micro", "Jurassic Park"
    ]
    
    authors = [
        "J.R.R. Tolkien", "J.K. Rowling", "George Orwell", "F. Scott Fitzgerald",
        "Harper Lee", "Ernest Hemingway", "John Steinbeck", "Jane Austen",
        "Charles Dickens", "Leo Tolstoy", "Fyodor Dostoevsky", "Homer",
        "George R.R. Martin", "Ray Bradbury", "Aldous Huxley", "Stephen King",
        "Agatha Christie", "Arthur Conan Doyle", "Edgar Allan Poe", "Mary Shelley",
        "Bram Stoker", "Oscar Wilde", "Virginia Woolf", "James Joyce",
        "H.G. Wells", "Kurt Vonnegut", "Douglas Adams", "Philip K. Dick",
        "Isaac Asimov", "Frank Herbert", "Dan Brown", "Paulo Coelho",
        "Neil Gaiman", "Terry Pratchett", "Robert Jordan", "Brandon Sanderson",
        "Patrick Rothfuss", "Suzanne Collins", "Veronica Roth", "James Dashner",
        "Orson Scott Card", "Andy Weir", "Michael Crichton", "Anne Rice"
    ]
    
    genres = [
        "Fiction", "Classic", "Fantasy", "Science Fiction", "Romance",
        "Mystery", "Thriller", "Horror", "Dystopian", "Adventure",
        "Young Adult", "Historical", "Philosophical", "Literary", "Magical Realism"
    ]
    
    # Generate books dataframe
    books = []
    for i in range(n_books):
        book = {
            'book_id': i + 1,
            'title': book_titles[i % len(book_titles)] if i < len(book_titles) else f"Sample Book {i+1}",
            'authors': authors[i % len(authors)],
            'average_rating': round(np.random.uniform(3.0, 5.0), 2),
            'ratings_count': np.random.randint(100, 10000),
            'genres': ', '.join(np.random.choice(genres, size=np.random.randint(1, 3), replace=False)),
            'description': f"This is a sample book description for {book_titles[i % len(book_titles)] if i < len(book_titles) else f'Sample Book {i+1}'}. It contains engaging content that readers love."
        }
        books.append(book)
    
    books_df = pd.DataFrame(books)
    books_df.to_csv(output_path / 'books.csv', index=False)
    print(f"✅ Generated {len(books_df)} books")
    
    # Generate users dataframe
    users = []
    for i in range(n_users):
        user = {
            'user_id': i + 1,
            'age': np.random.randint(18, 65),
            'location': np.random.choice(['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Toronto'])
        }
        users.append(user)
    
    users_df = pd.DataFrame(users)
    users_df.to_csv(output_path / 'users.csv', index=False)
    print(f"✅ Generated {len(users_df)} users")
    
    # Generate ratings dataframe
    ratings = []
    for i in range(n_ratings):
        rating = {
            'user_id': np.random.randint(1, n_users + 1),
            'book_id': np.random.randint(1, n_books + 1),
            'rating': np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.35, 0.25]),
        }
        ratings.append(rating)
    
    ratings_df = pd.DataFrame(ratings)
    ratings_df.to_csv(output_path / 'ratings.csv', index=False)
    print(f"✅ Generated {len(ratings_df)} ratings")
    
    # Summary
    print("\n📊 Sample Data Summary:")
    print(f"   Books: {len(books_df)}")
    print(f"   Users: {len(users_df)}")
    print(f"   Ratings: {len(ratings_df)}")
    print(f"   Average Rating: {ratings_df['rating'].mean():.2f}")
    
    print(f"\n📁 Files saved to: {output_path}")
    print("   - books.csv")
    print("   - users.csv")
    print("   - ratings.csv")
    
    return books_df, users_df, ratings_df


def create_minimal_dataset(output_dir='data'):
    """
    Create a minimal working dataset for quick testing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 10 Books
    books = pd.DataFrame({
        'book_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'The Hobbit',
            'The Fellowship of the Ring',
            'The Two Towers',
            'The Return of the King',
            'Harry Potter and the Sorcerer\'s Stone',
            'Harry Potter and the Chamber of Secrets',
            'The Lightning Thief',
            'The Sea of Monsters',
            'The Lion, the Witch and the Wardrobe',
            'The Magician\'s Nephew'
        ],
        'authors': [
            'J.R.R. Tolkien',
            'J.R.R. Tolkien',
            'J.R.R. Tolkien',
            'J.R.R. Tolkien',
            'J.K. Rowling',
            'J.K. Rowling',
            'Rick Riordan',
            'Rick Riordan',
            'C.S. Lewis',
            'C.S. Lewis'
        ],
        'average_rating': [4.26, 4.45, 4.43, 4.45, 4.47, 4.42, 4.28, 4.25, 4.21, 4.22],
        'ratings_count': [2100000, 1900000, 1800000, 1750000, 2000000, 1950000, 900000, 850000, 1200000, 800000],
        'genres': [
            'Fantasy, Adventure',
            'Fantasy, Adventure',
            'Fantasy, Adventure',
            'Fantasy, Adventure',
            'Fantasy, Young Adult',
            'Fantasy, Young Adult',
            'Fantasy, Young Adult',
            'Fantasy, Young Adult',
            'Fantasy, Children',
            'Fantasy, Children'
        ]
    })
    
    # 10 Users
    users = pd.DataFrame({
        'user_id': range(1, 11),
        'age': [25, 30, 22, 35, 28, 24, 32, 27, 29, 26]
    })
    
    # Ratings (matrix with some structure)
    ratings_data = []
    
    # User 1-3: Tolkien fans
    user_ratings = [
        (1, 1, 5), (1, 2, 5), (1, 3, 5), (1, 4, 5),
        (2, 1, 5), (2, 2, 5), (2, 3, 4), (2, 4, 4),
        (3, 1, 4), (3, 2, 5), (3, 3, 4), (3, 4, 5),
    ]
    
    # User 4-6: Rowling fans
    user_ratings += [
        (4, 5, 5), (4, 6, 5), (4, 7, 3), (4, 8, 3),
        (5, 5, 5), (5, 6, 5), (5, 7, 4), (5, 8, 4),
        (6, 5, 5), (6, 6, 4), (6, 7, 3), (6, 8, 3),
    ]
    
    # User 7-8: Mixed fantasy fans
    user_ratings += [
        (7, 1, 5), (7, 2, 4), (7, 5, 4), (7, 6, 4), (7, 9, 5),
        (8, 2, 5), (8, 3, 4), (8, 5, 4), (8, 6, 3), (8, 10, 5),
    ]
    
    # User 9-10: Broad readers
    user_ratings += [
        (9, 1, 4), (9, 3, 5), (9, 5, 4), (9, 7, 5), (9, 9, 4),
        (10, 2, 5), (10, 4, 4), (10, 6, 5), (10, 8, 4), (10, 10, 4),
    ]
    
    ratings = pd.DataFrame(user_ratings, columns=['user_id', 'book_id', 'rating'])
    
    # Save files
    books.to_csv(output_path / 'books.csv', index=False)
    users.to_csv(output_path / 'users.csv', index=False)
    ratings.to_csv(output_path / 'ratings.csv', index=False)
    
    print("✅ Created minimal dataset with 10 books, 10 users, and structured ratings")
    print(f"\n📁 Files saved to: {output_path}")
    
    return books, users, ratings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample data for testing')
    parser.add_argument('--minimal', action='store_true',
                       help='Create minimal dataset (faster)')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.minimal:
        create_minimal_dataset(args.output)
    else:
        generate_sample_data(args.output, n_books=100, n_users=50, n_ratings=1000)
