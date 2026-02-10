#!/usr/bin/env python3
"""
Book Recommender System - Main Execution Script
==============================================
This script runs the complete book recommendation pipeline.
"""

import sys
import time


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("📚 BOOK RECOMMENDER SYSTEM")
    print("=" * 70)
    print("A machine learning-based book recommendation system")
    print("Using Item-Based Collaborative Filtering with Cosine Similarity")
    print("=" * 70)


def main():
    """Main execution function."""
    print_header()
    
    # Track execution time
    start_time = time.time()
    
    try:
        # =======================
        # STEP 1: Load Data
        # =======================
        print("\n📖 STEP 1: Loading Data...")
        print("-" * 40)
        
        from data_loader import DataLoader
        
        loader = DataLoader()
        books, ratings, users = loader.load_all()
        loader.get_statistics()
        
        # =======================
        # STEP 2: Preprocess Data
        # =======================
        print("\n📖 STEP 2: Preprocessing Data...")
        print("-" * 40)
        
        from data_preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Preprocess books
        clean_books = preprocessor.preprocess_books(books)
        
        # Preprocess ratings
        clean_ratings = preprocessor.preprocess_ratings(ratings, clean_books)
        
        # Filter active users and popular books
        print("\n📖 STEP 2b: Filtering Active Users and Popular Books...")
        clean_ratings = preprocessor.filter_active_users(clean_ratings, min_ratings=5)
        clean_ratings, clean_books = preprocessor.filter_popular_books(
            clean_ratings, clean_books, min_ratings=10
        )
        
        # Save preprocessed data
        preprocessor.save_data(clean_books, clean_ratings)
        
        # Show popular books
        print("\n📖 TOP 10 MOST POPULAR BOOKS:")
        print("-" * 40)
        popular = preprocessor.get_popular_books(clean_books, clean_ratings, top_n=10)
        for i, (_, book) in enumerate(popular.iterrows(), 1):
            print(f"{i:2}. {book['title'][:50]:<50} | ⭐ {book['avg_rating']:.2f} | 👥 {book['num_ratings']:,}")
        
        # =======================
        # STEP 3: Create Visualizations
        # =======================
        print("\n📊 STEP 3: Creating Visualizations...")
        print("-" * 40)
        
        from visualization import DataVisualizer
        
        visualizer = DataVisualizer()
        visualizer.plot_rating_distribution(clean_ratings)
        visualizer.plot_popular_books(clean_books, clean_ratings)
        visualizer.create_summary_dashboard(clean_books, clean_ratings)
        
        # =======================
        # STEP 4: Build Recommender
        # =======================
        print("\n🤖 STEP 4: Building Recommendation Engine...")
        print("-" * 40)
        
        from recommender import BookRecommender
        
        recommender = BookRecommender()
        
        # Load preprocessed data
        recommender.books = clean_books
        recommender.ratings = clean_ratings
        
        # Create user-item matrix
        recommender.create_user_item_matrix()
        
        # Compute similarity
        recommender.compute_similarity(method='cosine')
        
        # =======================
        # STEP 5: Demo Recommendations
        # =======================
        print("\n🎯 STEP 5: Testing Recommendation Engine...")
        print("-" * 40)
        
        # Get recommendations for a sample book
        print("\n📚 RECOMMENDATIONS FOR 'THE HOBBIT':")
        print("=" * 70)
        
        recommendations = recommender.get_book_recommendations("The Hobbit", top_n=10)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. 📖 {rec['title']}")
                print(f"   ✍️  Author: {rec['authors']}")
                print(f"   ⭐ Rating: {rec['average_rating']}/5")
                print(f"   🎯 Similarity: {rec['similarity_score']:.3f}")
                if rec.get('ratings_count') != 'N/A':
                    print(f"   👥 Ratings: {rec['ratings_count']:,}")
        else:
            print("❌ No recommendations found. Try a different book title.")
        
        # =======================
        # STEP 6: Interactive Mode
        # =======================
        print("\n" + "=" * 70)
        print("🎮 INTERACTIVE MODE")
        print("=" * 70)
        print("Enter a book title to get recommendations, or 'quit' to exit.")
        
        while True:
            print("\n" + "-" * 40)
            user_input = input("📚 Enter book title: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Book Recommender System!")
                break
            
            if not user_input:
                print("⚠️ Please enter a book title.")
                continue
            
            # Search for books
            search_results = recommender.search_books(user_input, top_n=5)
            
            if search_results.empty:
                print(f"❌ No books found matching '{user_input}'")
                continue
            
            print(f"\n📖 Found {len(search_results)} matching books:")
            for i, (_, book) in enumerate(search_results.iterrows(), 1):
                print(f"   {i}. {book['title']} by {book['authors']}")
            
            # Get recommendations for the first match
            best_match = search_results.iloc[0]['title']
            print(f"\n📚 Top 10 Recommendations for '{best_match}':")
            print("=" * 70)
            
            recommendations = recommender.get_book_recommendations(best_match, top_n=10)
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i:2}. 📖 {rec['title'][:55]}")
                    print(f"    ✍️  {rec['authors'][:40]}")
                    print(f"    ⭐ {rec['average_rating']}/5 | 🎯 {rec['similarity_score']:.3f}")
            else:
                print("❌ No recommendations found.")
        
        # =======================
        # Completion
        # =======================
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print("📁 Output files saved in: output/")
        print("   - cleaned_books.csv")
        print("   - clean_ratings.csv")
        print("   - rating_distribution.png")
        print("   - popular_books.png")
        print("   - summary_dashboard.png")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📥 To run this project, you need to:")
        print("   1. Download the Goodbooks-10k dataset from Kaggle")
        print("      Visit: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k")
        print("   2. Extract the files to a 'data' folder:")
        print("      - books.csv")
        print("      - ratings.csv")
        print("      - users.csv (optional)")
        print("   3. Run this script again")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n💡 Try running: python data_loader.py to check dataset loading")
        sys.exit(1)


if __name__ == "__main__":
    main()
