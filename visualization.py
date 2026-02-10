"""
Visualization Module
====================
Creates visualizations for the Book Recommender System data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataVisualizer:
    """
    Creates visualizations for the recommender system data.
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_rating_distribution(self, ratings_df, save=True):
        """
        Plot the distribution of ratings.
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            save (bool): Whether to save the plot
        """
        print("\n📊 Creating Rating Distribution Plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rating counts
        ax1 = axes[0]
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        bars = ax1.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='white')
        ax1.set_xlabel('Rating', fontsize=12)
        ax1.set_ylabel('Number of Ratings', fontsize=12)
        ax1.set_title('Distribution of Ratings', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, count in zip(bars, rating_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # Rating percentages
        ax2 = axes[1]
        rating_pcts = (rating_counts / len(ratings_df) * 100).sort_index()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(rating_pcts)))
        wedges, texts, autotexts = ax2.pie(rating_pcts, labels=rating_pcts.index, 
                                           autopct='%1.1f%%', colors=colors,
                                           explode=[0.02] * len(rating_pcts))
        ax2.set_title('Rating Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'rating_distribution.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_popular_books(self, books_df, ratings_df, top_n=15, save=True):
        """
        Plot the most popular books.
        
        Args:
            books_df (pd.DataFrame): Books dataframe
            ratings_df (pd.DataFrame): Ratings dataframe
            top_n (int): Number of books to plot
            save (bool): Whether to save the plot
        """
        print("\n📊 Creating Popular Books Plot...")
        
        # Calculate popularity
        book_counts = ratings_df.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        book_counts.columns = ['book_id', 'avg_rating', 'num_ratings']
        
        popular = book_counts.merge(books_df, on='book_id')
        popular = popular.sort_values('num_ratings', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(popular))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(popular)))
        
        bars = ax.barh(y_pos, popular['num_ratings'], color=colors, edgecolor='white')
        
        # Add book titles
        ax.set_yticks(y_pos)
        ax.set_yticklabels([title[:40] + '...' if len(title) > 40 else title 
                           for title in popular['title']], fontsize=10)
        
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Number of Ratings', fontsize=12)
        ax.set_title(f'Top {top_n} Most Popular Books', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, rating in zip(bars, popular['avg_rating']):
            ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                   f'⭐ {rating:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'popular_books.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_ratings_per_user(self, ratings_df, save=True):
        """
        Plot the distribution of ratings per user.
        
        Args:
            ratings_df (pd.DataFrame): Ratings dataframe
            save (bool): Whether to save the plot
        """
        print("\n📊 Creating Ratings Per User Plot...")
        
        user_counts = ratings_df.groupby('user_id').size()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(user_counts, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(user_counts.median(), color='red', linestyle='--', 
                   label=f'Median: {user_counts.median():.0f}')
        ax1.axvline(user_counts.mean(), color='green', linestyle='--',
                   label=f'Mean: {user_counts.mean():.0f}')
        ax1.set_xlabel('Number of Ratings', fontsize=12)
        ax1.set_ylabel('Number of Users', fontsize=12)
        ax1.set_title('Distribution of Ratings Per User', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Box plot
        ax2 = axes[1]
        ax2.boxplot(user_counts, vert=True)
        ax2.set_ylabel('Number of Ratings', fontsize=12)
        ax2.set_title('User Rating Activity', fontsize=14, fontweight='bold')
        
        # Add statistics text
        stats_text = f"Max: {user_counts.max():,}\nQ3: {user_counts.quantile(0.75):.0f}\nMedian: {user_counts.median():.0f}\nQ1: {user_counts.quantile(0.25):.0f}\nMin: {user_counts.min():,}"
        ax2.text(1.2, user_counts.median(), stats_text, fontsize=10, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'ratings_per_user.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_average_rating_distribution(self, books_df, ratings_df, save=True):
        """
        Plot the distribution of average book ratings.
        
        Args:
            books_df (pd.DataFrame): Books dataframe
            ratings_df (pd.DataFrame): Ratings dataframe
            save (bool): Whether to save the plot
        """
        print("\n📊 Creating Average Rating Distribution Plot...")
        
        # Calculate average ratings per book
        book_avg_ratings = ratings_df.groupby('book_id')['rating'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(book_avg_ratings, bins=50, color='coral', edgecolor='white', alpha=0.7)
        ax.axvline(book_avg_ratings.mean(), color='red', linestyle='--',
                  label=f'Mean: {book_avg_ratings.mean():.2f}')
        ax.axvline(book_avg_ratings.median(), color='blue', linestyle='--',
                  label=f'Median: {book_avg_ratings.median():.2f}')
        
        ax.set_xlabel('Average Rating', fontsize=12)
        ax.set_ylabel('Number of Books', fontsize=12)
        ax.set_title('Distribution of Average Book Ratings', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'average_ratings.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_similarity_heatmap(self, recommender, sample_size=50, save=True):
        """
        Plot a heatmap of book similarity (sample).
        
        Args:
            recommender: BookRecommender instance
            sample_size (int): Number of books to sample
            save (bool): Whether to save the plot
        """
        print("\n📊 Creating Similarity Heatmap (this may take a moment)...")
        
        if recommender.item_similarity is None:
            print("⚠️ No similarity matrix found. Run recommender.compute_similarity() first.")
            return None
        
        # Sample books for visualization
        sample_indices = np.random.choice(
            recommender.item_similarity.shape[0], 
            size=min(sample_size, recommender.item_similarity.shape[0]),
            replace=False
        )
        
        sample_similarity = recommender.item_similarity[np.ix_(sample_indices, sample_indices)]
        
        # Get book titles for labels
        sample_book_ids = [recommender.idx_to_book_id[idx] for idx in sample_indices]
        sample_titles = []
        for book_id in sample_book_ids:
            book = recommender.books[recommender.books['book_id'] == book_id]
            if not book.empty:
                title = book.iloc[0]['title'][:20]
                sample_titles.append(title)
            else:
                sample_titles.append(f"Book {book_id}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(sample_similarity, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(sample_titles)))
        ax.set_yticks(np.arange(len(sample_titles)))
        ax.set_xticklabels(sample_titles, rotation=90, fontsize=8)
        ax.set_yticklabels(sample_titles, fontsize=8)
        
        ax.set_title(f'Book Similarity Heatmap (Sample of {len(sample_titles)} Books)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'similarity_heatmap.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filepath}")
        
        plt.show()
        return fig
    
    def create_summary_dashboard(self, books_df, ratings_df, save=True):
        """
        Create a summary dashboard with key metrics.
        
        Args:
            books_df (pd.DataFrame): Books dataframe
            ratings_df (pd.DataFrame): Ratings dataframe
            save (bool): Whether to save the plot
        """
        print("\n📊 Creating Summary Dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        
        # Calculate metrics
        book_counts = ratings_df.groupby('book_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        book_counts.columns = ['book_id', 'avg_rating', 'num_ratings']
        
        user_counts = ratings_df.groupby('user_id').size()
        
        # Create grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Total books
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, f'{len(books_df):,}', fontsize=40, ha='center', va='center',
                fontweight='bold', color='steelblue')
        ax1.text(0.5, 0.1, 'Total Books', fontsize=14, ha='center', va='center', color='gray')
        ax1.axis('off')
        
        # 2. Total ratings
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, f'{len(ratings_df):,}', fontsize=40, ha='center', va='center',
                fontweight='bold', color='coral')
        ax2.text(0.5, 0.1, 'Total Ratings', fontsize=14, ha='center', va='center', color='gray')
        ax2.axis('off')
        
        # 3. Total users
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, f'{ratings_df["user_id"].nunique():,}', fontsize=40, ha='center', va='center',
                fontweight='bold', color='seagreen')
        ax3.text(0.5, 0.1, 'Active Users', fontsize=14, ha='center', va='center', color='gray')
        ax3.axis('off')
        
        # 4. Rating distribution
        ax4 = fig.add_subplot(gs[1, 0])
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        ax4.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='white')
        ax4.set_xlabel('Rating')
        ax4.set_ylabel('Count')
        ax4.set_title('Rating Distribution', fontweight='bold')
        
        # 5. Top 5 books
        ax5 = fig.add_subplot(gs[1, 1])
        top5 = book_counts.nlargest(5, 'num_ratings')
        top5_titles = []
        for bid in top5['book_id']:
            book = books_df[books_df['book_id'] == bid]
            if not book.empty:
                title = book.iloc[0]['title'][:15]
                top5_titles.append(title)
            else:
                top5_titles.append(f"Book {bid}")
        
        ax5.barh(range(len(top5)), top5['num_ratings'], color='coral')
        ax5.set_yticks(range(len(top5)))
        ax5.set_yticklabels(top5_titles)
        ax5.invert_yaxis()
        ax5.set_xlabel('Number of Ratings')
        ax5.set_title('Top 5 Popular Books', fontweight='bold')
        
        # 6. User activity
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(user_counts[user_counts <= 100], bins=30, color='seagreen', 
                edgecolor='white', alpha=0.7)
        ax6.set_xlabel('Ratings per User')
        ax6.set_ylabel('Number of Users')
        ax6.set_title('User Activity (≤100 ratings)', fontweight='bold')
        
        plt.suptitle('📚 Book Recommender System - Data Overview', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'summary_dashboard.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {filepath}")
        
        plt.show()
        return fig
    
    def run_all_visualizations(self, books_df, ratings_df, recommender=None):
        """
        Run all visualizations.
        
        Args:
            books_df (pd.DataFrame): Books dataframe
            ratings_df (pd.DataFrame): Ratings dataframe
            recommender: Optional BookRecommender instance
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        self.plot_rating_distribution(ratings_df)
        self.plot_popular_books(books_df, ratings_df)
        self.plot_ratings_per_user(ratings_df)
        self.plot_average_rating_distribution(books_df, ratings_df)
        self.create_summary_dashboard(books_df, ratings_df)
        
        if recommender is not None:
            self.plot_similarity_heatmap(recommender, sample_size=30)
        
        print("\n✅ All visualizations complete!")
        print(f"📁 Saved to: {self.output_dir}")


# Quick test
if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    
    try:
        # Load and preprocess data
        loader = DataLoader()
        books, ratings, _ = loader.load_all()
        
        preprocessor = DataPreprocessor()
        clean_books = preprocessor.preprocess_books(books)
        clean_ratings = preprocessor.preprocess_ratings(ratings, clean_books)
        
        # Create visualizations
        visualizer = DataVisualizer()
        visualizer.run_all_visualizations(clean_books, clean_ratings)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📥 Please run the preprocessing first.")
