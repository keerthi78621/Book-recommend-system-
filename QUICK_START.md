# Quick Start Guide

## How to Run the Book Recommender System

### Step 1: Install Required Packages

Open your terminal/command prompt and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Step 2: Run the Project

**Option A: With Sample Data (Easiest - No Download Needed)**

Simply run:
```bash
python book_recommender_simple.py
```

The system will automatically generate sample data and run. You'll see:

```
BOOK RECOMMENDER SYSTEM
======================================
Item-Based Collaborative Filtering with Cosine Similarity
======================================
...
Dataset Summary:
  Books: 10
  Ratings: 40
  Users: 10
  Average Rating: 4.53
...
BOOK RECOMMENDATIONS FOR: 'The Hobbit'
======================================

1. Book: The Fellowship of the Ring
   Author: J.R.R. Tolkien
   Rating: 4.45/5
   Similarity: 0.943
...
```

### Step 3: Interactive Mode

After the demo, you can enter book names:
```
Enter a book title (or 'quit' to exit): Harry Potter
```

### If You Want Real Data

**Option B: Download from Kaggle**

1. Go to: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k
2. Click "Download"
3. Extract the ZIP file
4. Copy `books.csv` and `ratings.csv` to a folder named `data`
5. Run: `python book_recommender_simple.py`

**Option C: Generate Sample Data**

```bash
# Create a 'data' folder first
mkdir data

# Generate sample data
python sample_data_generator.py --minimal

# Run the project
python book_recommender_simple.py
```

### Troubleshooting

**Error: Module not found**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**Error: No data folder**
```bash
mkdir data
python sample_data_generator.py --minimal
```

**Error: File not found**
The system will automatically create sample data. Just run:
```bash
python book_recommender_simple.py
```

### Quick Demo (Copy-Paste This)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy -q
python book_recommender_simple.py
```

That's it! The program will run and show you recommendations.
