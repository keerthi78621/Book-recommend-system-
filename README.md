# 📚 Book Recommender System

A complete machine learning project using Item-Based Collaborative Filtering with Cosine Similarity.

---

## 🎯 What Does This Project Do?

This system recommends books to users based on their preferences. 

**Example:**
- If you liked "The Hobbit", it recommends "The Lord of the Rings" (same author, similar fantasy genre)
- If you liked "Harry Potter", it recommends other Harry Potter books

**How it works:**
1. 📊 Collects user ratings for books
2. 🔗 Creates a matrix of users and books
3. 📐 Calculates similarity between books using cosine similarity
4. 🎁 Recommends books similar to what you liked!

---

## 🚀 How to Run

### Option 1: Web Interface (Best for Presentation) 🌐

**For Faculty/Presentation:**
1. Go to folder: `simple_web`
2. Double-click: `index.html`
3. Opens in your browser!

**Features:**
- Beautiful purple gradient design
- Interactive search
- Book cards with ratings
- No installation needed!

---

### Option 2: Terminal Version 💻

**Quick Demo:**
```bash
python book_recommender_simple.py
```

**What you see:**
- Text-based output
- Dataset statistics
- Recommendations in terminal
- Interactive search mode

---

### Option 3: Full Python Version 🐍

**With Real Dataset:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy

python run_recommender.py
```

**Features:**
- Uses 10,000 books
- 981,756 ratings
- 53,424 users
- Data visualizations

---

### Option 4: Advanced Web App (Streamlit) 🎨

```bash
pip install streamlit

streamlit run web_app.py
```

Opens at: http://localhost:8501

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `simple_web/index.html` | 🌐 Web version (double-click to open!) |
| `book_recommender_simple.py` | 💻 Simple terminal version |
| `run_recommender.py` | 🐍 Full version with real data |
| `web_app.py` | 🎨 Streamlit web app |
| `recommender.py` | Core recommendation engine |
| `data_preprocessor.py` | Data cleaning |
| `visualization.py` | Charts and graphs |
| `data_loader.py` | Load datasets |
| `README.md` | This file |

---

## 📊 Dataset

**Real Dataset (Goodbooks-10k):**
- Books: 10,000
- Ratings: 981,756
- Users: 53,424
- Source: Kaggle

**Sample Dataset (Demo):**
- Books: 10
- Ratings: 40
- Users: 10

---

## 🧠 How It Works (Explain to Faculty)

### Step 1: Data Collection
```
User 1: Rated "The Hobbit" = 5 stars
User 1: Rated "Harry Potter" = 4 stars
User 2: Rated "The Hobbit" = 5 stars
User 2: Rated "Lord of the Rings" = 5 stars
```

### Step 2: Create User-Item Matrix
```
         | Hobbit | Harry Potter | Lord of Rings
User 1   |   5    |      4       |      -
User 2   |   5    |      -        |      5
```

### Step 3: Calculate Cosine Similarity
```
Hobbit ↔ Lord of Rings: 0.95 (Very similar!)
Harry Potter ↔ Hobbit: 0.68 (Somewhat similar)
```

### Step 4: Make Recommendations
If you liked "The Hobbit", recommend "Lord of the Rings" (highest similarity)!

---

## 📈 Algorithms Used

### 1. Item-Based Collaborative Filtering
- Instead of finding similar users, find similar items (books)
- More stable than user-based filtering
- Works well for large datasets

### 2. Cosine Similarity
```python
cos(θ) = (A · B) / (||A|| × ||B||)
```

Measures the angle between two rating vectors:
- High similarity (0.9) = Books rated similarly by users
- Low similarity (0.1) = Books rated differently

---

## 🎓 What to Tell Faculty

### "Sir/Ma'am, this project demonstrates:

1. **Machine Learning Concepts**
   - Collaborative filtering algorithm
   - Cosine similarity for recommendations
   - Data preprocessing and cleaning

2. **Practical Applications**
   - Same technology used by Amazon, Netflix, Spotify
   - Real-world recommendation systems

3. **Technical Skills**
   - Python programming
   - Data analysis with Pandas
   - Machine learning with Scikit-learn
   - Data visualization

4. **Results**
   - Successfully recommends similar books
   - Demonstrated with real-world dataset
   - Interactive web interface for demo"

---

## 📋 Sample Output

### Terminal Version:
```
📚 RECOMMENDATIONS FOR 'THE HOBBIT'
================================================================

1. 📖 The Fellowship of the Ring
   Author: J.R.R. Tolkien
   Rating: 4.45/5 | Similarity: 0.847

2. 📖 The Two Towers
   Author: J.R.R. Tolkien
   Rating: 4.43/5 | Similarity: 0.832

3. 📖 The Return of the King
   Author: J.R.R. Tolkien
   Rating: 4.45/5 | Similarity: 0.821
```

### Web Interface:
- Beautiful cards with book titles
- Star ratings displayed
- Similarity percentages
- Popular books sidebar

---

## 🔧 Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
streamlit (optional for web app)
```

---

## 📝 License

Educational project - Free to use!

---

## 👨‍💻 Author

Created as a mini college project for learning machine learning fundamentals.

---

**Happy Reading! 📚✨**
