# Where to Place Downloaded Files

## Folder Structure

Your project folder should look like this:

```
📁 Book Recommender System (your main project folder)
│
├── 📁 data                    ← CREATE THIS FOLDER
│   ├── 📄 books.csv          ← PASTE books.csv HERE
│   ├── 📄 ratings.csv         ← PASTE ratings.csv HERE
│   └── 📄 users.csv          ← (optional) PASTE users.csv HERE
│
├── 📁 output                  ← Created automatically
│   └── 📄 ...
│
├── 📄 book_recommender_simple.py
├── 📄 run_recommender.py
└── 📄 README.md
```

## Step-by-Step Instructions

### Step 1: Create the 'data' folder

1. Open your project folder (`Book recommened system`)
2. Right-click → New → Folder
3. Name it: `data`

### Step 2: Download files from Kaggle

1. Go to: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k
2. Click "Download" button
3. Wait for ZIP file to download
4. Extract the ZIP file

### Step 3: Copy files to 'data' folder

After extracting, you should see files like:
- books.csv
- ratings.csv
- users.csv (optional)

**Copy ALL these files** and paste them into the `data` folder you created.

### Step 4: Verify

Your structure should be:

```
📁 Book recommened system
├── 📁 data
│   ├── books.csv         ✓
│   ├── ratings.csv       ✓
│   └── users.csv         ✓ (optional)
├── 📄 book_recommender_simple.py
├── 📄 run_recommender.py
└── 📄 README.md
```

### Step 5: Run the project

```bash
python book_recommender_simple.py
```

## Important: Don't Paste in Wrong Place!

❌ WRONG:
```
📁 Desktop
└── 📄 books.csv          (Not here!)
```

✅ RIGHT:
```
📁 Book recommened system
└── 📁 data
    └── 📄 books.csv      (Inside the data folder!)
```

## If You Don't Want to Download

You don't need to download anything! Just run:

```bash
python book_recommender_simple.py
```

The program will automatically create sample data for you.

## Quick Command to Create Data Folder

Open terminal/command prompt and run:

```bash
mkdir data
python sample_data_generator.py --minimal
python book_recommender_simple.py
```

This will:
1. Create the data folder
2. Generate sample data
3. Run the project

No download required!
