# 🎓 Student Performance Prediction Using Linear Regression

A BCA college project that predicts a student's score (marks / performance index) from details such as
study hours, previous scores, sleep hours and extracurricular activities. It uses **Linear Regression**
(the only machine-learning model in this project) and shows everything in an interactive **Streamlit**
dashboard.

**Tools:** Python · VS Code · Streamlit · Pandas · Scikit-learn · Matplotlib · GitHub

---

## 1. What the project does

| Step | Page in the dashboard | What happens |
|---|---|---|
| 1 | **Upload Data** | You upload your Kaggle CSV file. The app shows the first rows, column types, empty cells and duplicates. |
| 2 | **Clean Data** | You pick the score column. The app fixes empty cells, duplicates, extra spaces, Yes/No text, and (optionally) extreme values. |
| 3 | **Visualize** | Charts: score distribution, correlation heatmap, input vs score scatter plot, group comparison, empty cells. |
| 4 | **Train & Evaluate** | Linear Regression is trained on 80% of students and tested on the other 20%. Shows R², MAE, MSE, RMSE, the equation and charts. |
| 5 | **Predict Score** | Type in one student's details and get a predicted score, or upload a CSV of many students and download their predictions. |

---

## 2. Project folder

```
student-performance-prediction/
├── app.py                  ← the Streamlit dashboard (run this)
├── train_model.py          ← trains the model from the terminal (optional)
├── requirements.txt        ← list of libraries to install
├── README.md               ← this guide
├── run_app.bat             ← Windows: double-click to install + run
├── run_app.sh              ← Mac/Linux: install + run
├── .gitignore              ← files GitHub should ignore
├── .streamlit/config.toml  ← dashboard colours and upload size
├── .vscode/                ← VS Code "Run" button settings
├── data/
│   └── sample_students.csv ← practice dataset (replace with your Kaggle file)
├── models/                 ← the trained model is saved here
└── src/
    ├── __init__.py
    ├── data_utils.py       ← loading + cleaning the data
    ├── model.py            ← Linear Regression training, testing, prediction
    └── visuals.py          ← all charts
```

---

## 3. Install the software (one time only)

1. **Install Python 3.10 or newer** from <https://www.python.org/downloads/>.
   On Windows, tick **"Add python.exe to PATH"** on the first screen of the installer.
2. **Install VS Code** from <https://code.visualstudio.com/>.
3. Open VS Code → click the **Extensions** icon (four squares on the left) → search **Python** → install the one by Microsoft.
4. **Install Git** from <https://git-scm.com/downloads> (needed only for uploading to GitHub).

---

## 4. Run the project (no coding needed)

### Easiest way – Windows
1. Unzip the project folder.
2. Double-click **`run_app.bat`**.
3. Wait while it installs the libraries (only slow the first time).
4. Your browser opens at **http://localhost:8501** with the dashboard.

### Easiest way – Mac / Linux
Open a terminal in the folder and type: `bash run_app.sh`

### Inside VS Code
1. VS Code → **File → Open Folder…** → choose `student-performance-prediction`.
2. Open the terminal: **Terminal → New Terminal**.
3. Copy-paste these lines one by one and press Enter after each:

   **Windows**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

   **Mac / Linux**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Next time you only need the 2nd and 4th lines (activate, then `streamlit run app.py`).
   Or press **F5** and choose **"Run Streamlit Dashboard"**.

To stop the dashboard, click in the terminal and press **Ctrl + C**.

> **What is `.venv`?** A "virtual environment" – a private box for this project's libraries so they
> don't mix with other Python projects on your computer.

---

## 5. Use your own Kaggle dataset

**Option A (easiest):** open the dashboard → **Upload Data** → drag your Kaggle CSV into the box.

**Option B:** copy your CSV into the `data/` folder, then train from the terminal:
```
python train_model.py --data data/YOUR_FILE_NAME.csv
```
Add `--target "Column Name"` if it picks the wrong score column.

The app works with any student CSV where:
- each row is one student,
- one column holds the score/marks (a number), and
- other columns are inputs (numbers, Yes/No, or text categories like gender).

It has been tested with both popular Kaggle formats:
- *Student Performance (Multiple Linear Regression)* – `Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Sample Question Papers Practiced, Performance Index`
- *Students Performance in Exams* – `gender, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score, writing score`

`data/sample_students.csv` is a practice file in the first format (made-up numbers, with some empty cells
and duplicate rows added on purpose so you can see the cleaning work). Use your real Kaggle file for
your final submission.

---

## 6. How every step works – in simple words

### Step 1 – Upload
`load_csv()` in `src/data_utils.py` reads the file with **Pandas** and turns it into a **DataFrame**
(a table with rows and columns, like Excel). It handles files that use `;` instead of `,` and files saved
with different text encodings.

### Step 2 – Data cleaning
Real data is messy. `clean_data()` fixes it in this order:
1. **Tidy column names** – `" Sleep Hours "` becomes `"Sleep Hours"`.
2. **Remove index columns** like `Unnamed: 0` that Excel sometimes adds.
3. **Fake empty values** – `?`, `N/A`, `null`, `-` are treated as empty.
4. **Remove completely empty rows/columns.**
5. **Numbers stored as text** are turned into real numbers.
6. **Yes/No → 1/0** – the model can only do maths with numbers.
7. **Rows without a score are removed** – the model can't learn from a student with no score.
8. **Duplicate rows are removed** – the same student counted twice would bias the model.
9. **Empty cells are filled** – numbers with the **median** (the middle value), Yes/No and text with the
   **most common value**.
10. **Outliers (optional)** – very extreme values are removed using the **IQR rule**
    (anything far outside the middle 50% of values).

### Step 3 – Visualization (`src/visuals.py`, Matplotlib)
- **Histogram** – how many students got each range of scores.
- **Correlation heatmap** – numbers from −1 to +1 showing how strongly two columns move together.
  +1 = both rise together, −1 = one rises while the other falls, 0 = no link.
- **Scatter plot + trend line** – each dot is a student; the line shows the overall trend.
- **Box plot** – compares scores between groups (e.g. male vs female).
- **Empty-cell chart** – which columns had missing data before cleaning.

### Step 4 – Model training (`src/model.py`, Scikit-learn)
**Linear Regression** finds the best straight-line equation:

```
Score = b0 + b1 × Hours Studied + b2 × Previous Scores + b3 × Sleep Hours + ...
```
- `b0` is the **intercept** (the starting value).
- `b1, b2, …` are **coefficients** – how many marks the score goes up (or down) when that input goes up by 1.
- The model chooses the numbers that make the total squared error as small as possible
  (the "least squares" method).

Before training:
- **Train/test split** – 80% of students are used to learn, 20% are hidden and used to test.
  This shows how the model does on students it has never seen.
- **One-hot encoding** – text columns like `gender` become 0/1 columns (`gender_male`, `gender_female`)
  so they can go into the equation.

Everything is wrapped in a Scikit-learn **Pipeline**, so new data is always prepared the same way.

### Step 5 – Evaluation
| Metric | Meaning in simple words | Good value |
|---|---|---|
| **R²** | What share of the differences in scores the model explains | Close to 1 (0.9 = 90%) |
| **MAE** | Average number of marks the prediction is off by | Small |
| **MSE** | Average of squared errors (big mistakes count more) | Small |
| **RMSE** | Square root of MSE, back in "marks" units | Small |
| **Cross-validation R²** | The train/test check repeated 5 times on different splits | Steady and close to test R² |

If **training R²** is much higher than **test R²**, the model has memorised the training data
(**overfitting**). Linear Regression rarely overfits because it is a simple model.

Charts:
- **Actual vs predicted** – dots near the dashed line = accurate predictions.
- **Residuals** – errors spread evenly above and below 0 = the model is not biased.
- **Coefficient chart** – which inputs raise or lower the score the most.

The trained model is saved to `models/student_model.joblib` with **Joblib**, so the Predict page can use
it even after you restart the app.

### Step 6 – Prediction
- **One student:** fill in the form → the app puts the values into the equation and shows the
  predicted score, where it falls between the lowest and highest score, and how much each input
  added or removed.
- **Many students:** upload a CSV with the same input columns (download the template from the page)
  → get a predicted score for every row → download the result as CSV.

---

## 7. Upload the project to GitHub

1. Create a free account at <https://github.com>.
2. Click **+** (top right) → **New repository** → name it `student-performance-prediction` →
   **Public** → do **not** tick "Add a README" → **Create repository**.
3. In the VS Code terminal (inside the project folder) paste these lines one by one.
   Replace `YOUR-USERNAME` with your GitHub username:
   ```
   git init
   git add .
   git commit -m "Student Performance Prediction using Linear Regression"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
   git push -u origin main
   ```
4. If Git asks who you are, run these first (with your details), then repeat the `git commit` line:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
5. A browser window may open asking you to sign in to GitHub – sign in and allow access.
6. Refresh your GitHub page – your project is online.

Later changes:
```
git add .
git commit -m "Describe what you changed"
git push
```

**Optional – put the dashboard online for free:** go to <https://share.streamlit.io>, sign in with GitHub,
choose your repository, set the main file to `app.py`, and click **Deploy**. You get a public link you
can show your examiner.

---

## 8. Common problems

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add python.exe to PATH**. On Mac use `python3`. |
| `streamlit: command not found` | Activate the virtual environment first (`.venv\Scripts\activate` or `source .venv/bin/activate`), or use `python -m streamlit run app.py`. |
| PowerShell says "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, type `Y`, then activate again. Or use **Command Prompt** instead of PowerShell. |
| "The score column contains text" | On the Clean Data page, choose the column that holds marks (numbers). |
| "These columns are missing from the new file" | The batch CSV must have the same input column names as the training file. Use the template button. |
| Predict page says "No trained model yet" | Go to **Train & Evaluate** and click **Train**. |
| Browser didn't open | Open <http://localhost:8501> yourself. |

---

## 9. Possible viva questions

- **Why Linear Regression?** The score is a number (not a category), and the inputs have a mostly
  straight-line relationship with it. It is simple, fast and easy to explain.
- **What is the target and what are the features?** The target is the score we predict; features are
  the inputs (hours studied, previous scores …).
- **Why split into train and test?** To check the model on students it has not seen, which shows how
  it will work in real life.
- **What does R² = 0.98 mean?** The model explains 98% of the differences in students' scores.
- **What is a coefficient?** How much the score changes when that input increases by 1, keeping the
  other inputs the same.
- **Why fill missing values with the median?** The median is not pulled up or down by a few extreme values.
- **What is one-hot encoding?** Turning a text column into several 0/1 columns so it can be used in an equation.
- **Limitations:** assumes straight-line relationships; can predict slightly above the maximum or below
  the minimum score for unusual inputs; only as good as the data it learns from.
- **Future scope:** collect more features (attendance, assignments), compare with other models,
  connect to a college database.
