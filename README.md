# 📊 Data Analysis Dashboard

A free, no-code/low-code web application for data analysis, hypothesis testing, machine learning, and sensory analysis. Built with Streamlit and designed for students and researchers without programming experience.

## Features

✅ **Data Upload & Validation**
- Excel and CSV file upload
- Automatic data format checking
- Missing value detection
- Variable type classification (continuous, categorical, binary)

✅ **Exploratory Data Analysis (EDA)**
- Summary statistics
- Distribution analysis (histograms, density plots)
- Correlation matrices and heatmaps
- Box plots and scatter plots
- Outlier detection

✅ **Hypothesis Testing**
- t-tests (independent samples)
- ANOVA (one-way)
- Mann-Whitney U test (non-parametric)
- Chi-Square test (categorical)
- Correlation tests (Pearson/Spearman)
- Normality tests (Shapiro-Wilk)

✅ **Sensory Analysis**
- Descriptive sensory profiling
- Hedonic (liking) analysis
- Attribute ratings summary
- Panelist performance evaluation
- Product comparison

✅ **Machine Learning**
- Classification models:
  - Logistic Regression
  - Random Forest
- Regression models:
  - Linear Regression
  - Random Forest
- Train/test split
- Model evaluation metrics
- Feature importance analysis

✅ **AI Model Recommender**
- Expert system (zero cost - no API calls)
- Recommends appropriate models based on:
  - Analysis type
  - Data characteristics
  - Dataset size and complexity

## Installation & Local Testing

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/data-analysis-dashboard.git
cd data-analysis-dashboard
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app locally
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deployment on Streamlit Cloud (Free!)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Initial commit: Data Analysis Dashboard"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect to your GitHub account (one-time setup)
4. Select:
   - Repository: `data-analysis-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"

✨ **Your dashboard is now live and shareable!**

### Share with Students
Give them the Streamlit Cloud URL (looks like: `https://your-username-data-analysis-dashboard.streamlit.app`)

- ✅ No software installation needed
- ✅ Works on any device with a browser
- ✅ Completely free
- ✅ Data is not stored on the server (users upload/download locally)

## Usage Guide

### For Students

1. **Upload Data**
   - Click "📤 Upload Data"
   - Select your Excel or CSV file
   - Verify the preview looks correct

2. **Validate Data**
   - Go to "✅ Data Validation"
   - Review column types and missing values
   - Specify which variables are dependent (target) vs independent (features)

3. **Explore Data**
   - Select "🔍 Exploratory Analysis"
   - Choose analysis components (distributions, correlations, etc.)
   - Interpret the visualizations

4. **Test Hypotheses**
   - Go to "📊 Hypothesis Testing"
   - Select the appropriate test for your question
   - Interpret p-values and effect sizes

5. **Build Models**
   - Select "🤖 Machine Learning"
   - Choose a model
   - Train and evaluate

6. **Get Recommendations**
   - Use "💡 Model Recommender" to understand which models to use
   - Learn when and why to use each approach

### For Instructors

**Suggested Learning Path:**
1. Week 1: EDA (understanding data)
2. Week 2: Hypothesis Testing (answering questions)
3. Week 3: Sensory Analysis (if applicable to your course)
4. Week 4: Machine Learning (prediction)

**Sample Datasets for Teaching:**
- Create sample Excel files with different data characteristics
- Include "clean" and "messy" versions to teach data validation
- Share via Google Drive or in the README

## Customization

### Adding Custom Analysis
Edit the modules to add:
- Additional statistical tests
- New machine learning models
- Domain-specific analyses

Example structure:
```python
# modules/your_analysis.py
import streamlit as st

def run_your_analysis(df):
    st.markdown("### Your Custom Analysis")
    # Add your code here
```

Then import and use in `app.py`.

### Styling
Customize colors and fonts in `app.py` by modifying the CSS in:
```python
st.markdown("""
    <style>
    /* Your CSS here */
    </style>
""", unsafe_allow_html=True)
```

## Technical Details

### Architecture
```
data-analysis-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── data_validation.py         # Data format checking
│   ├── eda.py                     # Exploratory analysis
│   ├── hypothesis_tests.py        # Statistical tests
│   ├── sensory_analysis.py        # Sensory evaluation
│   ├── ml_models.py               # ML pipeline
│   └── model_recommender.py       # Expert system for recommendations
└── README.md
```

### Dependencies
- **streamlit**: Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib/seaborn**: Visualization
- **scipy**: Statistical functions
- **openpyxl**: Excel file reading

### Performance Notes
- Streamlit Cloud free tier has some limitations:
  - Up to 3 apps
  - Automatic sleep if inactive for 1 hour
  - Reopens instantly on next access
- For large datasets (>100MB), performance may be slow

## FAQs

**Q: Is this really free?**
A: Yes! Streamlit Cloud has a free tier. You only pay if you need advanced features (private apps, more compute).

**Q: Will my data be stored?**
A: No. Data is only in memory while your session is open. Upload files are not saved to the server.

**Q: Can I modify the code?**
A: Yes! Fork the repo, make changes, and redeploy. Since it's Python, you can add any analysis you want.

**Q: What if I have 1000 students?**
A: You can share the same app URL. Streamlit Cloud can handle multiple concurrent users. For heavy traffic, consider upgrading or deploying on another platform.

**Q: How do I add my own statistical test?**
A: Edit `modules/hypothesis_tests.py`, add your test function, then update `app.py` to call it.

## Troubleshooting

### App won't deploy
- Check that all files are committed to GitHub
- Verify `requirements.txt` has all dependencies
- Check for syntax errors in Python files

### Slow performance
- Streamlit Cloud free tier can be slower than local
- Consider upgrading to paid tier or deploying on AWS/GCP
- Optimize code by reducing DataFrame operations

### Import errors
- Make sure all dependencies are in `requirements.txt`
- Verify module file names match imports in `app.py`

## Contributing

Want to improve this dashboard? 
1. Fork the repo
2. Create a feature branch
3. Submit a pull request

Suggestions:
- Additional statistical tests
- More machine learning models
- Advanced visualization options
- Better pedagogical explanations

## License

MIT License - Free to use, modify, and share

## Support

For questions or issues:
1. Check the README and FAQs
2. Review comments in the source code
3. Create a GitHub issue

## Changelog

### v1.0 (Initial Release)
- EDA functionality
- Hypothesis testing suite
- Machine learning pipeline
- Sensory analysis tools
- Model recommender system
- Streamlit Cloud deployment ready

---

**Built with ❤️ for educators and students**

Made with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), and [pandas](https://pandas.pydata.org)