# 📊 Data Analysis Dashboard

A free, no-code/low-code web application for data analysis, hypothesis testing, machine learning, and sensory analysis. Built with Streamlit and designed for students and researchers without programming experience.

## Features

✅ **Data Upload & Validation**
- Excel and CSV file upload, with the loaded filename always shown
- Rename Variables page — rename any column, with full support for accents, special
  characters, and any language script (e.g. 'Température', '甘み')
- Automatic data format checking (non-blocking warnings vs. blocking errors)
- Missing value detection
- Variable type classification (continuous, categorical, binary)
- Independent/Dependent variable role selection, reused across Explore Data and ML

✅ **Exploratory Data Analysis (EDA)**
- Organized into five Explore Data sub-menus: Distributions, Correlations,
  Comparisons, Part-to-Whole & Hierarchical, and Temporal
- 19 interactive Plotly chart types (Bar, Grouped Bar, Radar, Scatter, Bubble,
  Correlation Heatmap, Contour Plot, Pie, Donut, Treemap, Sunburst, Line, Area,
  Stacked Area, Histogram, Categorical Bar, Box Plot, Violin Plot, Strip Plot) plus
  a Summary Statistics table
- ⭐ Recommended charts based on your selected variables, each with a live preview
  thumbnail before you generate anything
- Fully configurable: theme, color palette, color scale, font, and plot height,
  plus per-chart color/title customization after generating

✅ **Statistical Inference**
- Assumption checks for one or several variables: Shapiro-Wilk,
  Kolmogorov-Smirnov, Levene's, Bartlett's, and Fligner-Killeen tests
- Distribution testing offers Automatic, Shapiro-Wilk,
  D'Agostino-Pearson K², Anderson-Darling, and Kolmogorov-Smirnov options,
  with a sample-size-based recommendation for every selected variable
- Histogram with a fitted normal curve, normal Q-Q plot, and box plot for
  visual distribution inspection alongside the statistical test
- Variance testing offers Automatic, Levene, Bartlett, and Fligner-Killeen
  options, with recommendations based on prior distribution results, group
  sizes, and extreme-outlier checks
- Numeric-variable selections are shared automatically between Test
  Distribution and Test Variance, so they only need to be chosen once
- Grouped box plots, violin plots, variance bars, and group-level variance and
  standard-deviation details support visual variance inspection
- **Suggested Analysis** matches distribution and variance results by variable,
  explains the supported inference path, and opens the recommended test with
  the numeric and grouping variables already selected
- Suggested Analysis uses the same shared variable selection and identifies
  every variable with a missing distribution or variance result, with buttons
  to complete the missing checks
- Recommendations cover independent groups, paired or repeated observations,
  one-sample questions, and numeric relationships
- Tests are grouped by research question in nested, collapsible sub-menus:
  Two Independent Groups, 3+ Groups (Single Factor), Multiple Factors,
  Repeated Measures, Correlations, Regression, and Categorical
- Parametric tests: one-sample, independent, and paired t-tests; one-way and
  two-way ANOVA; repeated-measures ANOVA; Pearson correlation; and linear regression
- Non-parametric tests: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis,
  Friedman, Mood's median, Spearman correlation, and Kendall's tau
- One-way ANOVA, Kruskal-Wallis, and Mood's median analyses offer selectable
  pairwise follow-ups with multiple-comparison correction and compact grouping
  letters. Parametric options include Tukey HSD, Bonferroni, Scheffé, Fisher's
  LSD, and Holm-adjusted Welch comparisons; non-parametric options include
  Dunn and Conover-Iman with Holm correction
- These 3+ group analyses support selecting multiple numeric outcome variables
  and report each outcome separately using the same grouping variable
- Chi-square testing for relationships between categorical variables
- Plain-language interpretations plus effect sizes where applicable, including
  Cohen's d and eta squared
- Interactive **How to Choose Tests** guide that compares parametric and
  non-parametric options for one, two, or three-or-more groups and opens the
  selected test directly

✅ **Hierarchical Navigation**
- Home plus six collapsible sections: Prepare Data, Explore Data, Analyze
  Relationships, Sensory Analysis, Advanced Analysis, and Help
- Sub-menus remain collapsed until selected and automatically reopen around the
  active page
- The sidebar always shows the uploaded filename and dataset dimensions

✅ **Sensory Analysis**
- ISO-aligned discrimination workflows: triangle, paired comparison, duo-trio,
  tetrad, and ranking tests
- Descriptive QDA/sensory profiling and temporal sensory methods
- Panel discrimination, repeatability, agreement/consensus, and scale-use checks
- Product characterization with inferential comparisons and product maps
- CATA frequency testing and product-attribute mapping
- JAR distributions and penalty analysis
- Hedonic testing, consumer segmentation, and internal/external preference maps
- Advanced multivariate tools: PCA, GPA, STATIS, hierarchical clustering, MFA,
  and PLS regression
- Twenty-four nested sensory routes keep each teaching workflow focused and
  automatically reopen the active submenu

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
   - Click "Upload data" in the sidebar
   - Select your Excel or CSV file — the filename stays visible once loaded
   - Verify the preview looks correct

2. **Rename Variables (optional)**
   - Click "Rename variables" in the sidebar
   - Edit column names directly in the table — accents, special characters, and
     any language script are supported
   - Renames automatically carry over to every other page (variable roles, EDA, ML)

3. **Check Data Quality**
   - Go to "Check data quality"
   - Review column types and missing values
   - Specify which variables are independent (features) vs. dependent (target)

4. **Explore Data**
   - Pick a category directly under Explore (Comparison, Correlation,
     Part-to-Whole & Hierarchical, Temporal, or Distribution)
   - Pick from the recommended (⭐) or other chart types, using the live preview
     to decide
   - Generate the analysis, then fine-tune colors/titles per chart

5. **Statistical Inference**
   - Open **Analyze Relationships** and run Test Distribution and Test Variance
     when the test you plan to use requires those assumptions
   - Open **Suggested Analysis** to combine the matching results and review the
     recommended parametric, adjusted-parametric, or non-parametric path
   - Choose a test under Two Independent Groups, 3+ Groups (Single Factor),
     Multiple Factors, Repeated Measures, Correlations, Regression, or Categorical
   - If you are unsure, open **Help → How to Choose Tests** and follow the
     interactive guide
   - Interpret p-values and effect sizes using the built-in guidance

6. **Build Models**
   - Select "Build predictions"
   - Choose a model
   - Train and evaluate

7. **Get Recommendations**
   - Use "Get recommendations" to understand which models to use
   - Learn when and why to use each approach

### For Instructors

**Suggested Learning Path:**
1. Week 1: EDA (understanding data)
2. Week 2: Statistical Inference (answering questions)
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
│   ├── sidebar_navigation.py      # Hierarchical navigation and test guide
│   ├── hypothesis_tests.py        # Legacy statistical workflow
│   ├── hypothesis_tests_comprehensive.py # Current statistical test suite
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
- **matplotlib/seaborn**: Visualization (hypothesis testing, sensory analysis, ML)
- **plotly**: Interactive visualization (EDA charts)
- **scipy**: Statistical functions
- **statsmodels**: Two-way and repeated-measures ANOVA
- **openpyxl**: Excel file reading

### Theming
The sidebar and overall color palette are configured in `.streamlit/config.toml`
(separate `[theme]` and `[theme.sidebar]` sections). Edit this file to change the
accent colors without touching any Python code.

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
A: Add the test to `modules/hypothesis_tests_comprehensive.py`, register its page
handler in `app.py`, and add its navigation button in
`modules/sidebar_navigation.py`.

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

### v1.1
- Redesigned navigation with Home and six collapsible sections: Prepare Data,
  Explore Data, Analyze Relationships, Sensory Analysis, Advanced Analysis,
  and Help
- New Rename Variables page (Prepare section) with full Unicode/special-character
  support, propagated automatically to variable-role selections, EDA, and ML
- Uploaded filename is now shown persistently after upload
- EDA rebuilt on Plotly: 5 chart categories, 19 chart types, ⭐ recommendations,
  live previews, and per-chart color/title customization after generating
- Statistical inference reorganized by research question with nested,
  click-to-expand test groups
- Added a comprehensive test suite covering assumption checks, parametric and
  non-parametric group comparisons, correlations, regression, and categorical
  analysis
- Added an interactive decision tree under Help → How to Choose Tests
- Added direct sensory-analysis routes for descriptive, hedonic, and product
  comparison views
- Data validation no longer flags parentheses as invalid, and the Issues tab
  distinguishes blocking errors from non-blocking warnings
- Custom color theme via `.streamlit/config.toml`

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