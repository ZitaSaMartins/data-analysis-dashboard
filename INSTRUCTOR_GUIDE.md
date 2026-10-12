# Instructor Guide: Using the Data Analysis Dashboard

This guide helps you integrate the Data Analysis Dashboard into your curriculum and teach data analysis to students without programming background.

## Course Integration

### Suggested Learning Path

#### Week 1-2: Data Preparation & Exploration
- **Topics**: Data types, validation, missing values, outliers
- **Dashboard sections**: Data Upload → Data Validation → EDA
- **Learning objective**: Students understand their data before analysis

#### Week 3-4: Descriptive Statistics
- **Topics**: Summary statistics, distributions, correlation
- **Dashboard sections**: EDA (Summary Statistics, Distributions, Correlations)
- **Learning objective**: Describe data patterns numerically and visually

#### Week 5-6: Hypothesis Testing
- **Topics**: p-values, t-tests, ANOVA, effect sizes
- **Dashboard sections**: Hypothesis Testing
- **Learning objective**: Test research questions statistically

#### Week 7: Sensory Analysis (if applicable)
- **Topics**: Consumer preferences, sensory evaluation
- **Dashboard sections**: Sensory Analysis
- **Learning objective**: Analyze human perception data

#### Week 8: Machine Learning
- **Topics**: Prediction, classification, model evaluation
- **Dashboard sections**: Machine Learning + Model Recommender
- **Learning objective**: Build predictive models

---

## Sample Lesson Plans

### Lesson 1: "Understanding Your Data" (1-2 hours)

**Objective**: Students learn to upload data and identify issues

**In-class activity**:
1. Show students the sample dataset (`sample_data.csv`)
2. Have them upload it to the dashboard
3. Guide them through Data Validation tab
4. **Discussion questions**:
   - What data types are present?
   - Are there missing values?
   - Which variables are independent vs. dependent?

**Homework**:
- Students download a public dataset (e.g., from Kaggle)
- Upload and validate it using the dashboard
- Write a 1-page report on data quality

---

### Lesson 2: "Exploring Relationships" (1.5 hours)

**Objective**: Students create and interpret visualizations

**In-class activity**:
1. Use the sample dataset
2. Go to EDA section
3. Create distributions, correlations, box plots
4. **Discussion**:
   - What patterns do you see?
   - Which variables are related?
   - Any surprises in the data?

**Interpretation guide**:
- **Correlation r = 0.8**: Strong positive relationship
- **Correlation r = 0.2**: Weak relationship
- **Correlation r = -0.6**: Moderate negative relationship

**Homework**:
- Students explore their own dataset from Lesson 1
- Create 3 visualizations
- Write interpretations (what each plot shows)

---

### Lesson 3: "Testing a Hypothesis" (2 hours)

**Objective**: Students formulate and test a research question

**In-class activity**:
1. Introduce hypothesis framework:
   - H₀ (Null): No difference
   - H₁ (Alternative): There is a difference
   - α (significance level) = 0.05

2. Use sample dataset with hypothesis:
   - "Different products have different liking scores"
   - Go to Hypothesis Testing → ANOVA
   - Interpret p-value

3. **Discussion**:
   - Is p-value < 0.05? (Reject H₀ or fail to reject)
   - What does this mean for your research question?
   - How confident are you?

**Key points to emphasize**:
- p-value ≠ probability that H₀ is true
- p < 0.05 doesn't mean "very important"
- Effect size also matters (e.g., Cohen's d)

**Homework**:
- Students formulate a research question for their dataset
- Perform appropriate hypothesis test
- Write results in APA format

---

### Lesson 4: "Predicting with Machine Learning" (2-3 hours)

**Objective**: Students build and evaluate a predictive model

**In-class activity**:

1. **Conceptual intro** (20 min):
   - What is machine learning?
   - Difference between explanation (hypothesis testing) vs. prediction (ML)
   - Example: Predict product liking from sensory attributes

2. **Model recommender** (20 min):
   - Go to Model Recommender tab
   - Discuss why different models are recommended
   - Key factors: sample size, data characteristics

3. **Hands-on** (60 min):
   - Prepare data (specify dependent/independent variables)
   - Train a Logistic Regression model
   - Discuss:
     - Training accuracy vs. test accuracy
     - Why they might differ (overfitting?)
     - Feature importance

4. **Interpretation**:
   - "Model is 85% accurate" = Good?
   - Depends on:
     - Baseline accuracy (what if we always guessed the most common class?)
     - Class balance (imbalanced data can mislead)
     - Business cost of errors

**Homework**:
- Build two models on their dataset: simple and complex
- Compare accuracy, training time, interpretability
- Write: Which model would you use and why?

---

## Teaching Tips

### 1. Start with Messy Data
- Students learn validation by fixing real data issues
- Include:
  - Wrong column names
  - Mixed data types
  - Missing values
  - Outliers

**Example**: Create two versions of sample data:
- Clean version (for learning)
- Messy version (for practice)

### 2. Connect to Their Interests
- Food science students: Use sensory/taste data
- Agriculture: Crop yield, environmental conditions
- Business: Sales, customer satisfaction
- Health: Patient outcomes, treatment effects

### 3. Use Real Research Questions
Don't ask: "Test if there's a difference between groups"

Instead ask: "Your client claims Product B is better. Test if consumers really prefer it."

### 4. Encourage Exploration
- Sensory Analysis section is great for this
- Students can:
  - Try different groupings
  - Spot patterns in panelist performance
  - Compare products visually

### 5. Emphasize Interpretation
**Don't focus on**: "Click here, get a number"

**Do focus on**: "What does this number mean for your question?"

Example:
- ANOVA p-value = 0.002
- Student: "It's significant because p < 0.05"
- Better: "Products differ in liking (p = 0.002). Based on mean scores, Product A is preferred"

---

## Assessment Ideas

### Individual Assignments

**Assignment 1: Data Validation Report**
- Upload a dataset
- Identify and document issues
- Recommend fixes
- Grading: Thoroughness, accuracy

**Assignment 2: EDA Analysis**
- Create visualizations for a dataset
- Interpret each plot
- Write insights
- Grading: Accuracy + depth of interpretation

**Assignment 3: Hypothesis Test**
- Formulate research question
- Select appropriate test
- Run analysis
- Report results in APA format
- Grading: Appropriate test choice + correct interpretation

**Assignment 4: ML Project**
- Build predictive model
- Compare 2+ models
- Evaluate and interpret
- Written report
- Grading: Methodology + results + interpretation

### Group Projects

**Project: Sensory Evaluation Study**
- Groups collect sensory data from classmates
- Upload to dashboard
- Analyze descriptive sensory profiles
- Compare products
- Present findings
- Grading: Data quality + analysis + presentation

**Project: Real Dataset Analysis**
- Groups pick public dataset
- Develop research questions
- Use dashboard for EDA → Hypothesis Testing → ML
- Final presentation/report
- Grading: Rigor of analysis + clarity of communication

---

## Troubleshooting in Class

### Problem: Student uploads wrong file format

**Solution**:
- Dashboard accepts: Excel (.xlsx), CSV
- Show students how to "Save As" CSV in Excel
- Have them try again

### Problem: Data validation fails (column names with spaces/special characters)

**Solution**:
- Explain why: Python/SQL don't like spaces in names
- Show correct format: `Product_Name` not `Product Name`
- Have them fix in Excel locally

### Problem: Student confused by p-value

**Solution**:
- Use the **Model Recommender** to explain
- Show: "What is a p-value?" explanations in each test section
- Use analogy: "p-value = evidence against the null hypothesis"
- Give examples with real numbers

### Problem: ML model has low accuracy

**Solution**:
- Discuss possible reasons:
  - Too many classes (multi-class is harder)
  - Not enough data
  - Features don't relate to outcome
  - Model is too simple

- Suggest improvements:
  - Try different model
  - Collect more data
  - Engineer better features

---

## Common Misconceptions to Address

### 1. "Significant = Important"
- p < 0.05 ≠ big effect
- Emphasize: Check effect size (e.g., Cohen's d)
- Show: Two models: one with p = 0.04 (small effect), one with p = 0.001 (large effect)

### 2. "Higher R² always better"
- High R² can indicate overfitting
- Compare training vs. test accuracy
- Show: Why simple models sometimes generalize better

### 3. "ML is magic"
- Models only learn from data provided
- Garbage in, garbage out
- Need good data and good features

### 4. "More data = always better"
- True up to a point
- Diminishing returns
- Data quality > quantity

### 5. "My model = truth"
- All models are wrong
- Use for guidance, not gospel
- Validate with real-world outcomes

---

## Student Feedback & Iteration

### Weekly Reflection
Ask students:
- What was confusing?
- What would help you learn better?
- Any features you'd like?

### Adapt the Dashboard
- Add more example visualizations
- Create tooltips explaining concepts
- Add video tutorials (optional)

### Open Issues for Contributing
If students are tech-savvy:
- Let them contribute features
- Add statistical tests they suggest
- Improve documentation

---

## Resources to Share with Students

### Conceptual Learning
- StatQuest YouTube channel (Josh Starmer)
- Khan Academy: Statistics
- OpenIntro Statistics (free book)

### Hands-on Practice
- Kaggle datasets (public)
- Our World in Data
- Google Trends data

### Python/Statistics
- W3Schools Python
- Real Python tutorials
- Stack Overflow (when they get errors)

---

## Office Hours / Support

### Set up a dashboard walkthrough
- Show EDA on real dataset
- Answer questions about interpretation
- Debug issues

### Create a FAQ document
- Screenshot common questions
- Explain each step
- Post on course portal

### Use example datasets
- Provide clean version for learning
- Messy version for practice
- Include codebook (variable definitions)

---

## Scalability

### Small Class (10-30 students)
- Share same Streamlit Cloud link
- All students use same deployment
- No issues with concurrent usage

### Large Class (100+ students)
- Streamlit Cloud can handle
- May be slightly slower
- Consider upgrading tier or running multiple instances

### Very Large (1000+ students)
- Deploy on own server (AWS/GCP)
- Or use Streamlit for Teams (paid)
- Explore alternative platforms (Shiny, Voila, etc.)

---

## Instructor Customization

### Adding Your Own Data
1. Create sample datasets relevant to your field
2. Include in GitHub repo under `data/samples/`
3. Reference in README with descriptions

### Extending Dashboard
1. Add new statistical tests in `modules/hypothesis_tests.py`
2. Add new ML models in `modules/ml_models.py`
3. Modify explanations and tooltips
4. Redeploy (auto-updates on GitHub push)

### Creating Assignments
Use template:
```markdown
# Assignment: [Topic]

**Objective**: Students will [learn X]

**Instructions**:
1. Upload the provided dataset
2. Use the dashboard to [do Y]
3. Report findings in [format Z]

**Grading rubric**:
- Correct methodology (40%)
- Accurate interpretation (40%)
- Clarity of writing (20%)
```

---

## Final Thoughts

The dashboard is a **tool for learning**, not a replacement for teaching:
- Your explanations matter
- Discussions about interpretation matter
- Connections to real problems matter

Use the dashboard to:
- ✅ Reduce barriers (no programming needed)
- ✅ Increase engagement (click, see results immediately)
- ✅ Encourage exploration (try different analyses)
- ✅ Build intuition (patterns become visual)

Focus on helping students **think like data analysts**, not just "use software."

---

**Happy teaching! 📊**

Have suggestions or stories about using this in class? Share them!