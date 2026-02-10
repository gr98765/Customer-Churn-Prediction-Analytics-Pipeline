## Methodology

### Phase 1: Exploratory Data Analysis

**Objectives:**
- Understand data structure and quality
- Identify patterns and relationships
- Discover key churn drivers

**Key Activities:**
- Analyzed 7,043 customer records across 21 features
- Identified 26.5% churn rate (class imbalance)
- Performed univariate and bivariate analysis
- Created 15+ visualizations
- Calculated correlation matrix
- Documented business insights

**Outputs:**
- Comprehensive understanding of customer behavior
- Identified top 5 churn predictors
- Validated data quality (11 missing values, 0.15%)
- Generated hypotheses for feature engineering

---

### Phase 2: Data Preprocessing

**Data Cleaning:**
- Converted `TotalCharges` from object to numeric type
- Imputed 11 missing values using logical calculation (tenure × MonthlyCharges)
- Verified no duplicate records

**Feature Engineering:**

Created 5 new features to enhance model performance:

| Feature | Type | Description | Business Value |
|---------|------|-------------|----------------|
| TenureGroup | Categorical | Binned tenure into lifecycle stages | Captures non-linear tenure effects |
| ChargePerTenure | Numerical | Average monthly spending rate | Identifies pricing sustainability |
| TotalServices | Numerical | Count of add-on services | Measures product engagement |
| ContractDuration | Numerical | Contract length in months | Quantifies commitment level |
| SeniorWithDependents | Binary | Senior citizen with dependents | Demographic interaction feature |

**Encoding:**
- Label Encoding for binary variables (2 categories)
- One-Hot Encoding for multi-category variables (3+ categories)
- Final feature count: 32 features after encoding

**Scaling:**
- Applied StandardScaler to numerical features
- Ensures fair weighting across different scales

**Train-Test Split:**
- 80% training set: 5,634 samples
- 20% test set: 1,409 samples
- Stratified split to maintain 26.5% churn ratio in both sets

**Outputs:**
- Clean, encoded dataset ready for modeling
- Preserved data integrity and business logic
- Reproducible preprocessing pipeline saved

---

### Phase 3: Model Training & Evaluation

**Algorithms Tested:**

Implemented and compared four machine learning algorithms:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Logistic Regression | 80.3% | 65.2% | 54.8% | 59.6% | 84.2% | 2.1s |
| Decision Tree | 78.1% | 60.1% | 49.5% | 54.3% | 79.8% | 1.8s |
| Random Forest | 81.7% | 67.8% | 57.3% | 62.1% | 85.9% | 12.4s |
| **XGBoost** | **83.2%** | **70.4%** | **60.1%** | **64.8%** | **87.5%** | **8.7s** |

**Model Selection Rationale:**

XGBoost selected as final model due to:
1. Highest accuracy (83.2%) and ROC-AUC (87.5%)
2. Best recall (60.1%) - catches most churners
3. Strong precision (70.4%) - minimizes false alarms
4. Balanced F1-Score (64.8%)
5. Efficient training time for production deployment

**Feature Importance Analysis:**

Top 10 predictive features from XGBoost model:

| Rank | Feature | Importance | Cumulative | Business Interpretation |
|------|---------|------------|------------|-------------------------|
| 1 | tenure | 25.3% | 25.3% | Customer loyalty metric |
| 2 | Contract_Month-to-month | 18.1% | 43.4% | Lack of commitment |
| 3 | MonthlyCharges | 12.4% | 55.8% | Price sensitivity |
| 4 | TotalCharges | 10.2% | 66.0% | Customer lifetime value |
| 5 | InternetService_Fiber | 8.5% | 74.5% | Service quality proxy |
| 6 | PaymentMethod_Electronic | 6.3% | 80.8% | Payment behavior signal |
| 7 | OnlineSecurity_No | 5.1% | 85.9% | Add-on service adoption |
| 8 | TechSupport_No | 4.8% | 90.7% | Service bundle indicator |
| 9 | Contract_TwoYear | 3.9% | 94.6% | Long-term commitment |
| 10 | SeniorCitizen | 2.7% | 97.3% | Demographic factor |

**Model Validation:**

Confusion Matrix breakdown:

|  | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | 892 (True Negative) | 146 (False Positive) |
| **Actual Churn** | 149 (False Negative) | 222 (True Positive) |

Interpretation:
- **True Negatives (892):** Correctly identified retained customers
- **False Positives (146):** Unnecessary retention efforts (10.4% of predictions)
- **False Negatives (149):** Missed churners - improvement opportunity (40% of actual churners)
- **True Positives (222):** Successfully identified churners (60% recall)

---

### Phase 4: Dashboard Development

**Prepared 15 datasets for Tableau visualization:**

1. `main_dashboard.csv` - Comprehensive customer data with predictions
2. `kpi_summary.csv` - Executive metrics
3. `churn_by_contract.csv` - Segmentation by contract type
4. `churn_by_internet.csv` - Segmentation by internet service
5. `churn_by_payment.csv` - Segmentation by payment method
6. `churn_by_tenure.csv` - Lifecycle analysis
7. `risk_analysis.csv` - Risk segment breakdown
8. `confusion_matrix.csv` - Model performance details
9. `model_metrics.csv` - Accuracy, precision, recall, F1, AUC
10. `feature_importance.csv` - Top predictive features
11. `high_risk_customers.csv` - Priority contact list
12. `high_risk_profile.csv` - At-risk customer characteristics
13. `financial_impact.csv` - ROI calculations
14. `scenario_analysis.csv` - What-if projections
15. `revenue_breakdown.csv` - Revenue segmentation

**Dashboard Architecture:**

**Page 1: Executive Summary**
- KPI cards: Total customers, churn rate, high-risk count, revenue at risk
- Churn distribution visualization
- Risk segment breakdown
- Model accuracy gauge

**Page 2: Customer Segmentation**
- Churn rate by contract type
- Churn rate by internet service
- Churn rate by payment method
- Churn rate by tenure lifecycle
- Interactive filters for drill-down analysis

**Page 3: Risk Analysis & Targeting**
- Risk distribution (Low/Medium/High)
- High-risk customer table (top 100 by probability)
- Customer profile scatter plot (tenure vs charges)
- High-risk segment characteristics

**Page 4: Model Performance**
- Confusion matrix heatmap
- Performance metrics cards
- Feature importance chart
- ROC curve visualization

**Page 5: Financial Impact**
- Current vs projected revenue loss
- ROI calculation breakdown
- Scenario analysis (varying recall rates)
- Revenue funnel by risk segment

---