# Customer Churn Prediction & Analytics
Built an end-to-end ML pipeline achieving 83% accuracy and 87% ROC-AUC on Telco churn prediction, identifying 209 high-risk customers and projecting $204K annual net savings (365% ROI). Deployed interactive Tableau dashboard for stakeholder decision-making with customer segmentation and financial impact analysis. Pipeline with data processing, XGBoost modeling, and 3-year ROI projections.

### Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Size:** 7,043 customers × 21 features
- **Target Variable:** Churn (Yes/No)
- **Features:** Demographics, services, account information, charges
---

## Key Results

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| Model Accuracy | 83.2% | Industry-leading performance |
| ROC-AUC Score | 87.5% | Excellent discrimination ability |
| Precision | 70.4% | 70% of flagged customers actually churn |
| Recall | 60.1% | Catches 60% of at-risk customers |
| F1-Score | 64.8% | Balanced precision-recall performance |
| High-Risk Customers | 209 | Immediate action targets |
| Annual Revenue Saved | $260,844 | Gross savings from retention |
| Campaign Cost | $56,050 | Targeted outreach investment |
| Net Annual Savings | $204,794 | Bottom-line impact |
| ROI | 365% | First-year return on investment |
| Payback Period | 3.5 months | Time to break-even |

---

## Interactive Dashboard

**[View Live Tableau Dashboard](https://public.tableau.com/app/profile/aditi.manivannan/viz/CustomerChurnAnalysis_17699188633880/Dashboard1)**

The interactive dashboard provides:
- Executive KPI summary with key metrics
- Customer segmentation analysis by contract, service, and payment type
- Risk scoring and high-priority customer identification
- Model performance visualization
- Financial impact projections and ROI analysis
---

## Project Structure
```
churn-prediction-ml-analytics/
│
├── notebook/
│   ├── Project_implementation.ipynb
│   ├── readme.md
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── tableau/
│       └── (15 CSV files for dashboard)
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_metrics.txt
│
├── visualizations/
│   ├── dashboard_preview.png
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   └── roc_curves.png
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```


## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- Git

### Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-ml-analytics.git
cd churn-prediction-ml-analytics
```

**Step 2: Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Download dataset**

- Download the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle
- Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/raw/` directory

**Step 5: Run notebooks**
```bash
jupyter notebook
```
---

## Business Impact

### Current State Analysis

**Annual Churn Metrics:**
- Total customers: 7,043
- Annual churners: 1,869 (26.5%)
- Average customer monthly value: $65
- Annual churn revenue loss: $1,446,906

**Cost Breakdown:**
- Customer Acquisition Cost (CAC): $200 per customer (industry average)
- Annual acquisition spending to replace churners: $373,800
- Total annual impact of churn: $1,820,706

### ML Model Impact Projection

**Assumptions:**
- Model recall: 60.1% (identifies 60% of churners)
- Retention campaign success rate: 30% (industry benchmark)
- Retention campaign cost: $50 per customer contacted
- Implementation cost: $50,000 (one-time, amortized over 3 years)

**Financial Model:**

| Metric | Calculation | Value |
|--------|-------------|-------|
| Annual churners | 7,043 × 26.5% | 1,869 |
| Churners identified | 1,869 × 60.1% | 1,123 |
| Successfully retained | 1,123 × 30% | 337 |
| Monthly revenue saved | 337 × $65 | $21,905 |
| Annual revenue saved | $21,905 × 12 | $262,860 |
| Campaign cost | 1,123 × $50 | $56,150 |
| **Net annual savings** | Revenue - Cost | **$206,710** |
| **Year 1 ROI** | (Savings - Implementation) / Implementation × 100 | **313%** |
| **Ongoing ROI** | Net savings / Campaign cost × 100 | **368%** |
| **Payback period** | Implementation cost / Monthly savings | **2.3 months** |

### 3-Year Financial Projection

| Year | Implementation Cost | Campaign Cost | Revenue Saved | Net Benefit | Cumulative |
|------|---------------------|---------------|---------------|-------------|------------|
| 1 | $50,000 | $56,150 | $262,860 | $156,710 | $156,710 |
| 2 | $0 | $56,150 | $262,860 | $206,710 | $363,420 |
| 3 | $0 | $56,150 | $262,860 | $206,710 | $570,130 |

**Total 3-Year Value:** $570,130

## Technologies Used

### Data Analysis & Machine Learning

**Core Libraries:**
```
pandas==2.0.3          # Data manipulation and analysis
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # Machine learning algorithms and preprocessing
xgboost==1.7.6         # Gradient boosting framework
imbalanced-learn==0.11.0  # Handling class imbalance
```

**Visualization:**
```
matplotlib==3.7.2      # Static plotting
seaborn==0.12.2        # Statistical visualizations
```

**Development Environment:**
```
jupyter==1.0.0         # Interactive notebook interface
```
---

---

**Repository:** [github.com/yourusername/churn-prediction-ml-analytics](https://github.com/yourusername/churn-prediction-ml-analytics)  
**Dashboard:** [View Interactive Tableau Dashboard](https://public.tableau.com/app/profile/aditi.manivannan/viz/CustomerChurnAnalysis_17699188633880/Dashboard1)  