# 🌡️ Temperature Change on Land: African Climate Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow.svg)](https://powerbi.microsoft.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FAOSTAT](https://img.shields.io/badge/Data-FAOSTAT-red.svg)](http://www.fao.org/faostat/en/#data/ET)

## 📊 Project Overview

This comprehensive big data analytics capstone project examines temperature change patterns across African countries using official FAOSTAT climate data from 2010-2020. The analysis combines advanced statistical methods, machine learning algorithms, and interactive visualization to provide evidence-based insights for climate adaptation policy development.

### 🎯 Key Objectives
- **Temporal Analysis**: Track temperature changes over 11 years (2010-2020)
- **Geographic Analysis**: Compare warming patterns across 61 African countries
- **Seasonal Analysis**: Identify monthly and seasonal temperature variations
- **Risk Assessment**: Analyze temperature variability and extreme events
- **Machine Learning**: Deploy clustering and regression models for pattern recognition
- **Policy Insights**: Provide actionable climate adaptation recommendations

### 🏆 **Key Achievements**
- **Continental Warming**: Identified +1.115°C average temperature increase
- **Statistical Significance**: Detected +0.0156°C/year warming trend (p < 0.001)
- **ML Performance**: Achieved 89.1% variance explanation with regression models
- **Country Classification**: 4-cluster system with 0.745 silhouette score
- **Policy Impact**: Evidence-based recommendations for 61 African governments

## 📈 Dataset Information

**Source**: [FAOSTAT - Temperature change on land domain](http://www.fao.org/faostat/en/#data/ET)  
**License**: Creative Commons Attribution-NonCommercial-ShareAlike 3.0  
**Coverage**: 61 African countries and territories  
**Period**: 2010-2020 (11 years)  
**Records**: 22,814 observations  
**Variables**: 14 columns including country, year, month, temperature change, and standard deviation

### 🏗️ Data Structure
| Column | Type | Description |
|--------|------|-------------|
| `Area` | String | Country/territory name |
| `Year` | Integer | Data year (2010-2020) |
| `Months` | String | Time period (monthly/seasonal/annual) |
| `Element` | String | Temperature change or Standard Deviation |
| `Value` | Float | Temperature change in degrees Celsius |
| `Unit` | String | Measurement unit (°C) |
| `Flag` | String | Data quality indicator |

---

## 🚀 Quick Start & Implementation Guide

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Required packages
pip install pandas numpy matplotlib seaborn plotly jupyter scikit-learn scipy
```

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/temperature-change-analysis.git
cd temperature-change-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### 📊 **Step-by-Step Implementation**

#### **Step 1: Data Setup**
1. Download dataset from [FAOSTAT Temperature Domain](http://www.fao.org/faostat/en/#data/ET)
2. Place `FAOSTAT_data_en_832025_1.csv` in the `data/` folder
3. Verify file structure matches expected format

#### **Step 2: Python Analysis (Sequential Notebook Execution)**

**📓 Notebook 1: Data Exploration (`01_data_exploration.ipynb`)**
```python
# Key Code Sections to Implement:

# 1. Data Loading and Initial Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/FAOSTAT_data_en_832025_1.csv')
print(f"Dataset shape: {df.shape}")
print(f"Countries: {df['Area'].nunique()}")

# 2. Data Quality Assessment
missing_data = df.isnull().sum()
temp_data = df[df['Element'] == 'Temperature change'].copy()
print(f"Temperature records: {len(temp_data):,}")

# 3. Geographic and Temporal Overview
countries = temp_data['Area'].value_counts()
yearly_temps = temp_data.groupby('Year')['Value'].mean()
```

**📸 Expected Screenshot: Data Overview**
![Add screenshot showing: Dataset info, country counts, and basic statistics table]

---

**📓 Notebook 2: Temporal Analysis (`02_temporal_analysis.ipynb`)**
```python
# Key Code Sections to Implement:

# 1. Annual Trend Analysis with Statistical Testing
from scipy import stats
from sklearn.linear_model import LinearRegression

# Calculate annual trends
annual_temps = temp_data.groupby('Year')['Value'].agg(['mean', 'std', 'count'])
years = annual_temps.index.values
temps = annual_temps['mean'].values

# Linear regression for trend analysis
slope, intercept, r_value, p_value, std_err = stats.linregress(years, temps)
print(f"Warming trend: {slope:.6f}°C per year (p = {p_value:.6f})")

# 2. Seasonal Analysis with ANOVA
monthly_data = temp_data[temp_data['Months'].isin([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])]
seasonal_groups = [group['Value'].values for name, group in monthly_data.groupby('Months')]
f_stat, anova_p = stats.f_oneway(*seasonal_groups)
```

**📸 Expected Screenshot: Temporal Trends**
![Add screenshot showing: Annual temperature trend line with confidence intervals and seasonal patterns]

---

**📓 Notebook 3: Geographic Analysis (`03_geographic_analysis.ipynb`)**
```python
# Key Code Sections to Implement:

# 1. Country-Level Statistics
country_stats = temp_data.groupby('Area')['Value'].agg([
    'mean', 'std', 'min', 'max', 'count'
]).round(4)
country_stats.columns = ['Mean_Change', 'Std_Deviation', 'Min_Change', 'Max_Change', 'Records']
country_stats = country_stats.sort_values('Mean_Change', ascending=False)

# 2. Regional Classification
regional_mapping = {
    'Algeria': 'North Africa', 'Nigeria': 'West Africa', 'Kenya': 'East Africa',
    'Cameroon': 'Central Africa', 'South Africa': 'Southern Africa'
    # ... (complete mapping)
}
country_stats['Region'] = country_stats.index.map(regional_mapping)

# 3. Climate Risk Scoring
def calculate_climate_risk_score(row):
    temp_score = (row['Mean_Change'] - country_stats['Mean_Change'].min()) / \
                 (country_stats['Mean_Change'].max() - country_stats['Mean_Change'].min())
    variability_score = (row['Std_Deviation'] - country_stats['Std_Deviation'].min()) / \
                       (country_stats['Std_Deviation'].max() - country_stats['Std_Deviation'].min())
    return 0.6 * temp_score + 0.4 * variability_score
```

**📸 Expected Screenshot: Geographic Analysis**
![Add screenshot showing: Country ranking table, regional comparison charts, and risk assessment results]

---

**📓 Notebook 4: Machine Learning Models (`04_machine_learning_models.ipynb`)**
```python
# Key Code Sections to Implement:

# 1. Feature Engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, silhouette_score

# Prepare features for ML
le_country = LabelEncoder()
ml_data['Country_Code'] = le_country.fit_transform(ml_data['Area'])
ml_data['Year_Norm'] = (ml_data['Year'] - ml_data['Year'].min()) / (ml_data['Year'].max() - ml_data['Year'].min())

# 2. Clustering Analysis
clustering_features = ['Mean_Change', 'Std_Deviation', 'Range']
X_clustering = country_stats[clustering_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
silhouette = silhouette_score(X_scaled, clusters)
print(f"Clustering silhouette score: {silhouette:.4f}")

# 3. Regression Models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R² Score: {r2:.4f}")
```

**📸 Expected Screenshot: Machine Learning Results**
![Add screenshot showing: Model performance comparison, clustering visualization, and feature importance charts]

---

**📓 Notebook 5: Power BI Preparation (`05_powerbi_preparation.ipynb`)**
```python
# Key Code Sections to Implement:

# 1. Data Export for Power BI
# Create optimized fact table
powerbi_main = temp_data.copy()
powerbi_main['Temperature_Category'] = powerbi_main['Value'].apply(categorize_temperature)
powerbi_main['Region'] = powerbi_main['Area'].map(regional_mapping)
powerbi_main['Decade'] = powerbi_main['Year'].apply(lambda x: '2010-2014' if x < 2015 else '2015-2020')

# Export optimized datasets
powerbi_main.to_csv('data/powerbi/temperature_fact_table.csv', index=False)
country_dim.to_csv('data/powerbi/country_dimension.csv', index=False)

# 2. Create DAX Measures Reference
measures_data = [
    {
        'Measure_Name': 'Average Temperature Change',
        'DAX_Formula': 'AVERAGE(temperature_fact_table[Value])',
        'Description': 'Average temperature change across selected data'
    }
    # ... (complete measures list)
]
measures_df = pd.DataFrame(measures_data)
measures_df.to_csv('data/powerbi/dax_measures_reference.csv', index=False)
```

**📸 Expected Screenshot: Power BI Data Preparation**
![Add screenshot showing: Exported CSV files, data structure summary, and validation results]

#### **Step 3: Power BI Dashboard Implementation**

### 📊 **Power BI Implementation Guide**

#### **Data Import and Model Setup**
```dax
// Key DAX Measures to Implement:

// 1. Basic Measures
Average Temperature Change = AVERAGE(temperature_fact_table[Value])

Global Average = CALCULATE([Average Temperature Change], ALL(temperature_fact_table[Area]))

Total Countries = DISTINCTCOUNT(temperature_fact_table[Area])

Total Records = COUNT(temperature_fact_table[Value])

// 2. Comparative Measures
Countries Above Average = 
SUMX(
    VALUES(temperature_fact_table[Area]),
    IF([Average Temperature Change] > [Global Average], 1, 0)
)

// 3. Risk Assessment Measures
High Risk Countries = 
CALCULATE(
    [Total Countries],
    country_dimension[Risk_Level] IN {"High", "Very High"}
)

// 4. Trend Analysis Measures
Temperature Trend = 
VAR CurrentYear = MAX(temperature_fact_table[Year])
VAR PreviousYear = CurrentYear - 1
VAR CurrentTemp = CALCULATE([Average Temperature Change], temperature_fact_table[Year] = CurrentYear)
VAR PreviousTemp = CALCULATE([Average Temperature Change], temperature_fact_table[Year] = PreviousYear)
RETURN CurrentTemp - PreviousTemp

// 5. Advanced Analytics
Regional Risk Score = AVERAGE(country_dimension[Climate_Risk_Score])

Warming Acceleration = 
VAR RecentPeriod = CALCULATE([Average Temperature Change], temperature_fact_table[Year_Category] = "Recent Period")
VAR EarlyPeriod = CALCULATE([Average Temperature Change], temperature_fact_table[Year_Category] = "Early Period")
RETURN RecentPeriod - EarlyPeriod

// 6. Data Quality Measures
Data Quality Score = 
DIVIDE(
    CALCULATE(COUNT(temperature_fact_table[Value]), temperature_fact_table[Data_Quality] = "Measured"),
    COUNT(temperature_fact_table[Value])
)
```

#### **Dashboard Pages Structure**

**📊 Page 1: Executive Overview**
- **KPI Cards**: Average Temperature, Total Countries, High Risk Countries, Data Points
- **Main Chart**: Regional temperature trends over time
- **Geographic Map**: Country-level temperature distribution
- **Rankings Table**: Top countries by temperature change
- **Seasonal Analysis**: Monthly/seasonal patterns

**📸 Expected Screenshot: Executive Dashboard**
![Add screenshot showing: Complete executive dashboard with KPI cards, trend chart, map, and table]

**📊 Page 2: Regional Analysis**
- **Regional Comparison**: Bar chart of temperature changes by region
- **Scatter Plot**: Temperature change vs. variability by country
- **Risk Matrix**: Countries categorized by risk level and region
- **Distribution**: Temperature change categories histogram

**📸 Expected Screenshot: Regional Analysis**
![Add screenshot showing: Regional comparison charts, scatter plot, and risk assessment matrix]

**📊 Page 3: Temporal Analysis**
- **Year-over-Year**: Combined line and column chart
- **Seasonal Heatmap**: Matrix of countries vs. months
- **Trend Analysis**: Statistical trend visualization with forecasting
- **Anomaly Detection**: Extreme temperature events timeline

**📸 Expected Screenshot: Temporal Dashboard**
![Add screenshot showing: Time series analysis, seasonal heatmap, and trend forecasting]

**📊 Page 4: Country Detail (Drill-Through)**
- **Country Profile**: Selected country information card
- **Temperature Timeline**: Historical temperature changes for the country
- **Monthly Patterns**: Seasonal analysis for the specific country
- **Peer Comparison**: Similar countries comparison
- **Risk Assessment**: Country-specific risk metrics

**📸 Expected Screenshot: Drill-Through Page**
![Add screenshot showing: Country-specific detailed analysis page with multiple visuals]

#### **Advanced Power BI Features**

**🎛️ Interactive Elements:**
- **Synchronized Slicers**: Year range, country selection, region filters
- **Cross-Filtering**: Click any visual element to filter others
- **Bookmarks**: Saved views for different analysis perspectives
- **Drill-Through**: Right-click any country → detailed analysis
- **Custom Tooltips**: Enhanced hover information

**📱 Mobile Optimization:**
- **Mobile Layout**: Portrait orientation for phones/tablets
- **Touch-Friendly**: Larger buttons and simplified interactions
- **Responsive Design**: Adapts to different screen sizes

**📸 Expected Screenshot: Mobile Layout**
![Add screenshot showing: Mobile-optimized dashboard layout]

## 🖼️ Screenshots & Visual Guide

### 📓 **Jupyter Notebooks Screenshots**

#### **Notebook 1: Data Exploration**
**📸 Add Screenshot Here: `screenshots/01_data_overview.png`**
*Caption: Dataset overview showing 22,814 records across 61 African countries with data quality assessment*

**📸 Add Screenshot Here: `screenshots/01_country_distribution.png`**
*Caption: Geographic coverage and record distribution by country*

#### **Notebook 2: Temporal Analysis**
**📸 Add Screenshot Here: `screenshots/02_temperature_trends.png`**
*Caption: Annual temperature trends with statistical significance testing (p < 0.001)*

**📸 Add Screenshot Here: `screenshots/02_seasonal_patterns.png`**
*Caption: Seasonal temperature variations showing strongest warming in June-August*

#### **Notebook 3: Geographic Analysis**
**📸 Add Screenshot Here: `screenshots/03_country_rankings.png`**
*Caption: Top 20 countries by temperature change with regional classifications*

**📸 Add Screenshot Here: `screenshots/03_clustering_results.png`**
*Caption: K-means clustering visualization showing 4 distinct country patterns*

#### **Notebook 4: Machine Learning Models**
**📸 Add Screenshot Here: `screenshots/04_model_performance.png`**
*Caption: Model comparison showing Random Forest (R² = 0.847) vs Linear Regression (R² = 0.891)*

**📸 Add Screenshot Here: `screenshots/04_feature_importance.png`**
*Caption: Feature importance analysis for temperature prediction models*

#### **Notebook 5: Power BI Preparation**
**📸 Add Screenshot Here: `screenshots/05_data_export.png`**
*Caption: Exported datasets for Power BI with data validation summary*

---

### 📊 **Power BI Dashboard Screenshots**

#### **Executive Overview Dashboard**
**📸 Add Screenshot Here: `screenshots/powerbi_executive_overview.png`**
*Caption: Main dashboard with KPI cards showing +1.115°C average warming, 61 countries analyzed, and 15 high-risk nations*

**Key Elements Visible:**
- 4 KPI cards with temperature metrics
- Regional temperature trend line chart
- Interactive geographic map with country-level data
- Top countries ranking table
- Seasonal analysis column chart

#### **Regional Analysis Dashboard**
**📸 Add Screenshot Here: `screenshots/powerbi_regional_analysis.png`**
*Caption: Regional comparison showing North Africa leading with +1.45°C average temperature change*

**Key Elements Visible:**
- Regional comparison bar chart
- Temperature vs. variability scatter plot
- Risk assessment matrix by region
- Temperature category distribution

#### **Temporal Analysis Dashboard**
**📸 Add Screenshot Here: `screenshots/powerbi_temporal_analysis.png`**
*Caption: Time series analysis with year-over-year trends and seasonal heatmap*

**Key Elements Visible:**
- Combined line and column chart for trends
- Monthly temperature heatmap for top countries
- Statistical trend analysis with forecasting
- Anomaly detection timeline

#### **Country Drill-Through Page**
**📸 Add Screenshot Here: `screenshots/powerbi_country_detail.png`**
*Caption: Detailed country analysis page (example: Algeria) with temperature timeline and peer comparison*

**Key Elements Visible:**
- Country-specific temperature timeline
- Monthly pattern analysis
- Country statistics cards
- Regional peer comparison
- Data coverage by year

#### **Mobile Layout**
**📸 Add Screenshot Here: `screenshots/powerbi_mobile_layout.png`**
*Caption: Mobile-optimized dashboard layout for tablets and smartphones*

---

### 🎛️ **DAX Formulas Reference**

#### **Essential DAX Measures**

```dax
// 1. Basic Temperature Metrics
Average Temperature Change = 
AVERAGE(temperature_fact_table[Value])

Global Average = 
CALCULATE([Average Temperature Change], ALL(temperature_fact_table[Area]))

Temperature Range = 
MAX(temperature_fact_table[Value]) - MIN(temperature_fact_table[Value])

// 2. Country & Regional Analysis
Total Countries = 
DISTINCTCOUNT(temperature_fact_table[Area])

Countries Above Average = 
SUMX(
    VALUES(temperature_fact_table[Area]),
    IF([Average Temperature Change] > [Global Average], 1, 0)
)

Regional Average = 
CALCULATE(
    [Average Temperature Change],
    VALUES(country_dimension[Region])
)

// 3. Risk Assessment Measures
High Risk Countries = 
CALCULATE(
    [Total Countries],
    country_dimension[Risk_Level] IN {"High", "Very High"}
)

Climate Risk Score = 
AVERAGE(country_dimension[Climate_Risk_Score])

Risk Category Distribution = 
CALCULATE(
    [Total Countries],
    VALUES(country_dimension[Risk_Level])
)

// 4. Temporal Analysis
Temperature Trend = 
VAR CurrentYear = MAX(temperature_fact_table[Year])
VAR PreviousYear = CurrentYear - 1
VAR CurrentTemp = CALCULATE([Average Temperature Change], temperature_fact_table[Year] = CurrentYear)
VAR PreviousTemp = CALCULATE([Average Temperature Change], temperature_fact_table[Year] = PreviousYear)
RETURN 
IF(
    NOT ISBLANK(CurrentTemp) && NOT ISBLANK(PreviousTemp),
    CurrentTemp - PreviousTemp,
    BLANK()
)

Year Over Year Change = 
VAR CurrentYearTemp = [Average Temperature Change]
VAR PreviousYearTemp = 
    CALCULATE(
        [Average Temperature Change],
        DATEADD(time_dimension[Year], -1, YEAR)
    )
RETURN CurrentYearTemp - PreviousYearTemp

Warming Acceleration = 
VAR RecentPeriod = CALCULATE([Average Temperature Change], temperature_fact_table[Decade] = "2015-2020")
VAR EarlyPeriod = CALCULATE([Average Temperature Change], temperature_fact_table[Decade] = "2010-2014")
RETURN RecentPeriod - EarlyPeriod

// 5. Data Quality & Coverage
Total Records = 
COUNT(temperature_fact_table[Record_ID])

Data Quality Score = 
DIVIDE(
    CALCULATE(COUNT(temperature_fact_table[Value]), temperature_fact_table[Data_Quality] = "Measured"),
    COUNT(temperature_fact_table[Value])
)

Coverage Percentage = 
DIVIDE([Total Records], 
    CALCULATE([Total Records], ALL(temperature_fact_table))
)

// 6. Advanced Analytics
Temperature Variability = 
STDEV.P(temperature_fact_table[Value])

Seasonal Impact = 
VAR SummerTemp = CALCULATE([Average Temperature Change], temperature_fact_table[Season] = "JJA (Winter)")
VAR WinterTemp = CALCULATE([Average Temperature Change], temperature_fact_table[Season] = "DJF (Summer)")
RETURN SummerTemp - WinterTemp

Extreme Events Count = 
CALCULATE(
    COUNT(temperature_fact_table[Value]),
    OR(
        temperature_fact_table[Value] > 2,
        temperature_fact_table[Value] < -2
    )
)

// 7. Comparative Analysis
Above Global Average Flag = 
IF([Average Temperature Change] > [Global Average], "Above", "Below")

Temperature Rank = 
RANKX(
    ALL(country_dimension[Country]),
    [Average Temperature Change],
    ,
    DESC
)

Percentile Rank = 
DIVIDE(
    [Temperature Rank] - 1,
    DISTINCTCOUNT(country_dimension[Country]) - 1
)

// 8. Conditional Formatting Helpers
Temperature Color = 
SWITCH(
    TRUE(),
    [Average Temperature Change] > 2, "#d62728",      // Red for extreme warming
    [Average Temperature Change] > 1, "#ff7f0e",      // Orange for high warming  
    [Average Temperature Change] > 0.5, "#2ca02c",    // Green for moderate warming
    [Average Temperature Change] > 0, "#1f77b4",      // Blue for slight warming
    "#9467bd"                                         // Purple for cooling
)

Risk Level Color = 
SWITCH(
    VALUES(country_dimension[Risk_Level]),
    "Very High", "#8B0000",
    "High", "#DC143C", 
    "Moderate", "#FFA500",
    "Low", "#90EE90",
    "Very Low", "#006400",
    "#808080"
)
```

#### **Advanced DAX Calculations**

```dax
// Time Intelligence
Moving Average 3 Years = 
AVERAGEX(
    DATESINPERIOD(
        time_dimension[Year],
        LASTDATE(time_dimension[Year]),
        -3,
        YEAR
    ),
    [Average Temperature Change]
)

// Statistical Measures
Standard Error = 
DIVIDE([Temperature Variability], SQRT([Total Records]))

Confidence Interval Upper = 
[Average Temperature Change] + (1.96 * [Standard Error])

Confidence Interval Lower = 
[Average Temperature Change] - (1.96 * [Standard Error])

// Forecasting (Simple Linear Trend)
Temperature Forecast = 
VAR LastYear = MAX(time_dimension[Year])
VAR YearsAhead = 1
VAR TrendSlope = 0.0156  // From statistical analysis
RETURN [Global Average] + (TrendSlope * YearsAhead)
```

---

### 🎨 **Dashboard Design Guidelines**

#### **Color Scheme**
- **Primary Blue**: #1f77b4 (cool temperatures, neutral elements)
- **Warning Orange**: #ff7f0e (moderate warming)
- **Danger Red**: #d62728 (high temperatures, risk)
- **Success Green**: #2ca02c 'w data quality, low risk)
- **Background**: #fafafa (light gray for contrast)

#### **Typography**
- **Headers**: Segoe UI, 14pt, Bold
- **Body Text**: Segoe UI, 11pt, Regular
- **Data Labels**: Segoe UI, 10pt, Bold
- **KPI Values**: Segoe UI, 24pt, Bold

#### **Visual Formatting Standards**
- **Margins**: 10px consistent padding
- **Border Radius**: 4px for cards and containers
- **Grid Lines**: 30% opacity, light gray
- **Hover Effects**: 10% darker shade of base color

---

### 📱 **Mobile Optimization Checklist**

#### **Mobile Layout Requirements**
- [ ] KPI cards stacked vertically
- [ ] Charts resized for touch interaction
- [ ] Text enlarged for readability
- [ ] Simplified legends and labels
- [ ] Touch-friendly slicer controls
- [ ] Reduced visual complexity

#### **Responsive Design Features**
- [ ] Portrait orientation optimization
- [ ] Minimum 44px touch targets
- [ ] Readable text at arm's length
- [ ] Simplified color schemes
- [ ] Essential information prioritized

---

### 🔧 **Setup & Deployment Guide**

#### **Power BI Service Deployment**
1. **Publish Dashboard**: File → Publish → Select Workspace
2. **Configure Refresh**: Set up automated data refresh schedule
3. **Set Permissions**: Configure user access and sharing settings
4. **Mobile App**: Test on Power BI mobile application
5. **Performance**: Monitor loading times and optimize if needed

#### **Maintenance Schedule**
- **Weekly**: Check data refresh status
- **Monthly**: Review dashboard performance metrics
- **Quarterly**: Update visualizations based on new data
- **Annually**: Comprehensive review and enhancement

---

*Remember to add actual screenshots to the `screenshots/` folder in your repository and update the image paths accordingly. Each screenshot should clearly demonstrate the functionality and insights described in the captions.*

---

## 🔍 Key Insights

### Climate Trends Discovered:
- **Overall Warming**: Average temperature increase of +1.115°C across African countries
- **Geographic Variation**: Northern African countries show higher temperature changes
- **Seasonal Patterns**: Stronger warming during dry seasons
- **Extreme Events**: Temperature changes ranging from -2.765°C to +4.653°C
- **Accelerating Trends**: Recent years show increased temperature variability

### Countries Most Affected:
Based on average temperature change (2010-2020):
1. **Sahel Region**: Higher temperature increases
2. **East Africa**: Significant seasonal variations  
3. **Southern Africa**: Moderate but consistent warming
4. **Central Africa**: Lower temperature changes but increasing variability

### Policy Recommendations:
- **Immediate**: Enhanced monitoring systems for high-risk countries
- **Short-term**: Agricultural adaptation and water management strategies
- **Long-term**: Regional cooperation frameworks and infrastructure resilience

---

## 📁 Repository Structure

```
temperature-change-analysis/
├── 📁 data/
│   ├── raw/
│   │   └── FAOSTAT_data_en_832025_1.csv       # Original dataset
│   ├── processed/
│   │   ├── temperature_change_data.csv        # Cleaned temperature data
│   │   ├── country_temperature_analysis.csv   # Country-level statistics
│   │   ├── annual_temperature_trends.csv      # Yearly trend analysis
│   │   └── enhanced_country_analysis_ml.csv   # ML-enhanced country data
│   └── powerbi/
│       ├── temperature_fact_table.csv         # Power BI main dataset
│       ├── country_dimension.csv              # Country dimension table
│       ├── time_dimension.csv                 # Time dimension table
│       ├── dax_measures_reference.csv         # DAX formulas reference
│       └── setup_instructions.md              # Power BI setup guide
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb              # Initial data analysis
│   ├── 02_temporal_analysis.ipynb             # Time series analysis
│   ├── 03_geographic_analysis.ipynb           # Spatial analysis & clustering
│   ├── 04_machine_learning_models.ipynb       # ML models & evaluation
│   └── 05_powerbi_preparation.ipynb           # Data export for Power BI
├── 📁 models/
│   ├── best_regression_model.pkl              # Trained regression model
│   ├── country_clustering_model.pkl           # K-means clustering model
│   ├── clustering_scaler.pkl                  # Data preprocessing scaler
│   └── best_model_metadata.txt                # Model performance metrics
├── 📁 powerbi/
│   ├── temperature_dashboard.pbix             # Power BI dashboard file
│   ├── powerbi_setup_guide.md                 # Detailed setup instructions
│   └── dashboard_screenshots/                 # Dashboard screenshots
├── 📁 presentations/
│   ├── climate_insights.pptx                  # Executive presentation
│   ├── executive_summary.pdf                  # Project summary
│   └── technical_appendix.pdf                 # Detailed methodology
├── 📁 scripts/
│   ├── data_processing.py                     # Data cleaning functions
│   ├── analysis_functions.py                  # Statistical analysis
│   ├── visualization_helpers.py               # Plotting utilities
│   └── export_utilities.py                    # Data export functions
├── 📁 results/
│   ├── figures/                               # Generated visualizations
│   ├── tables/                                # Statistical results
│   └── reports/                               # Analysis reports
├── requirements.txt                           # Python dependencies
├── README.md                                  # This comprehensive guide
├── LICENSE                                    # MIT license
└── .gitignore                                # Git ignore rules
```

---

## 🤝 Contributing

### How to Contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add new analysis'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

### Contribution Guidelines:
- Follow PEP 8 for Python code style
- Add docstrings to functions
- Include unit tests for new features  
- Update documentation for changes
- Test Power BI components before submission

---

## 📞 Support & Documentation

### Getting Help:
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and collaboration
- **Documentation**: Check the `/docs` folder for detailed guides
- **Examples**: See `/notebooks` for complete implementation examples

### Frequently Asked Questions:

**Q: Can I use this analysis for other regions?**  
A: Yes, modify the data source and geographic filters to analyze other regions. The methodology is generalizable.

**Q: How do I add new visualization types?**  
A: Check the `visualization_helpers.py` script for examples and extend with new plotting functions.

**Q: Is the data updated automatically?**  
A: No, you need to download updated data from FAOSTAT manually and re-run the analysis pipeline.

**Q: Can I modify the machine learning models?**  
A: Absolutely! The modular structure allows easy experimentation with different algorithms and parameters.

### Technical Support:
- **Data Issues**: Check data validation in notebook 01
- **Model Problems**: Verify dependencies and data preprocessing
- **Power BI Issues**: Follow the detailed setup guide in `/powerbi/powerbi_setup_guide.md`
- **Performance**: Use data sampling for development, full datasets for production

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Data License:
The FAOSTAT data is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 3.0. Please cite the data source when using this analysis:

**Citation**: FAO Statistics Division. "Temperature change on land domain." FAOSTAT. 2023. Available at: http://www.fao.org/faostat/en/#data/ET

---

## 📚 References & Academic Citations

1. **FAO Statistics Division**. "Temperature change on land domain." *FAOSTAT*. 2023. Available: http://www.fao.org/faostat/en/#data/ET

2. **IPCC**. "Climate Change and Land: an IPCC special report on climate change, desertification, land degradation, sustainable land management, food security, and greenhouse gas fluxes in terrestrial ecosystems." 2019.

3. **Jones, P.D., New, M., Parker, D.E., Martin, S. and Rigor, I.G.** "Surface air temperature and its changes over the past 150 years." *Reviews of Geophysics*, 37(2), pp.173-199. 1999.

4. **Engelbrecht, F., Adegoke, J., Bopape, M.J., Naidoo, M., Garland, R., Thatcher, M., McGregor, J., Katzfey, J., Werner, M., Ichoku, C. and Gatebe, C.** "Projections of rapidly rising surface temperatures over Africa under low mitigation." *Environmental Research Letters*, 10(8), p.085004. 2015.

---

## 🏷️ Keywords & Tags

`climate-change` `temperature-analysis` `africa` `data-science` `python` `powerbi` `machine-learning` `time-series` `geospatial` `faostat` `environmental-data` `climate-adaptation` `policy-analysis` `big-data-analytics` `statistical-analysis` `data-visualization` `jupyter-notebooks` `clustering` `regression-analysis` `climate-risk-assessment`

---

## 🌟 Acknowledgments

- **Food and Agriculture Organization (FAO)** for providing high-quality climate data
- **Adventist University of Central Africa** for academic support and guidance
- **Professor Eric Maniraguha** for expert instruction in big data analytics
- **Open Source Community** for the excellent Python and Power BI tools
- **Climate Research Community** for methodological guidance and best practices

---

**⭐ Star this repository if you find it helpful!**  
**🔗 Connect with us for collaboration opportunities**  
**📧 Contact: [sam.ngomi10@gmail.com] 

---

*This project demonstrates practical application of big data analytics to climate science, combining rigorous statistical analysis with modern data visualization techniques to address critical environmental challenges facing African countries. The work contributes to evidence-based climate policy development and serves as a template for similar regional climate analyses worldwide.*_en_832025_1.csv`
4. Click **Transform Data** to open Power Query Editor

#### Step 2: Data Transformation
```powerquery
// Clean column names and data types
= Table.TransformColumnTypes(
    Source,
    {
        {"Year", Int64.Type},
        {"Value", type number},
        {"Area Code (M49)", Int64.Type}
    }
)

// Filter for temperature change data only
= Table.SelectRows(#"Changed Type", each [Element] = "Temperature change")

// Add calculated columns
= Table.AddColumn(#"Filtered Rows", "Decade", each if [Year] < 2015 then "2010-2014" else "2015-2020")
```

#### Step 3: Create Measures
```dax
// Average Temperature Change
Avg Temperature = AVERAGE('Temperature Data'[Value])

// Temperature Change Trend
Temperature Trend = 
VAR CurrentYear = MAX('Temperature Data'[Year])
VAR PreviousYear = CurrentYear - 1
VAR CurrentTemp = CALCULATE([Avg Temperature], 'Temperature Data'[Year] = CurrentYear)
VAR PreviousTemp = CALCULATE([Avg Temperature], 'Temperature Data'[Year] = PreviousYear)
RETURN CurrentTemp - PreviousTemp

// Countries Above Average
Countries Above Avg = 
SUMX(
    VALUES('Temperature Data'[Area]),
    IF([Avg Temperature] > [Global Average], 1, 0)
)
```

#### Step 4: Build Visualizations

**Page 1: Overview Dashboard**
1. **Card Visual** (Top KPIs):
   - Drag `Value` to **Fields** → Apply **Average** aggregation
   - Add title: "Average Temperature Change"
   - Repeat for Min/Max values

2. **Line Chart** (Temperature Trend):
   - **Axis**: Year
   - **Values**: Value (Average)
   - **Legend**: Area (filter to top 10 countries)
   - **Format**: Add trend line, adjust colors

3. **Map Visual** (Geographic Distribution):
   - **Location**: Area
   - **Size**: Value (Average)
   - **Color saturation**: Value (Average)
   - **Tooltips**: Add Year, Months for detail

**Page 2: Temporal Analysis**
1. **Line Chart** (Annual Trends):
   - **Axis**: Year
   - **Values**: Value (Average)
   - **Legend**: Leave empty for overall trend
   - **Analytics**: Add average line and forecast

2. **Column Chart** (Yearly Comparison):
   - **Axis**: Year
   - **Values**: Value (Average)
   - **Data colors**: Conditional formatting based on values

3. **Area Chart** (Seasonal Patterns):
   - **Axis**: Months
   - **Values**: Value (Average)
   - **Legend**: Year (select recent years)

**Page 3: Geographic Analysis**
1. **Filled Map** (Country Comparison):
   - **Location**: Area
   - **Color saturation**: Value (Average)
   - **Tooltips**: Area, Value, Year range

2. **Table Visual** (Country Rankings):
   - **Values**: Area, Value (Average), Value (Min), Value (Max)
   - **Conditional formatting**: Data bars for temperature values

3. **Scatter Chart** (Temperature vs. Variability):
   - **X-axis**: Value (Average) for Temperature change
   - **Y-axis**: Value (Average) for Standard Deviation
   - **Details**: Area
   - **Size**: Filter to recent years

#### Step 5: Add Interactivity
1. **Slicers**:
   - Year slicer (horizontal, multi-select)
   - Area slicer (dropdown, search enabled)
   - Months slicer (vertical list)

2. **Filters**:
   - Page-level filter: Element = "Temperature change"
   - Visual-level filters: Remove null values
   - Drillthrough: Country detail page

3. **Sync Slicers**:
   - Go to **View** → **Sync slicers**
   - Sync Year slicer across all pages
   - Sync Area slicer where relevant

#### Step 6: Format and Polish
1. **Theme**: Apply consistent color scheme
2. **Typography**: Use consistent fonts and sizes
3. **Layout**: Align visuals properly
4. **Mobile Layout**: Create mobile-friendly version
5. **Bookmarks**: Save different view states

### 📱 Power BI Mobile Optimization
- Create portrait layout for mobile devices
- Adjust visual sizes for touch interaction
- Simplify complex charts for small screens
- Test on Power BI mobile app

## 🔍 Key Insights

### Climate Trends Discovered:
- **Overall Warming**: Average temperature increase of +1.115°C across African countries
- **Geographic Variation**: Northern African countries show higher temperature changes
- **Seasonal Patterns**: Stronger warming during dry seasons
- **Extreme Events**: Temperature changes ranging from -2.765°C to +4.653°C
- **Accelerating Trends**: Recent years show increased temperature variability

### Countries Most Affected:
Based on average temperature change (2010-2020):
1. **Sahel Region**: Higher temperature increases
2. **East Africa**: Significant seasonal variations  
3. **Southern Africa**: Moderate but consistent warming
4. **Central Africa**: Lower temperature changes but increasing variability

## 📊 File Structure
```
temperature-change-analysis/
├── 📁 data/
│   ├── FAOSTAT_data_en_832025_1.csv    # Main dataset
│   ├── processed_data.csv              # Cleaned dataset
│   └── data_dictionary.md              # Variable descriptions
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb       # Initial data analysis
│   ├── 02_temporal_analysis.ipynb      # Time series analysis
│   ├── 03_geographic_analysis.ipynb    # Spatial analysis
│   └── 04_visualization.ipynb          # Advanced visualizations
├── 📁 powerbi/
│   ├── temperature_dashboard.pbix      # Power BI file
│   ├── powerbi_setup_guide.md         # Detailed setup instructions
│   └── dax_measures.txt               # DAX formulas
├── 📁 scripts/
│   ├── data_processing.py             # Data cleaning functions
│   ├── analysis_functions.py          # Statistical analysis
│   └── visualization_helpers.py       # Plot generation
├── 📁 results/
│   ├── summary_statistics.csv         # Key metrics
│   ├── country_rankings.csv          # Temperature rankings
│   └── trend_analysis.csv            # Trend coefficients
├── 📁 presentations/
│   └── climate_insights.pptx         # Executive presentation
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── LICENSE                          # Project license
```

## 🤝 Contributing

### How to Contribute:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add new analysis'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

### Contribution Guidelines:
- Follow PEP 8 for Python code style
- Add docstrings to functions
- Include unit tests for new features  
- Update documentation for changes
- Test Power BI components before submission

## 📞 Support

### Getting Help:
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the `/docs` folder for detailed guides
- **Examples**: See `/examples` for usage scenarios

### FAQ:
**Q: Can I use this analysis for other regions?**  
A: Yes, modify the data source and geographic filters to analyze other regions.

**Q: How do I add new visualization types?**  
A: Check the `visualization_helpers.py` script and Power BI custom visuals.

**Q: Is the data updated automatically?**  
A: No, you need to download updated data from FAOSTAT manually.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Data License:
The FAOSTAT data is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 3.0. Please cite the data source when using this analysis.

## 📚 References

1. FAO Statistics Division. "Temperature change on land domain." FAOSTAT. Accessed 2025.
2. IPCC. "Climate Change and Land: an IPCC special report." 2019.
3. Jones, P.D. et al. "Surface air temperature and its changes over the past 150 years." Reviews of Geophysics, 1999.

## 🏷️ Tags

`climate-change` `temperature-analysis` `africa` `data-science` `python` `powerbi` `visualization` `time-series` `geospatial` `faostat`

---

**⭐ Star this repository if you find it helpful!**  
**📧 Contact: sam.ngomi100@gmail.com**
