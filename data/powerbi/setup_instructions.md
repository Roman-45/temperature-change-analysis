
# Power BI Setup Instructions for Temperature Change Analysis

## Data Import Steps:

### Step 1: Import Main Fact Table
1. Open Power BI Desktop
2. Get Data → Text/CSV
3. Select: data/powerbi/temperature_fact_table.csv
4. Load data and verify 22,000+ records

### Step 2: Import Dimension Tables
1. Get Data → Text/CSV → country_dimension.csv
2. Get Data → Text/CSV → time_dimension.csv
3. Get Data → Text/CSV → summary_statistics.csv

### Step 3: Create Relationships
1. Go to Model View
2. Create relationship: temperature_fact_table[Area] → country_dimension[Country]
3. Set as Many-to-One, Single direction
4. Create relationship: temperature_fact_table[Year] → time_dimension[Year]
5. Set as Many-to-Many, Both directions

### Step 4: Import DAX Measures
1. Open data/powerbi/dax_measures_reference.csv in Excel
2. Copy each DAX formula from the reference
3. Create New Measure in Power BI
4. Paste formula and set appropriate formatting

### Step 5: Create Calculated Columns (if needed)
- Temperature Change Rank = RANKX(ALL(country_dimension), [Avg_Temperature_Change], , DESC)
- Above Global Average = IF([Avg_Temperature_Change] > [Global Average Temperature], "Yes", "No")

## Recommended Visualizations:

### Page 1: Executive Overview
- Card visuals for key metrics
- Line chart for temperature trends
- Map visual for geographic distribution
- Bar chart for top countries

### Page 2: Regional Analysis
- Stacked bar chart by region
- Scatter plot: temperature vs variability
- Table with country rankings
- Slicer for region filtering

### Page 3: Temporal Analysis
- Line chart with multiple countries
- Column chart for year-over-year changes
- Heatmap for seasonal patterns
- Trend analysis with forecasting

## Performance Optimization:
1. Enable Query Reduction in Options
2. Use DirectQuery only if data updates frequently
3. Create aggregation tables for large datasets
4. Optimize DAX formulas for performance

## Mobile Layout:
1. Switch to Mobile Layout view
2. Resize visuals for phone screens
3. Stack key metrics vertically
4. Simplify complex charts for touch interaction
