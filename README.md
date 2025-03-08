# Spatial Analysis of Economic Impact

## Overview

This project analyzes the economic impact of COVID-19 on Texas counties by examining various economic indicators, including GDP, unemployment rates, and poverty rates. The analysis utilizes spatial statistics to understand the spatial autocorrelation of these indicators over time.

## Data Sources

The data used in this analysis comes from the following sources:

- **GDP by County, Metro, and Other Areas**: 
  - [Bureau of Economic Analysis (BEA)](https://www.bea.gov/data/gdp/gdp-county-metro-and-other-areas)
  
- **Unemployment Report**: 
  - [U.S. Department of Agriculture Economic Research Service](https://data.ers.usda.gov/reports.aspx?ID=4038)
  
- **Poverty Report**: 
  - [U.S. Census Bureau](https://www.census.gov/data-tools/demo/saipe/#/?s_state=48&s_geography=county&s_measures=aa&x_tableYears=2023&s_county=&s_district=&map_yearSelector=2023)

## Requirements

To run the analysis, ensure you have the following Python packages installed:

- `geopandas`
- `numpy`
- `matplotlib`
- `libpysal`
- `esda`
- `pandas`
- `json`
- `scipy`

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Ensure that the economic data files (poverty, GDP, unemployment) are placed in the `data` directory. The shapefile for Texas counties should also be in the same directory.

2. **Run the Analysis**: Execute the `spatial_analysis.py` script to perform the spatial analysis. The script will:
   - Load the necessary data.
   - Calculate the economic index for each county.
   - Perform spatial autocorrelation analysis using Moran's I.
   - Generate visualizations, including choropleth maps and scatter plots.

3. **Output**: The results will be saved in the `processed_data` directory, including GeoJSON files for spatial analysis and visualizations.

## Functions

### `SpatialAnalyzer`

- **`__init__(self, data_dir="processed_data", output_dir="processed_data")`**: Initializes the spatial analyzer with specified data and output directories.
  
- **`create_weights_matrix(self, gdf)`**: Creates a spatial weights matrix from the GeoDataFrame.

- **`calculate_moran_i(self, gdf, weights, variable="Economic_Index")`**: Calculates global Moran's I for the specified variable.

- **`calculate_local_moran(self, gdf, weights, variable="Economic_Index")`**: Calculates local Moran's I and adds results to the GeoDataFrame.

- **`plot_spatial_analysis(self, gdf, moran_i, year, variable="Economic_Index")`**: Plots spatial analysis figures, including choropleth maps and Moran scatter plots.

- **`analyze_year(self, year)`**: Analyzes data for a specific year, calculating Moran's I and generating visualizations.

- **`calculate_recovery_metrics(self, pre_covid_gdf, covid_gdf, post_covid_gdf)`**: Calculates recovery metrics between pre-COVID, during COVID, and post-COVID data.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
