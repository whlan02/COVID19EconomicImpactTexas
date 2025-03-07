import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

class DataProcessor:
    def __init__(self, data_dir="data", output_dir="processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shapefile_path = self.data_dir / "Tx_CntyBndry_Jurisdictional_TIGER.shp"
        
    def load_shapefile(self):
        """Load Texas county boundary shapefile data"""
        try:
            gdf = gpd.read_file(self.shapefile_path)
            print(f"Successfully loaded Shapefile, total {len(gdf)} counties")
            return gdf
        except Exception as e:
            print(f"Error loading Shapefile: {e}")
            return None

    def load_economic_data(self):
        """Load economic data (poverty rate, GDP, unemployment rate)"""
        data_files = {
            "poverty": "poverty rate_2019to2023.xlsx",
            "gdp": "Taxes_gdp_2019to2023.xlsx",
            "unemployment": "unemploymentReport_2019to2023.xlsx"
        }
        
        data_frames = {}
        
        for data_type, file_name in data_files.items():
            file_path = self.data_dir / file_name
            try:
                df = pd.read_excel(file_path)
                df = self._transform_to_long_format(df, data_type)
                data_frames[data_type] = df
                print(f"Successfully loaded {data_type} data, total {len(df)} records")
            except Exception as e:
                print(f"Error loading {data_type} data: {e}")
                data_frames[data_type] = None
        
        return data_frames

    def _transform_to_long_format(self, df, data_type):
        """Transform wide format data to long format"""
        value_vars_map = {
            "poverty": {
                "vars": ["Percent in Poverty_2019", "Percent in Poverty_2020", 
                        "Percent in Poverty_2021", "Percent in Poverty_2022", 
                        "Percent in Poverty_2023"],
                "value_name": "Poverty_Rate"
            },
            "gdp": {
                "vars": ["GDP_2019", "GDP_2020", "GDP_2021", "GDP_2022", "GDP_2023"],
                "value_name": "GDP"
            },
            "unemployment": {
                "vars": ["Unemployment Rate _2019", "Unemployment Rate _2020",
                        "Unemployment Rate _2021", "Unemployment Rate _2022",
                        "Unemployment Rate _2023"],
                "value_name": "Unemployment_Rate"
            }
        }
        
        config = value_vars_map.get(data_type)
        if config:
            df = pd.melt(df,
                        id_vars=["GEOID10", "Name"],
                        value_vars=config["vars"],
                        var_name="Year",
                        value_name=config["value_name"])
            df["Year"] = df["Year"].str.extract("(\d{4})").astype(int)
            df = df.rename(columns={"Name": "County"})
        
        return df

    def clean_data(self, data_frames):
        """Clean and standardize economic data"""
        cleaned_data = {}
        
        for data_type, df in data_frames.items():
            if df is None:
                continue
            
            df = df.copy()
            # Handle missing values
            value_col = {"poverty": "Poverty_Rate", 
                        "gdp": "GDP", 
                        "unemployment": "Unemployment_Rate"}[data_type]
            df[value_col] = df[value_col].fillna(df[value_col].mean())
            
            # Ensure correct data types
            df["County"] = df["County"].astype(str)
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
            
            cleaned_data[data_type] = df
        
        return cleaned_data

    def integrate_data(self, cleaned_data, year):
        """Integrate economic data for a specific year"""
        if not all(df is not None for df in cleaned_data.values()):
            print("Cannot integrate data because some datasets are missing")
            return None
        
        dfs = []
        for data_type, df in cleaned_data.items():
            year_data = df[df["Year"] == int(year)].copy()
            if not year_data.empty:
                dfs.append(year_data)
        
        if not dfs:
            print(f"No data available for the year {year}")
            return None
        
        # Merge all datasets
        merged_data = dfs[0]
        for df in dfs[1:]:
            merged_data = merged_data.merge(df[["GEOID10", "Year"] + 
                                             [col for col in df.columns if col not in ["County", "Year", "GEOID10"]]],
                                          on=["GEOID10", "Year"],
                                          how="inner")
        
        print(f"Successfully integrated data for the year {year}, total {len(merged_data)} counties")
        return merged_data

    def calculate_economic_index(self, integrated_data):
        """Calculate economic status index"""
        if integrated_data is None or integrated_data.empty:
            print("Cannot calculate economic index because the integrated data is empty")
            return None
        
        df = integrated_data.copy()
        
        # Check for Starr County before processing
        starr_county = df[df["County"].str.contains("Starr", case=False, na=False)]
        if not starr_county.empty:
            print(f"Starr County data before processing: {starr_county.to_dict('records')}")
        else:
            print("Starr County not found in input data")
        
        # Use RobustScaler which is based on percentiles and is less influenced by outliers
        # It uses the interquartile range (IQR) which makes it very robust to outliers
        scaler = RobustScaler()
        numeric_columns = ["Poverty_Rate", "GDP", "Unemployment_Rate"]
        
        # Check for NaN values before scaling
        for col in numeric_columns:
            if df[col].isna().any():
                print(f"Warning: NaN values found in {col} column. Filling with mean.")
                df[col] = df[col].fillna(df[col].mean())
        
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Calculate economic index (inverting poverty rate and unemployment rate)
        weights = {"Poverty_Rate": -1, "GDP": 1, "Unemployment_Rate": -1}
        
        df["Economic_Index"] = 0
        for col, weight in weights.items():
            if weight < 0:
                # For negative indicators (where lower is better), invert the value
                df["Economic_Index"] += -df[col] * abs(weight)
            else:
                df["Economic_Index"] += df[col] * weight
        
        # Check Starr County after initial calculation
        starr_county = df[df["County"].str.contains("Starr", case=False, na=False)]
        if not starr_county.empty:
            print(f"Starr County Economic_Index before normalization: {starr_county['Economic_Index'].values[0]}")
        
        # Normalize economic index to a 0-100 scale for better interpretability
        # First shift to ensure all values are positive
        min_val = df["Economic_Index"].min()
        if min_val < 0:
            df["Economic_Index"] = df["Economic_Index"] - min_val
        
        # Then scale to 0-100 range
        max_val = df["Economic_Index"].max()
        if max_val > 0:  # Avoid division by zero
            df["Economic_Index"] = (df["Economic_Index"] / max_val) * 100
        
        # Check Starr County after normalization
        starr_county = df[df["County"].str.contains("Starr", case=False, na=False)]
        if not starr_county.empty:
            print(f"Starr County Economic_Index after normalization: {starr_county['Economic_Index'].values[0]}")
            print(f"Starr County full data: {starr_county.to_dict('records')}")
        
        print(f"Successfully calculated economic index, range: {df['Economic_Index'].min():.4f} - {df['Economic_Index'].max():.4f}")
        return df

    def process_all_years(self, years=None):
        """Process data for all years"""
        if years is None:
            years = ["2019", "2020", "2021", "2022", "2023"]
            
        # Load data
        economic_data = self.load_economic_data()
        if not all(df is not None for df in economic_data.values()):
            print("Cannot proceed because some necessary data is missing")
            return None
        
        # Clean data
        cleaned_data = self.clean_data(economic_data)
        results = {}
        
        for year in years:
            # Integrate data
            integrated_data = self.integrate_data(cleaned_data, year)
            
            if integrated_data is not None:
                # Calculate economic index
                economic_index_df = self.calculate_economic_index(integrated_data)
                
                if economic_index_df is not None:
                    results[year] = economic_index_df
                    # Save processed data
                    output_file = self.output_dir / f"economic_index_{year}.csv"
                    economic_index_df.to_csv(output_file, index=False)
                    print(f"Saved {year} economic index data to {output_file}")
        
        return results
    
    def process_and_save_all_data(self):
        """Process all data and save for quick loading"""
        # Process all years
        results = self.process_all_years()
        if not results:
            print("Failed to process economic data")
            return False
            
        # Load and save shapefile
        gdf = self.load_shapefile()
        if gdf is None:
            print("Failed to load shapefile")
            return False
            
        # Save shapefile in a more accessible format
        shapefile_output = self.output_dir / "texas_counties.geojson"
        gdf.to_file(shapefile_output, driver="GeoJSON")
        print(f"Saved shapefile to {shapefile_output}")
        
        # Create merged geodataframes for each year
        for year, df in results.items():
            # Ensure the data type of GEOID10 column is consistent
            gdf["GEOID10"] = gdf["GEOID10"].astype(str)
            df["GEOID10"] = df["GEOID10"].astype(str)
                
            # Merge geographic data and economic data
            merged_gdf = gdf.merge(df, 
                                 left_on="GEOID10",
                                 right_on="GEOID10",
                                 how="inner")
            
            # Save merged geodataframe
            merged_output = self.output_dir / f"merged_data_{year}.geojson"
            merged_gdf.to_file(merged_output, driver="GeoJSON")
            print(f"Saved merged data for {year} to {merged_output}")
        
        print("All data processed and saved successfully")
        return True

def main():
    """Main function"""
    processor = DataProcessor()
    processor.process_and_save_all_data()

if __name__ == "__main__":
    main() 