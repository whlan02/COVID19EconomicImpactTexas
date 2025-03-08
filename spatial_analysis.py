import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import libpysal
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
import json
import pandas as pd
import os

# 设置环境变量以使用 pyogrio 而不是 fiona
os.environ['USE_PYGEOS'] = '0'  # 禁用已弃用的 PyGEOS
os.environ['PYOGRIO_BACKEND'] = 'GDAL'  # 确保使用 GDAL 后端

class SpatialAnalyzer:
    def __init__(self, data_dir="processed_data", output_dir="processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_weights_matrix(self, gdf):
        """Create spatial weights matrix"""
        try:
            weights = Queen.from_dataframe(gdf)
            print(f"Successfully created spatial weights matrix, connections: {sum(weights.cardinalities.values())}")
            return weights
        except Exception as e:
            print(f"Error creating spatial weights matrix: {e}")
            return None

    def calculate_moran_i(self, gdf, weights, variable="Economic_Index"):
        """Calculate global Moran's I"""
        try:
            moran = Moran(gdf[variable], weights)
            print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")
            return moran
        except Exception as e:
            print(f"Error calculating Moran's I: {e}")
            return None

    def calculate_local_moran(self, gdf, weights, variable="Economic_Index"):
        """Calculate local Moran's I"""
        try:
            # Calculate local Moran's I
            local_moran = Moran_Local(gdf[variable], weights)
            
            # Add results to GeoDataFrame
            gdf = gdf.copy()
            gdf["local_moran"] = local_moran.Is
            gdf["local_moran_p"] = local_moran.p_sim
            gdf["local_moran_sig"] = gdf["local_moran_p"] < 0.05
            
            # Create cluster types
            gdf["local_moran_cluster"] = "Not Significant"
            
            # Calculate standardized values and spatial lag
            z = (gdf[variable] - gdf[variable].mean()) / gdf[variable].std()
            z_lag = libpysal.weights.lag_spatial(weights, z)
            
            # Classify clusters
            sig = gdf["local_moran_sig"]
            conditions = {
                "High-High": sig & (z > 0) & (z_lag > 0),
                "Low-Low": sig & (z < 0) & (z_lag < 0),
                "High-Low": sig & (z > 0) & (z_lag < 0),
                "Low-High": sig & (z < 0) & (z_lag > 0)
            }
            
            for cluster_type, condition in conditions.items():
                gdf.loc[condition, "local_moran_cluster"] = cluster_type
            
            print("Successfully calculated local Moran's I")
            return gdf
        except Exception as e:
            print(f"Error calculating local Moran's I: {e}")
            return gdf

    def plot_spatial_analysis(self, gdf, moran_i, year, variable="Economic_Index"):
        """Plot spatial analysis figures"""
        try:
            # 1. Economic index map
            self._plot_choropleth(gdf, year, variable)
            
            # 2. Moran scatter plot
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Standardize variable
            z = (gdf[variable] - gdf[variable].mean()) / gdf[variable].std()
            
            # Calculate spatial lag
            weights = self.create_weights_matrix(gdf)
            if weights is not None:
                lag = libpysal.weights.lag_spatial(weights, z)
                
                # Plot scatter plot
                ax.scatter(z, lag, alpha=0.6)
                
                # Add regression line
                slope, intercept = np.polyfit(z, lag, 1)
                x_range = np.linspace(z.min(), z.max(), 100)
                ax.plot(x_range, intercept + slope * x_range, 'r')
                
                # Add reference lines
                ax.axhline(y=0, color='k', linestyle=':')
                ax.axvline(x=0, color='k', linestyle=':')
                
                # Add labels
                for i, txt in enumerate(gdf["COUNTY"]):
                    ax.annotate(txt, (z.iloc[i], lag[i]), fontsize=8)
                
                ax.set_title(f"{year} Moran Scatter Plot (Moran's I = {moran_i.I:.4f}, p = {moran_i.p_sim:.4f})")
                ax.set_xlabel(f"Standardized {variable}")
                ax.set_ylabel(f"Spatial Lag {variable}")
                
                # Save the scatter plot
                plt.savefig(self.output_dir / f"moran_scatterplot_{year}.png",
                           dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved Moran scatter plot for {year}")
            
            # 3. Local Moran cluster map
            if "local_moran_cluster" in gdf.columns:
                self._plot_local_moran_clusters(gdf, year)
            
            print(f"Successfully saved spatial analysis figures for {year}")
        except Exception as e:
            print(f"Error plotting spatial analysis figures: {e}")

    def _plot_choropleth(self, gdf, year, variable):
        """Plot choropleth map"""
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf.plot(column=variable, ax=ax, legend=True,
                cmap="viridis", edgecolor="black", linewidth=0.5)
        ax.set_title(f"{year} Texas County {variable} Distribution")
        ax.axis("off")
        plt.savefig(self.output_dir / f"{variable}_map_{year}.png", 
                   dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_local_moran_clusters(self, gdf, year):
        """Plot local Moran cluster map"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define color mapping
        cluster_colors = {
            "High-High": "#FF0000",  # Red
            "Low-Low": "#0000FF",    # Blue
            "High-Low": "#FFA500",   # Orange
            "Low-High": "#00FFFF",   # Cyan
            "Not Significant": "#CCCCCC"  # Gray
        }
        
        # Plot map
        for category in cluster_colors.keys():
            subset = gdf[gdf["local_moran_cluster"] == category]
            if not subset.empty:
                subset.plot(ax=ax, color=cluster_colors[category],
                          edgecolor="black", linewidth=0.5, label=category)
        
        ax.set_title(f"{year} Local Moran Cluster Map")
        ax.axis("off")
        ax.legend(title="Cluster Type")
        
        plt.savefig(self.output_dir / f"local_moran_clusters_{year}.png",
                   dpi=300, bbox_inches="tight")
        plt.close(fig)

    def analyze_year(self, year):
        """Analyze data for a specific year"""
        # Load merged data
        try:
            gdf_path = self.data_dir / f"merged_data_{year}.geojson"
            gdf = gpd.read_file(gdf_path, engine='pyogrio')
            print(f"Loaded merged data for {year}")
        except Exception as e:
            print(f"Failed to load merged data for {year}: {e}")
            return None
            
        # Create spatial weights matrix
        weights = self.create_weights_matrix(gdf)
        if weights is None:
            return None
            
        # Calculate global Moran's I
        moran_i = self.calculate_moran_i(gdf, weights)
        if moran_i is None:
            return None
            
        # Save Moran's I results
        moran_results = {
            "I": moran_i.I,
            "p_value": moran_i.p_sim
        }
        with open(self.output_dir / f"moran_i_{year}.json", "w") as f:
            json.dump(moran_results, f)
        print(f"Saved Moran's I results for {year}")
            
        # Calculate local Moran's I
        gdf = self.calculate_local_moran(gdf, weights)
        
        # Save the updated GeoDataFrame with local Moran results
        gdf.to_file(self.output_dir / f"spatial_analysis_{year}.geojson", driver="GeoJSON", engine='pyogrio')
        print(f"Saved spatial analysis results for {year}")
        
        # Plot analysis figures
        self.plot_spatial_analysis(gdf, moran_i, year)
        
        return gdf
    
    def calculate_recovery_metrics(self, pre_covid_gdf, covid_gdf, post_covid_gdf):
        """Calculate recovery metrics between years"""
        recovery_metrics = {}
        
        if pre_covid_gdf is not None and covid_gdf is not None:
            # Calculate immediate impact (2019-2020)
            merged = pre_covid_gdf[["GEOID10", "Economic_Index"]].merge(
                covid_gdf[["GEOID10", "Economic_Index"]],
                on="GEOID10",
                suffixes=('_2019', '_2020')
            )
            merged['immediate_impact'] = merged['Economic_Index_2020'] - merged['Economic_Index_2019']
            recovery_metrics['immediate_impact'] = merged[['GEOID10', 'immediate_impact']]

        if covid_gdf is not None and post_covid_gdf is not None:
            # Calculate recovery (2020-2023)
            merged = covid_gdf[["GEOID10", "Economic_Index"]].merge(
                post_covid_gdf[["GEOID10", "Economic_Index"]],
                on="GEOID10",
                suffixes=('_2020', '_2023')
            )
            merged['recovery'] = merged['Economic_Index_2023'] - merged['Economic_Index_2020']
            recovery_metrics['recovery'] = merged[['GEOID10', 'recovery']]

        if pre_covid_gdf is not None and post_covid_gdf is not None:
            # Calculate overall change (2019-2023)
            merged = pre_covid_gdf[["GEOID10", "Economic_Index"]].merge(
                post_covid_gdf[["GEOID10", "Economic_Index"]],
                on="GEOID10",
                suffixes=('_2019', '_2023')
            )
            merged['overall_change'] = merged['Economic_Index_2023'] - merged['Economic_Index_2019']
            recovery_metrics['overall_change'] = merged[['GEOID10', 'overall_change']]

        return recovery_metrics

    def analyze_recovery_metrics(self, base_gdf, recovery_metrics, metric_type):
        """Analyze spatial autocorrelation for recovery metrics"""
        if metric_type not in recovery_metrics:
            return None

        # Merge recovery metrics with base geodataframe
        gdf = base_gdf.merge(recovery_metrics[metric_type], on="GEOID10", how="inner")
        
        # Create spatial weights matrix
        weights = self.create_weights_matrix(gdf)
        if weights is None:
            return None

        # Calculate global Moran's I for recovery metric
        moran_i = self.calculate_moran_i(gdf, weights, variable=metric_type)
        if moran_i is None:
            return None

        # Save Moran's I results for recovery metric
        moran_results = {
            "I": moran_i.I,
            "p_value": moran_i.p_sim
        }
        with open(self.output_dir / f"moran_i_{metric_type}.json", "w") as f:
            json.dump(moran_results, f)

        # Calculate local Moran's I for recovery metric
        gdf = self.calculate_local_moran(gdf, weights, variable=metric_type)
        
        # Save the updated GeoDataFrame with local Moran results
        gdf.to_file(self.output_dir / f"spatial_analysis_{metric_type}.geojson", driver="GeoJSON", engine='pyogrio')
        
        # Plot analysis figures
        self.plot_spatial_analysis(gdf, moran_i, metric_type, variable=metric_type)

        return gdf

    def process_all_years(self):
        """Process spatial analysis for all years and recovery metrics"""
        years = ["2019", "2020", "2021", "2022", "2023"]
        results = {}
        
        # First, load all necessary data
        gdfs = {}
        for year in years:
            print(f"\nProcessing spatial analysis for {year}...")
            try:
                gdf_path = self.data_dir / f"merged_data_{year}.geojson"
                if gdf_path.exists():
                    gdfs[year] = gpd.read_file(gdf_path, engine='pyogrio')
                    result = self.analyze_year(year)
                    if result is not None:
                        results[year] = result
            except Exception as e:
                print(f"Error loading data for {year}: {e}")
                gdfs[year] = None

        # Calculate and analyze recovery metrics
        if "2019" in gdfs and "2020" in gdfs and "2023" in gdfs:
            print("\nProcessing recovery metrics analysis...")
            recovery_metrics = self.calculate_recovery_metrics(
                gdfs.get("2019"),
                gdfs.get("2020"),
                gdfs.get("2023")
            )

            # Analyze each recovery metric
            for metric_type in ['immediate_impact', 'recovery', 'overall_change']:
                if metric_type in recovery_metrics:
                    print(f"\nAnalyzing {metric_type}...")
                    result = self.analyze_recovery_metrics(gdfs["2019"], recovery_metrics, metric_type)
                    if result is not None:
                        results[metric_type] = result

        # Save year comparison data
        self._save_year_comparison_data(years)
        
        return results
    
    def _save_year_comparison_data(self, years):
        """Save data for year comparison"""
        comparison_data = {}
        
        for year in years:
            try:
                # Load spatial analysis results
                gdf_path = self.output_dir / f"spatial_analysis_{year}.geojson"
                if not gdf_path.exists():
                    continue
                    
                gdf = gpd.read_file(gdf_path, engine='pyogrio')
                
                # Extract relevant data
                county_data = {}
                for _, row in gdf.iterrows():
                    county_data[row["County"]] = {
                        "Economic_Index": row["Economic_Index"],
                        "Poverty_Rate": row["Poverty_Rate"],
                        "GDP": row["GDP"],
                        "Unemployment_Rate": row["Unemployment_Rate"],
                        "local_moran_cluster": row["local_moran_cluster"] if "local_moran_cluster" in row else "Unknown"
                    }
                
                comparison_data[year] = county_data
                
            except Exception as e:
                print(f"Error processing comparison data for {year}: {e}")
        
        # Save comparison data
        with open(self.output_dir / "year_comparison_data.json", "w") as f:
            json.dump(comparison_data, f)
        print("Saved year comparison data")

def main():
    """Main function"""
    # Initialize analyzer
    spatial_analyzer = SpatialAnalyzer()
    
    # Process all years
    spatial_analyzer.process_all_years()

if __name__ == "__main__":
    main() 