import streamlit as st
import pandas as pd
from pathlib import Path
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import time
import json
import os


os.environ['USE_PYGEOS'] = '0'  
os.environ['PYOGRIO_BACKEND'] = 'GDAL'  

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Economic Impact on Texas Counties",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("COVID-19 Economic Impact on Texas Counties")
st.markdown("### Analyzing economic changes before, during, and after the pandemic")

class App:
    def __init__(self, data_dir="processed_data"):
        self.data_dir = Path(data_dir)
        # Define key years for COVID analysis
        self.pre_covid_year = "2019"  # Before COVID
        self.covid_year = "2020"      # During COVID
        self.post_covid_year = "2023" # After COVID recovery period
        # Cache the data at initialization
        self._cache_data()
        
    def _cache_data(self):
        """Cache all necessary data at initialization"""
        self.cached_data = {}
        years = [self.pre_covid_year, self.covid_year, self.post_covid_year]
        
        for year in years:
            try:
                file_path = self.data_dir / f"spatial_analysis_{year}.geojson"
                if file_path.exists():
                    # Use pyogrio engine to read the file
                    gdf = gpd.read_file(str(file_path), 
                                    columns=["GEOID10", "COUNTY", "Economic_Index", 
                                            "Poverty_Rate", "GDP", "Unemployment_Rate", 
                                            "geometry", "local_moran_cluster"],
                                    engine='pyogrio')
                    self.cached_data[year] = gdf
            except Exception as e:
                print(f"Error caching data for {year}: {e}")
                st.error(f"Error loading data for {year}: {e}")
    
    @st.cache_data(ttl=3600)
    def load_data(_self, year):
        """Load pre-calculated data for a specific year with optimized loading"""
        return _self.cached_data.get(year)
                

    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_moran_results(_self, year):
        """Load Moran's I results for a specific year"""
        try:
            file_path = _self.data_dir / f"moran_i_{year}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Error loading Moran's I results for {year}: {e}")
            return None
    
    def create_covid_impact_metrics(self, covid_data):
        """Calculate COVID impact metrics"""
        metrics = {}
        
        # Check which years are available
        available_years = list(covid_data.keys())
        
        # Calculate immediate COVID impact (2019 to 2020)
        if self.pre_covid_year in available_years and self.covid_year in available_years:
            pre_covid = covid_data[self.pre_covid_year]
            during_covid = covid_data[self.covid_year]
            
            # Merge data
            impact_data = pre_covid[["GEOID10", "COUNTY", "Economic_Index", "geometry"]].merge(
                during_covid[["GEOID10", "Economic_Index"]],
                on="GEOID10",
                how="inner",
                suffixes=(f"_{self.pre_covid_year}", f"_{self.covid_year}")
            )
            
            # Calculate impact
            impact_data["COVID_Impact"] = impact_data[f"Economic_Index_{self.covid_year}"] - impact_data[f"Economic_Index_{self.pre_covid_year}"]
            
            # Handle division by zero or null values
            impact_data["COVID_Impact_Percent"] = impact_data.apply(
                lambda row: (row["COVID_Impact"] / row[f"Economic_Index_{self.pre_covid_year}"]) * 100 
                if pd.notnull(row[f"Economic_Index_{self.pre_covid_year}"]) and row[f"Economic_Index_{self.pre_covid_year}"] != 0 
                else None, 
                axis=1
            )
            
            metrics["immediate_impact"] = impact_data
        
        # Calculate recovery (2020 to 2023)
        if self.covid_year in available_years and self.post_covid_year in available_years:
            during_covid = covid_data[self.covid_year]
            post_covid = covid_data[self.post_covid_year]
            
            # Merge data
            recovery_data = during_covid[["GEOID10", "COUNTY", "Economic_Index", "geometry"]].merge(
                post_covid[["GEOID10", "Economic_Index"]],
                on="GEOID10",
                how="inner",
                suffixes=(f"_{self.covid_year}", f"_{self.post_covid_year}")
            )
            
            # Calculate recovery
            recovery_data["Recovery"] = recovery_data[f"Economic_Index_{self.post_covid_year}"] - recovery_data[f"Economic_Index_{self.covid_year}"]
            recovery_data["Recovery_Percent"] = (recovery_data["Recovery"] / recovery_data[f"Economic_Index_{self.covid_year}"]) * 100
            
            metrics["recovery"] = recovery_data
        
        # Calculate overall change (2019 to 2023)
        if self.pre_covid_year in available_years and self.post_covid_year in available_years:
            pre_covid = covid_data[self.pre_covid_year]
            post_covid = covid_data[self.post_covid_year]
            
            # Merge data
            overall_data = pre_covid[["GEOID10", "COUNTY", "Economic_Index", "geometry"]].merge(
                post_covid[["GEOID10", "Economic_Index"]],
                on="GEOID10",
                how="inner",
                suffixes=(f"_{self.pre_covid_year}", f"_{self.post_covid_year}")
            )
            
            # Calculate overall change
            overall_data["Overall_Change"] = overall_data[f"Economic_Index_{self.post_covid_year}"] - overall_data[f"Economic_Index_{self.pre_covid_year}"]
            overall_data["Overall_Change_Percent"] = (overall_data["Overall_Change"] / overall_data[f"Economic_Index_{self.pre_covid_year}"]) * 100
            
            metrics["overall_change"] = overall_data
        
        return metrics
    
    def show_covid_impact_map(self, impact_data, metric_column, title, color_scale="RdBu"):
        """Display COVID impact map with optimized rendering"""
        # Convert to WGS84 for folium if needed
        gdf_wgs84 = impact_data.copy()
        if gdf_wgs84.crs != "EPSG:4326":
            gdf_wgs84 = gdf_wgs84.to_crs(epsg=4326)
        
        # Simplify geometries for faster rendering
        gdf_wgs84['geometry'] = gdf_wgs84['geometry'].simplify(tolerance=0.01)
        
        # Create a simplified folium map
        m = folium.Map(location=[31.0, -100.0], zoom_start=6, 
                      tiles='CartoDB positron',
                      prefer_canvas=True)  # Use canvas renderer for better performance
        
        # Create a choropleth layer with simplified styling
        folium.Choropleth(
            geo_data=gdf_wgs84.__geo_interface__,
            name="choropleth",
            data=gdf_wgs84,
            columns=["GEOID10", metric_column],
            key_on="feature.properties.GEOID10",
            fill_color=color_scale,
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=title,
            smooth_factor=2  # Smooth lines for better performance
        ).add_to(m)
        
        # Simplified tooltip with fewer fields
        tooltip = folium.features.GeoJsonTooltip(
            fields=["COUNTY", metric_column],
            aliases=["County: ", f"{title}: "],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
        
        # Add GeoJson layer with simplified styling
        folium.GeoJson(
            gdf_wgs84,
            name="Counties",
            tooltip=tooltip,
            style_function=lambda x: {'fillOpacity': 0, 'weight': 0.5}
        ).add_to(m)
        
        # Display the map
        folium_static(m)
    
    def show_moran_cluster_map(self, gdf, year):
        """Display interactive local Moran cluster map using folium"""
        # Convert to WGS84 for folium if needed
        gdf_wgs84 = gdf.copy()
        if gdf_wgs84.crs != "EPSG:4326":
            gdf_wgs84 = gdf_wgs84.to_crs(epsg=4326)
        
        # Simplify geometries for faster rendering
        gdf_wgs84['geometry'] = gdf_wgs84['geometry'].simplify(tolerance=0.01)
        
        # Create a folium map
        m = folium.Map(location=[31.0, -100.0], zoom_start=6, 
                      tiles='CartoDB positron',
                      prefer_canvas=True)
        
        # Define color mapping for clusters
        cluster_colors = {
            "High-High": "#FF0000",  # Red
            "Low-Low": "#0000FF",    # Blue
            "High-Low": "#FFA500",   # Orange
            "Low-High": "#00FFFF",   # Cyan
            "Not Significant": "#CCCCCC"  # Gray
        }
        
        # Create a function to style features based on cluster type
        def style_function(feature):
            cluster_type = feature['properties']['local_moran_cluster']
            return {
                'fillColor': cluster_colors.get(cluster_type, "#CCCCCC"),
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.7
            }
        
        # Add GeoJson layer with cluster styling
        tooltip = folium.features.GeoJsonTooltip(
            fields=["COUNTY", "Economic_Index", "local_moran_cluster"],
            aliases=["County: ", "Economic Index: ", "Cluster Type: "],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
        
        folium.GeoJson(
            gdf_wgs84,
            name="Local Moran Clusters",
            tooltip=tooltip,
            style_function=style_function
        ).add_to(m)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
        padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><b>Cluster Types</b></p>
        <p><i style="background: #FF0000; width: 15px; height: 15px; display: inline-block;"></i> High-High</p>
        <p><i style="background: #0000FF; width: 15px; height: 15px; display: inline-block;"></i> Low-Low</p>
        <p><i style="background: #FFA500; width: 15px; height: 15px; display: inline-block;"></i> High-Low</p>
        <p><i style="background: #00FFFF; width: 15px; height: 15px; display: inline-block;"></i> Low-High</p>
        <p><i style="background: #CCCCCC; width: 15px; height: 15px; display: inline-block;"></i> Not Significant</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Display the map
        folium_static(m)
    
    def show_top_bottom_counties(self, data, metric_column, title, n=10):
        """Show top and bottom counties for a given metric"""
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Filter out rows with null values in the metric column
        valid_data = data.dropna(subset=[metric_column])
        
        # Check if we have any valid data
        if valid_data.empty:
            st.warning(f"No valid data available for {title}")
            return
            
        # Count and report null values
        null_count = len(data) - len(valid_data)
        if null_count > 0:
            st.warning(f"{null_count} counties have null values for {title}")
            
            # Show counties with null values
            null_counties = data[data[metric_column].isna()]["COUNTY"].tolist()
            if null_counties:
                with st.expander("Counties with null values"):
                    for county in null_counties:
                        st.write(f"- {county}")
        
        # Sort data
        sorted_data = valid_data.sort_values(metric_column, ascending=False)
        
        # Display top counties
        with col1:
            st.subheader(f"Top {n} Counties ({title})")
            top_counties = sorted_data.head(n)[["COUNTY", metric_column]]
            st.dataframe(top_counties, use_container_width=True)
        
        # Display bottom counties
        with col2:
            st.subheader(f"Bottom {n} Counties ({title})")
            bottom_counties = sorted_data.tail(n)[["COUNTY", metric_column]].sort_values(metric_column)
            st.dataframe(bottom_counties, use_container_width=True)
    
    def show_moran_results(self):
        """Display Moran's I results and plots for selected years and recovery metrics"""
        st.subheader("Spatial Autocorrelation Analysis")
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Annual Economic Index", "Recovery Metrics"])
        
        # Tab 1: Annual Economic Index
        with tab1:
            year_tab1, year_tab2, year_tab3 = st.tabs(["Pre-COVID (2019)", "During COVID (2020)", "Post-COVID (2023)"])
            
            with year_tab1:
                self.display_moran_results_for_year("2019")
            with year_tab2:
                self.display_moran_results_for_year("2020")
            with year_tab3:
                self.display_moran_results_for_year("2023")
        
        # Tab 2: Recovery Metrics
        with tab2:
            metric_tab1, metric_tab2, metric_tab3 = st.tabs([
                "Immediate Impact (2019-2020)",
                "Recovery (2020-2023)",
                "Overall Change (2019-2023)"
            ])
            
            with metric_tab1:
                self.display_moran_results_for_metric("immediate_impact")
            with metric_tab2:
                self.display_moran_results_for_metric("recovery")
            with metric_tab3:
                self.display_moran_results_for_metric("overall_change")

    def display_moran_results_for_year(self, year):
        """Helper function to display Moran's I results for a specific year"""
        # Load Moran's I results
        moran_results = self.load_moran_results(year)
        
        if moran_results:
            # Display interactive local Moran cluster map first
            gdf = self.load_data(year)
            if gdf is not None and "local_moran_cluster" in gdf.columns:
                st.subheader(f"Local Moran Cluster Map ({year})")
                self.show_moran_cluster_map(gdf, year)
            else:
                st.warning(f"Local Moran cluster data not available for {year}")
            
            # Display Moran's I statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Moran's I", f"{moran_results['I']:.4f}")
            with col2:
                st.metric("P-value", f"{moran_results['p_value']:.4f}")
            
            # Add interpretation
            if moran_results['p_value'] < 0.05:
                if moran_results['I'] > 0:
                    st.info("There is significant positive spatial autocorrelation, indicating that counties with similar economic conditions tend to cluster together.")
                elif moran_results['I'] < 0:
                    st.info("There is significant negative spatial autocorrelation, indicating that counties with different economic conditions tend to be neighbors.")
            else:
                st.info("No significant spatial autocorrelation detected, suggesting that economic conditions are randomly distributed across counties.")
            
            # Display Moran scatter plot
            scatter_path = self.data_dir / f"moran_scatterplot_{year}.png"
            if scatter_path.exists():
                st.image(str(scatter_path), caption=f"Moran Scatter Plot ({year})")
        else:
            st.warning(f"No Moran's I results available for {year}")
    
    def display_moran_results_for_metric(self, metric_type):
        """Helper function to display Moran's I results for a recovery metric"""
        # Load Moran's I results for the metric
        try:
            with open(self.data_dir / f"moran_i_{metric_type}.json", 'r') as f:
                moran_results = json.load(f)
        except Exception as e:
            st.warning(f"No Moran's I results available for {metric_type}")
            return

        # Load spatial analysis results
        try:
            gdf = gpd.read_file(self.data_dir / f"spatial_analysis_{metric_type}.geojson", engine='pyogrio')
        except Exception as e:
            st.warning(f"No spatial analysis results available for {metric_type}")
            return

        # Display interactive local Moran cluster map
        if "local_moran_cluster" in gdf.columns:
            st.subheader(f"Local Moran Cluster Map ({metric_type})")
            self.show_moran_cluster_map(gdf, metric_type)

        # Display Moran's I statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Moran's I", f"{moran_results['I']:.4f}")
        with col2:
            st.metric("P-value", f"{moran_results['p_value']:.4f}")

        # Add interpretation
        if moran_results['p_value'] < 0.05:
            if moran_results['I'] > 0:
                st.info(f"There is significant positive spatial autocorrelation in {metric_type}, indicating that counties with similar recovery patterns tend to cluster together.")
            elif moran_results['I'] < 0:
                st.info(f"There is significant negative spatial autocorrelation in {metric_type}, indicating that counties with different recovery patterns tend to be neighbors.")
        else:
            st.info(f"No significant spatial autocorrelation detected in {metric_type}, suggesting that recovery patterns are randomly distributed across counties.")

        # Display Moran scatter plot
        scatter_path = self.data_dir / f"moran_scatterplot_{metric_type}.png"
        if scatter_path.exists():
            st.image(str(scatter_path), caption=f"Moran Scatter Plot ({metric_type})")

        # Show top and bottom counties for the recovery metric
        if metric_type in gdf.columns:
            st.subheader(f"Top and Bottom Counties ({metric_type})")
            self.show_top_bottom_counties(gdf, metric_type, f"{metric_type} Value", n=5)

    def show_covid_impact_dashboard(self, covid_data, covid_metrics):
        """Display COVID impact dashboard"""
        st.header("COVID-19 Economic Impact Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Immediate Impact (2019-2020)", 
                                  "Recovery (2020-2023)", 
                                  "Overall Change (2019-2023)"])
        
        # Tab 1: Immediate Impact
        with tab1:
            if "immediate_impact" in covid_metrics:
                impact_data = covid_metrics["immediate_impact"]
                
                st.subheader("Economic Impact During COVID-19 (2019 to 2020)")
                
                # Display key statistics
                avg_impact = impact_data["COVID_Impact_Percent"].mean()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Impact", f"{avg_impact:.2f}%")
                with col2:
                    st.metric("Most Negative Impact", 
                             f"{impact_data['COVID_Impact_Percent'].min():.2f}%",
                             f"{impact_data.loc[impact_data['COVID_Impact_Percent'].idxmin(), 'COUNTY']}")
                with col3:
                    st.metric("Most Positive Impact", 
                             f"{impact_data['COVID_Impact_Percent'].max():.2f}%",
                             f"{impact_data.loc[impact_data['COVID_Impact_Percent'].idxmax(), 'COUNTY']}")
                
                # Display map
                self.show_covid_impact_map(impact_data, "COVID_Impact_Percent", 
                                         "COVID-19 Impact (%)")
                
                # Show top and bottom counties
                self.show_top_bottom_counties(impact_data, "COVID_Impact_Percent", 
                                            "COVID-19 Impact (%)")
            else:
                st.warning("Insufficient data to analyze immediate COVID-19 impact")
        
        # Tab 2: Recovery
        with tab2:
            if "recovery" in covid_metrics:
                recovery_data = covid_metrics["recovery"]
                
                st.subheader("Economic Recovery After COVID-19 (2020 to 2023)")
                
                # Display key statistics
                avg_recovery = recovery_data["Recovery_Percent"].mean()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Recovery", f"{avg_recovery:.2f}%")
                with col2:
                    st.metric("Slowest Recovery", 
                             f"{recovery_data['Recovery_Percent'].min():.2f}%",
                             f"{recovery_data.loc[recovery_data['Recovery_Percent'].idxmin(), 'COUNTY']} County")
                with col3:
                    st.metric("Fastest Recovery", 
                             f"{recovery_data['Recovery_Percent'].max():.2f}%",
                             f"{recovery_data.loc[recovery_data['Recovery_Percent'].idxmax(), 'COUNTY']} County")
                
                # Display map
                self.show_covid_impact_map(recovery_data, "Recovery_Percent", 
                                         "Recovery Rate (%)", "RdYlGn")
                
                # Show top and bottom counties
                self.show_top_bottom_counties(recovery_data, "Recovery_Percent", 
                                            "Recovery Rate (%)")
            else:
                st.warning("Insufficient data to analyze COVID-19 recovery")
        
        # Tab 3: Overall Change
        with tab3:
            if "overall_change" in covid_metrics:
                overall_data = covid_metrics["overall_change"]
                
                st.subheader("Overall Economic Change (2019 to 2023)")
                
                # Display key statistics
                avg_change = overall_data["Overall_Change_Percent"].mean()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Change", f"{avg_change:.2f}%")
                with col2:
                    st.metric("Worst Overall Change", 
                             f"{overall_data['Overall_Change_Percent'].min():.2f}%",
                             f"{overall_data.loc[overall_data['Overall_Change_Percent'].idxmin(), 'COUNTY']} County")
                with col3:
                    st.metric("Best Overall Change", 
                             f"{overall_data['Overall_Change_Percent'].max():.2f}%",
                             f"{overall_data.loc[overall_data['Overall_Change_Percent'].idxmax(), 'COUNTY']} County")
                
                # Display map
                self.show_covid_impact_map(overall_data, "Overall_Change_Percent", 
                                         "Overall Change (%)", "RdYlGn")
                
                # Show top and bottom counties
                self.show_top_bottom_counties(overall_data, "Overall_Change_Percent", 
                                            "Overall Change (%)")
            else:
                st.warning("Insufficient data to analyze overall economic change")

    def is_data_loaded(self):
        """Check if data has been loaded"""
        return bool(self.cached_data)

def main():
    """Main function"""
    # Display loading message
    with st.spinner("Loading data... This may take a few seconds."):
        start_time = time.time()
        
        # Initialize app
        app = App()
        
        # Initialize variables
        covid_data = None
        covid_metrics = None
        
        # Load data
        if not app.is_data_loaded():
            st.error("Failed to load data. Please check your data files and permissions.")
        else:
            covid_data = app.cached_data
            covid_metrics = app.create_covid_impact_metrics(covid_data)
            load_time = time.time() - start_time
            st.success(f"Data loaded successfully in {load_time:.2f} seconds")

    # Create sidebar for navigation
    st.sidebar.header("Navigation")
    
    # Create navigation options
    nav_options = ["COVID-19 Impact Analysis", "Spatial Analysis (Moran's I)"]
    selected_nav = st.sidebar.radio("Select Analysis", nav_options)
    
    # Display selected content based on navigation
    if selected_nav == "COVID-19 Impact Analysis":
        if covid_data is not None and covid_metrics is not None:
            app.show_covid_impact_dashboard(covid_data, covid_metrics)
        else:
            st.error("Data loading failed, please check if the data files exist and are formatted correctly.")
    elif selected_nav == "Spatial Analysis (Moran's I)":
        if app.is_data_loaded():
            app.show_moran_results()
        else:
            st.error("Data loading failed, unable to display spatial analysis results.")
    
    # Add information footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    This dashboard analyzes the economic impact of COVID-19 on Texas counties by comparing:
    - Pre-COVID (2019)
    - During COVID (2020)
    - Post-COVID recovery (2023)
    """)

if __name__ == "__main__":
    main() 
