import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import os
from datetime import datetime
from collections import Counter
import string
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Import the DataCleaningAgent class
from data_cleaning_agent import DataCleaningAgent

# Set page configuration
st.set_page_config(
    page_title="Data Cleaning Agent",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .success-text {
        color: #28a745;
        font-weight: 600;
    }
    .warning-text {
        color: #ffc107;
        font-weight: 600;
    }
    .error-text {
        color: #dc3545;
        font-weight: 600;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'data_analyzed' not in st.session_state:
    st.session_state.data_analyzed = False
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = []
if 'dashboard_recommendations' not in st.session_state:
    st.session_state.dashboard_recommendations = []
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = []
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Overview"
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# Main header
st.markdown("<div class='main-header'>Data Cleaning Agent 🧹</div>", unsafe_allow_html=True)
st.markdown("<div class='info-text'>Upload your data, clean it automatically, and get analysis with visualization recommendations.</div>", unsafe_allow_html=True)

# Create sidebar
st.sidebar.markdown("<div class='sidebar-header'>Data Processing Pipeline</div>", unsafe_allow_html=True)

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls', 'json', 'txt'])

# Process uploaded file
if uploaded_file is not None:
    try:
        # Save the file temporarily
        file_extension = uploaded_file.name.split('.')[-1].lower()
        temp_file_path = f"temp_upload.{file_extension}"
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Create a new agent if not already created or if different file
        if st.session_state.agent is None or st.session_state.file_name != uploaded_file.name:
            st.session_state.agent = DataCleaningAgent()
            st.session_state.file_name = uploaded_file.name
            st.session_state.data_loaded = False
            st.session_state.data_cleaned = False
            st.session_state.data_analyzed = False
        
        # Display file info
        st.sidebar.success(f"File uploaded: {uploaded_file.name}")
        
        # Load data button
        if st.sidebar.button("1. Load Data") or st.session_state.data_loaded:
            if not st.session_state.data_loaded:
                with st.spinner("Loading data..."):
                    load_result = st.session_state.agent.load_data(temp_file_path)
                st.session_state.data_loaded = True
                
            st.sidebar.success("✅ Data loaded successfully")
            
            # Clean data button
            if st.sidebar.button("2. Clean Data") or st.session_state.data_cleaned:
                if not st.session_state.data_cleaned:
                    with st.spinner("Cleaning data..."):
                        st.session_state.cleaning_log = st.session_state.agent.clean_data()
                    st.session_state.data_cleaned = True
                
                st.sidebar.success(f"✅ Data cleaned with {len(st.session_state.cleaning_log)} actions")
                
                # Analyze data button
                if st.sidebar.button("3. Analyze Data") or st.session_state.data_analyzed:
                    if not st.session_state.data_analyzed:
                        with st.spinner("Analyzing data..."):
                            st.session_state.insights = st.session_state.agent.analyze_data()
                            st.session_state.viz_recommendations = st.session_state.agent.get_visualization_recommendations()
                            st.session_state.dashboard_recommendations = st.session_state.agent.generate_powerbi_recommendations()
                        st.session_state.data_analyzed = True
                    
                    st.sidebar.success("✅ Data analysis complete")
    
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Download button for cleaned data
if st.session_state.data_cleaned and st.session_state.agent is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div class='sidebar-header'>Download Cleaned Data</div>", unsafe_allow_html=True)
    
    # Add download format options
    download_format = st.sidebar.selectbox("Select download format", ["CSV", "Excel", "JSON"])
    
    if st.sidebar.button("Download Cleaned Data"):
        try:
            # Create a download link based on the selected format
            if download_format == "CSV":
                csv_data = st.session_state.agent.df.to_csv(index=False)
                b64 = base64.b64encode(csv_data.encode()).decode()
                download_filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">Click to download CSV</a>'
            
            elif download_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.agent.df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                b64 = base64.b64encode(output.getvalue()).decode()
                download_filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{download_filename}">Click to download Excel</a>'
            
            elif download_format == "JSON":
                json_data = st.session_state.agent.df.to_json(orient='records')
                b64 = base64.b64encode(json_data.encode()).decode()
                download_filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                href = f'<a href="data:file/json;base64,{b64}" download="{download_filename}">Click to download JSON</a>'
            
            st.sidebar.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.sidebar.error(f"Download error: {str(e)}")

# Create tabs for different sections
if st.session_state.data_loaded:
    tabs = ["Overview", "Cleaning", "Analysis", "Visualization", "Dashboard Recommendations"]
    
    col1, col2, col3, col4, col5 = st.tabs(tabs)
    
    # Overview Tab
    with col1:
        st.markdown("<div class='sub-header'>Dataset Overview</div>", unsafe_allow_html=True)
        
        if st.session_state.agent is not None:
            # Display basic information about the dataset
            df = st.session_state.agent.df
            
            col1a, col1b = st.columns(2)
            
            with col1a:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**Rows:** {df.shape[0]}")
                st.markdown(f"**Columns:** {df.shape[1]}")
                st.markdown(f"**Missing Values:** {df.isna().sum().sum()}")
                st.markdown(f"**Duplicate Rows:** {df.duplicated().sum()}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col1b:
                if st.session_state.agent.column_types:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("**Column Types:**")
                    type_counts = Counter(st.session_state.agent.column_types.values())
                    st.markdown(f"• Numeric: {type_counts.get('numeric', 0)}")
                    st.markdown(f"• Categorical: {type_counts.get('categorical', 0)}")
                    st.markdown(f"• Datetime: {type_counts.get('datetime', 0)}")
                    st.markdown(f"• Text: {type_counts.get('text', 0)}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Display sample data
            st.markdown("<div class='sub-header'>Data Sample</div>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display column information
            st.markdown("<div class='sub-header'>Column Information</div>", unsafe_allow_html=True)
            
            col_info = []
            for col in df.columns:
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                
                col_type = st.session_state.agent.column_types.get(col, "unknown")
                unique_count = df[col].nunique()
                
                col_info.append({
                    "Column": col,
                    "Type": col_type.capitalize(),
                    "Non-Null Count": f"{len(df) - missing} ({100 - missing_pct:.1f}%)",
                    "Missing Count": f"{missing} ({missing_pct:.1f}%)",
                    "Unique Values": unique_count
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
    
    # Cleaning Tab
    with col2:
        if st.session_state.data_cleaned:
            st.markdown("<div class='sub-header'>Cleaning Actions</div>", unsafe_allow_html=True)
            
            # Display cleaning log
            if st.session_state.cleaning_log:
                for i, action in enumerate(st.session_state.cleaning_log):
                    st.markdown(f"<div class='card'>{i+1}. {action}</div>", unsafe_allow_html=True)
            else:
                st.info("No cleaning actions performed")
            
            # Display before-after comparison
            if st.session_state.agent.original_df is not None:
                st.markdown("<div class='sub-header'>Before vs After Cleaning</div>", unsafe_allow_html=True)
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.markdown("**Before Cleaning:**")
                    st.dataframe(st.session_state.agent.original_df.head(5), use_container_width=True)
                
                with col2b:
                    st.markdown("**After Cleaning:**")
                    st.dataframe(st.session_state.agent.df.head(5), use_container_width=True)
        else:
            st.info("Please clean the data first")
    
    # Analysis Tab
    with col3:
        if st.session_state.data_analyzed:
            st.markdown("<div class='sub-header'>Data Insights</div>", unsafe_allow_html=True)
            
            # Display insights
            for insight in st.session_state.insights:
                insight_type = insight.get("type", "")
                description = insight.get("description", "")
                
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{insight_type.replace('_', ' ').title()}**")
                st.markdown(description)
                
                # Display additional details based on insight type
                if insight_type == "numeric_analysis":
                    if "stats" in insight:
                        stats = insight["stats"]
                        st.markdown("**Statistics:**")
                        st.markdown(f"• Mean: {stats['mean']:.2f}")
                        st.markdown(f"• Median: {stats['median']:.2f}")
                        st.markdown(f"• Std Dev: {stats['std']:.2f}")
                        st.markdown(f"• Min: {stats['min']:.2f}")
                        st.markdown(f"• Max: {stats['max']:.2f}")
                
                elif insight_type == "categorical_analysis":
                    if "top_categories" in insight:
                        top_cats = insight["top_categories"]
                        st.markdown("**Top Categories:**")
                        for cat, count in top_cats.items():
                            st.markdown(f"• {cat}: {count}")
                
                elif insight_type == "correlations":
                    if "details" in insight:
                        details = insight["details"]
                        for detail in details:
                            st.markdown(f"• {detail}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Create some basic visualizations
            st.markdown("<div class='sub-header'>Quick Visualizations</div>", unsafe_allow_html=True)
            
            # Get column types
            numeric_columns = [col for col, type_ in st.session_state.agent.column_types.items() 
                              if type_ == "numeric" and col in st.session_state.agent.df.columns]
            categorical_columns = [col for col, type_ in st.session_state.agent.column_types.items() 
                                  if type_ == "categorical" and col in st.session_state.agent.df.columns]
            
            # Create some plots
            if numeric_columns:
                col3a, col3b = st.columns(2)
                
                with col3a:
                    st.markdown("**Distribution of a Numeric Variable**")
                    selected_num_col = st.selectbox("Select numeric column", numeric_columns)
                    fig = px.histogram(st.session_state.agent.df, x=selected_num_col, 
                                      marginal="box", title=f"Distribution of {selected_num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col3b:
                    if len(numeric_columns) >= 2:
                        st.markdown("**Scatter Plot**")
                        x_col = st.selectbox("Select X axis", numeric_columns, key="x_axis")
                        y_col = st.selectbox("Select Y axis", numeric_columns, key="y_axis", 
                                              index=min(1, len(numeric_columns)-1))
                        
                        fig = px.scatter(st.session_state.agent.df, x=x_col, y=y_col, 
                                        title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            if categorical_columns:
                st.markdown("**Categorical Variable Analysis**")
                selected_cat_col = st.selectbox("Select categorical column", categorical_columns)
                
                fig = px.bar(st.session_state.agent.df[selected_cat_col].value_counts().reset_index(), 
                            x="index", y=selected_cat_col, title=f"Count by {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please analyze the data first")
    
    # Visualization Tab
    with col4:
        if st.session_state.data_analyzed:
            st.markdown("<div class='sub-header'>Visualization Recommendations</div>", unsafe_allow_html=True)
            
            # Display visualization recommendations
            for i, viz in enumerate(st.session_state.viz_recommendations):
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{viz.get('title', f'Visualization {i+1}')}**")
                st.markdown(f"*{viz.get('description', '')}*")
                st.markdown(f"**Type:** {viz.get('type', '').replace('_', ' ').title()}")
                
                # Display specific visualization (for common types)
                viz_type = viz.get('type', '')
                
                if viz_type == 'histogram' and 'column' in viz:
                    column = viz['column']
                    if column in st.session_state.agent.df.columns:
                        fig = px.histogram(st.session_state.agent.df, x=column, 
                                          title=f"Distribution of {column}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == 'bar_chart' and 'column' in viz:
                    column = viz['column']
                    if column in st.session_state.agent.df.columns:
                        value_counts = st.session_state.agent.df[column].value_counts().reset_index()
                        fig = px.bar(value_counts, x='index', y=column, 
                                    title=f"Count by {column}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == 'box_plot' and 'column' in viz:
                    column = viz['column']
                    if column in st.session_state.agent.df.columns:
                        fig = px.box(st.session_state.agent.df, y=column, 
                                    title=f"Box Plot of {column}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == 'scatter_plot' and 'x_axis' in viz and 'y_axis' in viz:
                    x_col = viz['x_axis']
                    y_col = viz['y_axis']
                    if x_col in st.session_state.agent.df.columns and y_col in st.session_state.agent.df.columns:
                        fig = px.scatter(st.session_state.agent.df, x=x_col, y=y_col, 
                                        title=f"{x_col} vs {y_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == 'line_chart' and 'column' in viz and 'time_grouping' in viz:
                    column = viz['column']
                    time_grouping = viz['time_grouping']
                    
                    if column in st.session_state.agent.df.columns:
                        # Create time series visualization
                        try:
                            # Make sure column is datetime
                            df_temp = st.session_state.agent.df.copy()
                            if not pd.api.types.is_datetime64_any_dtype(df_temp[column]):
                                df_temp[column] = pd.to_datetime(df_temp[column], errors='coerce')
                            
                            # Group by time period
                            if time_grouping == 'year':
                                grouped = df_temp.groupby(df_temp[column].dt.year).size().reset_index(name='count')
                                x_col = column
                            elif time_grouping == 'month':
                                grouped = df_temp.groupby(df_temp[column].dt.to_period('M')).size().reset_index(name='count')
                                grouped[column] = grouped[column].dt.to_timestamp()
                                x_col = column
                            else:
                                grouped = df_temp.groupby(column).size().reset_index(name='count')
                                x_col = column
                            
                            fig = px.line(grouped, x=x_col, y='count', 
                                        title=f"Time Series by {time_grouping.capitalize()}")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Couldn't create time series visualization: {str(e)}")
                
                elif viz_type == 'correlation_matrix':
                    if 'columns' in viz:
                        columns = viz['columns']
                        # Filter for columns that exist in the dataframe
                        valid_columns = [col for col in columns if col in st.session_state.agent.df.columns]
                        
                        if len(valid_columns) >= 2:
                            corr_matrix = st.session_state.agent.df[valid_columns].corr()
                            
                            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                          title="Correlation Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please analyze the data first")
    
    # Dashboard Recommendations Tab
    with col5:
        if st.session_state.data_analyzed:
            st.markdown("<div class='sub-header'>PowerBI Dashboard Recommendations</div>", unsafe_allow_html=True)
            
            # Create a sidebar for selecting dashboard pages
            if st.session_state.dashboard_recommendations:
                pages = [page['page_name'] for page in st.session_state.dashboard_recommendations]
                selected_page = st.selectbox("Select Dashboard Page", pages)
                
                # Find the selected page
                selected_page_data = next((page for page in st.session_state.dashboard_recommendations 
                                         if page['page_name'] == selected_page), None)
                
                if selected_page_data:
                    st.markdown(f"### {selected_page_data['page_name']}")
                    st.markdown(f"*{selected_page_data['description']}*")
                    
                    # Display visualizations for this page
                    for i, viz in enumerate(selected_page_data.get('visualizations', [])):
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"**{viz.get('title', f'Visualization {i+1}')}**")
                        st.markdown(f"*{viz.get('description', '')}*")
                        st.markdown(f"**Type:** {viz.get('type', '').replace('_', ' ').title()}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Generate a mockup of the dashboard page
                    st.markdown("<div class='sub-header'>Dashboard Mockup</div>", unsafe_allow_html=True)
                    
                    # Create the mockup visualization
                    try:
                        # Create a figure with subplots based on the number of visualizations
                        num_viz = len(selected_page_data.get('visualizations', []))
                        
                        if num_viz > 0:
                            # Determine layout (rows and columns)
                            cols = min(2, num_viz)
                            rows = (num_viz + cols - 1) // cols
                            
                            # Create figure with subplots
                            fig = go.Figure()
                            
                            # Add a background rectangle to simulate a dashboard
                            fig.add_shape(
                                type="rect",
                                x0=0, y0=0, x1=1, y1=1,
                                fillcolor="white",
                                line=dict(color="lightgray", width=2)
                            )
                            
                            # Add title
                            fig.add_annotation(
                                x=0.5, y=0.95,
                                text=f"<b>{selected_page_data['page_name']} Dashboard</b>",
                                showarrow=False,
                                font=dict(size=20)
                            )
                            
                            # Add placeholder boxes for visualizations
                            for i, viz in enumerate(selected_page_data.get('visualizations', [])):
                                col = i % cols
                                row = i // cols
                                
                                x_center = (col + 0.5) / cols
                                y_center = 0.85 - (row + 0.5) / (rows + 1)
                                
                                width = 0.9 / cols
                                height = 0.7 / (rows + 0.5)
                                
                                # Add rectangle for visualization
                                fig.add_shape(
                                    type="rect",
                                    x0=x_center - width/2, 
                                    y0=y_center - height/2,
                                    x1=x_center + width/2, 
                                    y1=y_center + height/2,
                                    fillcolor="aliceblue",
                                    line=dict(color="royalblue")
                                )
                                
                                # Add visualization title
                                fig.add_annotation(
                                    x=x_center, 
                                    y=y_center + height/2 - 0.03,
                                    text=f"<b>{viz.get('title', f'Visualization {i+1}')}</b>",
                                    showarrow=False,
                                    yanchor="top",
                                    font=dict(size=12)
                                )
                                
                                # Add visualization type
                                fig.add_annotation(
                                    x=x_center, 
                                    y=y_center,
                                    text=f"{viz.get('type', '').replace('_', ' ').title()}",
                                    showarrow=False,
                                    font=dict(size=10)
                                )
                            
                            # Update layout
                            fig.update_layout(
                                height=600,
                                showlegend=False,
                                plot_bgcolor='white',
                                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                yaxis=dict(showgrid=False, zeroline=False, visible=False)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Couldn't create dashboard mockup: {str(e)}")
                        
            else:
                st.info("No dashboard recommendations available")
        else:
            st.info("Please analyze the data first")

else:
    # Show instructions when no data is loaded
    st.info("Please upload a data file and follow the steps in the sidebar to get started.")
    
    st.markdown("<div class='sub-header'>How to Use This App</div>", unsafe_allow_html=True)
    st.markdown("""
    1. **Upload your data file** (CSV, Excel, JSON, or TXT) using the uploader in the sidebar
    2. **Load the data** by clicking the "Load Data" button
    3. **Clean the data** automatically with the "Clean Data" button
    4. **Analyze the data** to get insights and visualization recommendations
    5. **Explore the different tabs** to see the results:
       - Overview: Basic dataset information
       - Cleaning: Cleaning actions performed
       - Analysis: Data insights and visualizations
       - Visualization: Recommended visualizations
       - Dashboard Recommendations: PowerBI dashboard mockups
    6. **Download the cleaned data** in your preferred format
    """)
    
    st.markdown("<div class='sub-header'>Features</div>", unsafe_allow_html=True)
    st.markdown("""
    - Automatic data type detection
    - Missing value handling
    - Outlier detection
    - Format standardization
    - Data quality insights
    - Visualization recommendations
    - PowerBI dashboard suggestions
    """)

# Add footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Data Cleaning Agent © 2023</div>", unsafe_allow_html=True)
