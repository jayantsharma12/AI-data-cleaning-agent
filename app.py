import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import google.generativeai as genai
from data_cleaning_agent import DataCleaningAgent

# Page configuration
st.set_page_config(
    page_title="Automated Data Cleaning & Analysis",
    page_icon="🧹",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'agent' not in st.session_state: 
    st.session_state.agent = DataCleaningAgent()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'cleaning_performed' not in st.session_state:
    st.session_state.cleaning_performed = False
if 'analysis_performed' not in st.session_state:
    st.session_state.analysis_performed = False
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = []
if 'powerbi_recommendations' not in st.session_state:
    st.session_state.powerbi_recommendations = []
if 'gemini_recommendations' not in st.session_state:
    st.session_state.gemini_recommendations = ""
if 'df' not in st.session_state:
    st.session_state.df = None

# Function to handle file upload and data loading
def load_data(uploaded_file):
    try:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(".", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the data using the agent
        result = st.session_state.agent.load_data(temp_file_path)
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        if result["status"] == "success":
            st.session_state.data_loaded = True
            st.session_state.df = st.session_state.agent.df
            return result
        else:
            st.error(f"Error loading data: {result['message']}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Function to get AI recommendations using Google Gemini API
def get_gemini_recommendations(df_info, insights, visualizations):
    try:
        # Configure the Gemini API with your key
        genai.configure(api_key="AIzaSyA5X_Aq7hUR6NddwKn2RwqtKSbvz0yEQyM")
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prepare the prompt with dataset information
        prompt = f"""
        I have a dataset with the following properties:
        - Rows: {df_info.get('rows', 'N/A')}
        - Columns: {df_info.get('columns', 'N/A')}
        - Column Types: {df_info.get('column_types', {})}
        
        Here are some insights from the data analysis:
        {json.dumps(insights, indent=2)}
        
        And here are the suggested visualizations:
        {json.dumps(visualizations, indent=2)}
        
        Based on this information, please provide:
        1. A summary of what this dataset likely represents
        2. 3-5 key business questions this data could help answer
        3. Additional analysis recommendations beyond what's already suggested
        4. Any data quality issues that should be addressed
        5. Suggested next steps for advanced analytics (e.g., predictive modeling)
        
        Format your response in markdown with clear headings for each section.
        """
        
        # Generate content
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"

# App header
st.title("🧹 Automated Data Cleaning & Analysis")
st.markdown("""
This app helps you clean, analyze, and visualize your data automatically.
Upload a CSV, Excel, or text file to get started.
""")

# Create sidebar for navigation
sidebar = st.sidebar.radio(
    "Navigation",
    ["📤 Upload Data", "🧹 Clean Data", "📊 Analyze Data", "🔍 AI Recommendations"]
)

# Upload Data Page
if sidebar == "📤 Upload Data":
    st.header("📤 Upload Your Data")
    
    uploaded_file = st.file_uploader("Choose a CSV, Excel, or text file", 
                                    type=["csv", "xlsx", "xls", "txt"])
    
    if uploaded_file is not None:
        with st.spinner('Loading data...'):
            result = load_data(uploaded_file)
            
            if result:
                st.success(f"Data loaded successfully! Found {result['rows']} rows and {result['columns']} columns.")
                
                # Display column types
                st.subheader("Column Types Detected")
                col_types_df = pd.DataFrame({
                    'Column': list(result['column_types'].keys()),
                    'Type': list(result['column_types'].values())
                })
                st.dataframe(col_types_df)
                
                # Display sample data
                st.subheader("Sample Data")
                st.dataframe(st.session_state.df.head())
    
    if st.session_state.data_loaded:
        # Summary statistics
        if st.checkbox("Show Summary Statistics"):
            st.subheader("Summary Statistics")
            st.dataframe(st.session_state.df.describe())

# Clean Data Page
elif sidebar == "🧹 Clean Data":
    st.header("🧹 Clean Data")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first!")
    else:
        st.info("Click the button below to automatically clean the data.")
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                cleaning_log = st.session_state.agent.clean_data()
                st.session_state.cleaning_performed = True
                
                # Update the dataframe in session state
                st.session_state.df = st.session_state.agent.df
            
            st.success("Data cleaning completed!")
            
            # Display cleaning log
            st.subheader("Cleaning Operations Performed")
            for log_entry in cleaning_log:
                st.write(f"• {log_entry}")
            
            # Display sample of cleaned data
            st.subheader("Sample of Cleaned Data")
            st.dataframe(st.session_state.df.head())
            
            # Display before/after stats
            if st.checkbox("Show Before/After Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Before Cleaning")
                    st.write(f"**Rows:** {st.session_state.agent.original_df.shape[0]}")
                    st.write(f"**Missing Values:** {st.session_state.agent.original_df.isna().sum().sum()}")
                
                with col2:
                    st.subheader("After Cleaning")
                    st.write(f"**Rows:** {st.session_state.df.shape[0]}")
                    st.write(f"**Missing Values:** {st.session_state.df.isna().sum().sum()}")

# Analyze Data Page
elif sidebar == "📊 Analyze Data":
    st.header("📊 Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first!")
    else:
        st.info("Click the button below to analyze the dataset.")
        
        if st.button("Analyze Data"):
            with st.spinner("Analyzing data..."):
                # Run analysis
                st.session_state.insights = st.session_state.agent.analyze_data()
                st.session_state.viz_recommendations = st.session_state.agent.get_visualization_recommendations()
                st.session_state.powerbi_recommendations = st.session_state.agent.generate_powerbi_recommendations()
                st.session_state.analysis_performed = True
            
            st.success("Analysis completed!")
        
        if st.session_state.analysis_performed:
            # Display insights
            st.subheader("Data Insights")
            
            for insight in st.session_state.insights:
                insight_type = insight.get("type", "")
                description = insight.get("description", "")
                
                if insight_type == "overview":
                    st.info(description)
                elif insight_type == "numeric_analysis":
                    st.write(f"**{insight.get('column', '')}**: {description}")
                    if "stats" in insight:
                        stats = insight["stats"]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mean", f"{stats['mean']:.2f}")
                        col2.metric("Median", f"{stats['median']:.2f}")
                        col3.metric("Std Dev", f"{stats['std']:.2f}")
                elif insight_type == "categorical_analysis":
                    st.write(f"**{insight.get('column', '')}**: {description}")
                    if "top_categories" in insight:
                        st.write("Top categories:")
                        for cat, count in insight["top_categories"].items():
                            st.write(f"• {cat}: {count}")
                elif insight_type == "datetime_analysis":
                    st.write(f"**{insight.get('column', '')}**: {description}")
                elif insight_type == "text_analysis":
                    st.write(f"**{insight.get('column', '')}**: {description}")
                elif insight_type == "correlations":
                    st.write(f"**Correlations**: {description}")
                    if "details" in insight:
                        for detail in insight["details"]:
                            st.write(f"• {detail}")
            
            # Display visualization recommendations
            st.subheader("Visualization Recommendations")
            
            for i, viz in enumerate(st.session_state.viz_recommendations):
                with st.expander(f"{viz['title']}"):
                    st.write(f"**Type:** {viz['type']}")
                    st.write(f"**Description:** {viz['description']}")
            
            # Display PowerBI recommendations
            st.subheader("Dashboard Recommendations")
            
            for page in st.session_state.powerbi_recommendations:
                with st.expander(f"📊 {page['page_name']}"):
                    st.write(f"**Description:** {page['description']}")
                    st.write("**Visualizations:**")
                    for viz in page['visualizations']:
                        st.write(f"• **{viz['title']}**: {viz['description']}")

# AI Recommendations Page
elif sidebar == "🔍 AI Recommendations":
    st.header("🔍 AI-Powered Recommendations")
    
    if not st.session_state.analysis_performed:
        st.warning("Please analyze your data first!")
    else:
        st.info("Generate AI-powered recommendations using Google Gemini.")
        
        if st.button("Generate AI Recommendations") or st.session_state.gemini_recommendations:
            if not st.session_state.gemini_recommendations:
                with st.spinner("Generating AI recommendations..."):
                    # Get information about the dataframe
                    df_info = {
                        "rows": st.session_state.df.shape[0],
                        "columns": st.session_state.df.shape[1],
                        "column_types": st.session_state.agent.column_types
                    }
                    
                    # Get recommendations from Gemini
                    st.session_state.gemini_recommendations = get_gemini_recommendations(
                        df_info, 
                        st.session_state.insights, 
                        st.session_state.viz_recommendations
                    )
            
            # Display the recommendations
            st.markdown(st.session_state.gemini_recommendations)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application helps you automate data cleaning, analysis, and visualization. "
    "Upload your data, perform automated cleaning, and get insights and recommendations."
)