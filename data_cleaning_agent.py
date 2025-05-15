import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import Counter
import string
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json

class DataCleaningAgent:
    """
    A class that provides automated data cleaning, analysis, and visualization recommendations.
    """
    
    def __init__(self):
        """Initialize the DataCleaningAgent."""
        self.df = None
        self.original_df = None
        self.column_types = {}
        self.cleaning_log = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
    
    def load_data(self, file_path):
        """
        Load data from various file formats.
        
        Args:
            file_path (str): Path to the data file.
            
        Returns:
            dict: A dictionary with loading status and information.
        """
        try:
            # Get file extension
            file_extension = file_path.split('.')[-1].lower()
            
            # Load data based on file extension
            if file_extension == 'csv':
                self.df = pd.read_csv(file_path, low_memory=False)
            elif file_extension in ['xlsx', 'xls']:
                self.df = pd.read_excel(file_path)
            elif file_extension == 'json':
                self.df = pd.read_json(file_path)
            elif file_extension == 'txt':
                # Try to infer delimiter for text files
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                
                # Check for common delimiters
                if ',' in first_line:
                    self.df = pd.read_csv(file_path)
                elif '\t' in first_line:
                    self.df = pd.read_csv(file_path, sep='\t')
                elif ';' in first_line:
                    self.df = pd.read_csv(file_path, sep=';')
                elif '|' in first_line:
                    self.df = pd.read_csv(file_path, sep='|')
                else:
                    # If delimiter can't be inferred, try space
                    self.df = pd.read_csv(file_path, delim_whitespace=True)
            else:
                return {"status": "error", "message": f"Unsupported file format: {file_extension}"}
            
            # Make a copy of the original data
            self.original_df = self.df.copy()
            
            # Infer column types
            self._infer_column_types()
            
            return {
                "status": "success",
                "rows": self.df.shape[0],
                "columns": self.df.shape[1],
                "column_types": self.column_types
            }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _infer_column_types(self):
        """Infer the data types of columns."""
        self.column_types = {}
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        
        # Define date/time related keywords
        datetime_keywords = ['date', 'month', 'year', 'day']
        
        for column in self.df.columns:
            # Check if column name contains datetime keywords
            col_lower = column.lower()
            contains_datetime_keyword = any(keyword in col_lower for keyword in datetime_keywords)
            
            # Try to convert to datetime only if column name contains datetime keywords
            if contains_datetime_keyword:
                try:
                    pd.to_datetime(self.df[column], errors='raise')
                    self.column_types[column] = "datetime"
                    self.datetime_columns.append(column)
                    continue
                except:
                    # If conversion fails, continue with other type checks
                    pass
            
            # Check if numeric
            if pd.api.types.is_numeric_dtype(self.df[column]):
                # If few unique values relative to total, treat as categorical
                if self.df[column].nunique() < min(10, self.df.shape[0] * 0.05):
                    self.column_types[column] = "categorical"
                    self.categorical_columns.append(column)
                else:
                    self.column_types[column] = "numeric"
                    self.numeric_columns.append(column)
            
            # Check if categorical
            elif self.df[column].nunique() < min(20, self.df.shape[0] * 0.1):
                self.column_types[column] = "categorical"
                self.categorical_columns.append(column)
            
            # Check if text (strings with many characters)
            elif (self.df[column].dtype == 'object' and 
                  self.df[column].fillna('').astype(str).str.len().mean() > 20):
                self.column_types[column] = "text"
                self.text_columns.append(column)
            
            # Default to categorical for remaining columns
            else:
                self.column_types[column] = "categorical"
                self.categorical_columns.append(column)
    
    def clean_data(self):
        """
        Perform automated data cleaning operations.
        
        Returns:
            list: A list of cleaning operations performed.
        """
        if self.df is None:
            return ["No data loaded."]
        
        self.cleaning_log = []
        
        # 1. Handle duplicate rows
        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = initial_rows - self.df.shape[0]
        if duplicates_removed > 0:
            self.cleaning_log.append(f"Removed {duplicates_removed} duplicate rows.")
        
        # 2. Handle column names
        # Replace spaces and special characters in column names
        old_columns = self.df.columns.tolist()
        new_columns = [re.sub(r'[^\w]', '_', col).strip('_').lower() for col in old_columns]
        self.df.columns = new_columns
        
        renamed_cols = sum(1 for old, new in zip(old_columns, new_columns) if old != new)
        if renamed_cols > 0:
            self.cleaning_log.append(f"Standardized {renamed_cols} column names.")
        
        # 3. Convert data types
        datetime_keywords = ['time', 'date', 'month', 'year', 'day']
        
        for col in self.df.columns:
            col_type = self.column_types.get(col, "unknown")
            
            # Handle datetime columns - only convert if column name contains datetime keywords
            if col_type == "datetime":
                col_lower = col.lower()
                contains_datetime_keyword = any(keyword in col_lower for keyword in datetime_keywords)
                
                if contains_datetime_keyword:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        self.cleaning_log.append(f"Converted '{col}' to datetime format.")
                    except:
                        pass
            
            # Handle numeric columns
            elif col_type == "numeric":
                # Check if column contains numeric-like strings
                if self.df[col].dtype == 'object':
                    try:
                        # Remove currency symbols, commas, etc.
                        cleaned_col = self.df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        self.df[col] = pd.to_numeric(cleaned_col, errors='coerce')
                        self.cleaning_log.append(f"Converted '{col}' to numeric format.")
                    except:
                        pass
        
        # 4. Handle missing values
        for col in self.df.columns:
            col_type = self.column_types.get(col, "unknown")
            missing_count = self.df[col].isna().sum()
            
            if missing_count > 0:
                missing_pct = (missing_count / len(self.df)) * 100
                
                # If too many missing (>50%), consider dropping the column
                if missing_pct > 50:
                    self.cleaning_log.append(f"Column '{col}' has {missing_pct:.1f}% missing values. Consider removing it.")
                else:
                    # Impute based on data type
                    if col_type == "numeric":
                        # Use median for numeric columns
                        median_val = self.df[col].median()
                        self.df[col].fillna(median_val, inplace=True)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f}).")
                    
                    elif col_type == "categorical":
                        # Use mode for categorical columns
                        mode_val = self.df[col].mode()[0]
                        self.df[col].fillna(mode_val, inplace=True)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with mode ({mode_val}).")
                    
                    elif col_type == "datetime":
                        # For datetime, fill with previous or next value if makes sense
                        if missing_pct < 10:  # Only if small percentage
                            self.df[col].fillna(method='ffill', inplace=True)
                            remaining_missing = self.df[col].isna().sum()
                            if remaining_missing > 0:
                                self.df[col].fillna(method='bfill', inplace=True)
                            
                            filled = missing_count - self.df[col].isna().sum()
                            if filled > 0:
                                self.cleaning_log.append(f"Filled {filled} missing datetime values in '{col}' using interpolation.")
                    
                    elif col_type == "text":
                        # For text, fill with empty string
                        self.df[col].fillna("", inplace=True)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with empty string.")
        
        # 5. Handle outliers in numeric columns
        for col in self.numeric_columns:
            if col in self.df.columns:  # Make sure column still exists
                # Use IQR method to detect outliers
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                if outliers > 0 and outliers < len(self.df) * 0.05:  # Only if small percentage
                    self.cleaning_log.append(f"Detected {outliers} outliers in '{col}'. Consider handling them.")
        
        # 6. Standardize text columns
        for col in self.text_columns:
            if col in self.df.columns:  # Make sure column still exists
                # Convert to string and strip whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Check for mixed case and standardize if needed
                uppercase_pct = (self.df[col].str.isupper() & (self.df[col].str.len() > 3)).mean() * 100
                if uppercase_pct > 30:  # If significant portion is all uppercase
                    self.df[col] = self.df[col].str.title()
                    self.cleaning_log.append(f"Standardized capitalization in '{col}'.")
        
        # 7. Format standardization for categorical columns
        for col in self.categorical_columns:
            if col in self.df.columns:  # Make sure column still exists
                # Only process string categories
                if self.df[col].dtype == 'object':
                    # Strip whitespace and standardize case
                    old_nunique = self.df[col].nunique()
                    self.df[col] = self.df[col].astype(str).str.strip().str.lower()
                    new_nunique = self.df[col].nunique()
                    
                    if old_nunique != new_nunique:
                        diff = old_nunique - new_nunique
                        self.cleaning_log.append(f"Standardized {diff} category values in '{col}'.")
        
        # 8. Final data type conversion
        for col in self.df.columns:
            col_type = self.column_types.get(col, "unknown")
            
            if col_type == "numeric" and not pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass
        
        # Re-infer column types after cleaning
        self._infer_column_types()
        
        return self.cleaning_log

    def analyze_data(self):
        """
        Perform basic data analysis and generate insights.
        
        Returns:
            list: A list of insights about the data.
        """
        if self.df is None:
            return []
        
        insights = []
        
        # 1. Basic dataset statistics
        total_rows = self.df.shape[0]
        total_cols = self.df.shape[1]
        missing_cells = self.df.isna().sum().sum()
        missing_pct = (missing_cells / (total_rows * total_cols)) * 100
        
        insights.append({
            "type": "overview",
            "description": f"Dataset contains {total_rows} rows and {total_cols} columns with {missing_pct:.2f}% missing values overall."
        })
        
        # 2. Analyze numeric columns
        for col in self.numeric_columns:
            if col in self.df.columns:  # Ensure column exists
                try:
                    # Calculate basic statistics
                    stats = {
                        "mean": self.df[col].mean(),
                        "median": self.df[col].median(),
                        "std": self.df[col].std(),
                        "min": self.df[col].min(),
                        "max": self.df[col].max()
                    }
                    
                    # Check distribution
                    skewness = self.df[col].skew()
                    distribution = "normally distributed"
                    if skewness > 1:
                        distribution = "right-skewed (positively skewed)"
                    elif skewness < -1:
                        distribution = "left-skewed (negatively skewed)"
                    
                    insights.append({
                        "type": "numeric_analysis",
                        "column": col,
                        "description": f"The column '{col}' is {distribution} with values ranging from {stats['min']:.2f} to {stats['max']:.2f}.",
                        "stats": stats
                    })
                except Exception as e:
                    pass
        
        # 3. Analyze categorical columns
        for col in self.categorical_columns:
            if col in self.df.columns:  # Ensure column exists
                try:
                    # Get value counts
                    value_counts = self.df[col].value_counts()
                    total_categories = len(value_counts)
                    
                    # Get top categories (up to 5)
                    top_categories = value_counts.head(5).to_dict()
                    
                    # Calculate diversity (entropy)
                    entropy = stats.entropy(value_counts.values) if total_categories > 1 else 0
                    diversity = "very diverse" if entropy > 3 else "moderately diverse" if entropy > 1.5 else "not very diverse"
                    
                    insights.append({
                        "type": "categorical_analysis",
                        "column": col,
                        "description": f"The column '{col}' has {total_categories} unique values and is {diversity}.",
                        "top_categories": top_categories
                    })
                except Exception as e:
                    pass
        
        # 4. Analyze datetime columns
        for col in self.datetime_columns:
            if col in self.df.columns:  # Ensure column exists
                try:
                    # Make sure column is datetime type
                    if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    
                    # Get range
                    min_date = self.df[col].min()
                    max_date = self.df[col].max()
                    date_range = (max_date - min_date).days
                    
                    insights.append({
                        "type": "datetime_analysis",
                        "column": col,
                        "description": f"The column '{col}' spans {date_range} days from {min_date.date()} to {max_date.date()}."
                    })
                except Exception as e:
                    pass
        
        # 5. Analyze text columns
        for col in self.text_columns:
            if col in self.df.columns:  # Ensure column exists
                try:
                    # Calculate text length statistics
                    avg_length = self.df[col].astype(str).str.len().mean()
                    max_length = self.df[col].astype(str).str.len().max()
                    
                    insights.append({
                        "type": "text_analysis",
                        "column": col,
                        "description": f"The text column '{col}' has an average length of {avg_length:.1f} characters (max: {max_length})."
                    })
                except Exception as e:
                    pass
        
        # 6. Correlation analysis for numeric columns
        if len(self.numeric_columns) >= 2:
            try:
                # Calculate correlation matrix
                corr_matrix = self.df[self.numeric_columns].corr()
                
                # Find strong correlations (absolute value > 0.7)
                strong_correlations = []
                for i in range(len(self.numeric_columns)):
                    for j in range(i+1, len(self.numeric_columns)):
                        col1 = self.numeric_columns[i]
                        col2 = self.numeric_columns[j]
                        
                        if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                            corr = corr_matrix.loc[col1, col2]
                            
                            if abs(corr) > 0.7:
                                direction = "positive" if corr > 0 else "negative"
                                strong_correlations.append(f"{col1} and {col2} have a strong {direction} correlation ({corr:.2f})")
                
                if strong_correlations:
                    insights.append({
                        "type": "correlations",
                        "description": f"Found {len(strong_correlations)} strong correlations between numeric columns.",
                        "details": strong_correlations
                    })
            except Exception as e:
                pass
        
        return insights

    def get_visualization_recommendations(self):
        """
        Generate recommendations for visualizations based on data types.
        
        Returns:
            list: A list of recommended visualizations.
        """
        if self.df is None:
            return []
        
        recommendations = []
        
        # 1. Histograms for numeric columns
        for col in self.numeric_columns[:3]:  # Limit to first 3 for brevity
            if col in self.df.columns:
                recommendations.append({
                    "type": "histogram",
                    "column": col,
                    "title": f"Distribution of {col}",
                    "description": f"Histogram showing the distribution of values in the {col} column."
                })
        
        # 2. Bar charts for categorical columns
        for col in self.categorical_columns[:3]:  # Limit to first 3
            if col in self.df.columns:
                recommendations.append({
                    "type": "bar_chart",
                    "column": col,
                    "title": f"Counts by {col}",
                    "description": f"Bar chart showing the count of occurrences for each category in the {col} column."
                })
        
        # 3. Box plots for numeric columns
        for col in self.numeric_columns[:3]:  # Limit to first 3
            if col in self.df.columns:
                recommendations.append({
                    "type": "box_plot",
                    "column": col,
                    "title": f"Box Plot of {col}",
                    "description": f"Box plot showing the distribution and potential outliers in the {col} column."
                })
        
        # 4. Scatter plots for pairs of numeric columns
        if len(self.numeric_columns) >= 2:
            # Create scatter plot recommendations for first few pairs
            for i in range(min(2, len(self.numeric_columns))):
                for j in range(i+1, min(3, len(self.numeric_columns))):
                    col1 = self.numeric_columns[i]
                    col2 = self.numeric_columns[j]
                    
                    if col1 != col2 and col1 in self.df.columns and col2 in self.df.columns:
                        recommendations.append({
                            "type": "scatter_plot",
                            "x_axis": col1,
                            "y_axis": col2,
                            "title": f"{col1} vs {col2}",
                            "description": f"Scatter plot showing the relationship between {col1} and {col2}."
                        })
        
        # 5. Line charts for datetime columns
        for col in self.datetime_columns[:2]:  # Limit to first 2
            if col in self.df.columns:
                # Choose a time grouping based on date range
                try:
                    date_range = (self.df[col].max() - self.df[col].min()).days
                    
                    if date_range > 365*2:  # More than 2 years
                        time_grouping = "year"
                    elif date_range > 90:  # More than 3 months
                        time_grouping = "month"
                    else:
                        time_grouping = "day"
                    
                    recommendations.append({
                        "type": "line_chart",
                        "column": col,
                        "time_grouping": time_grouping,
                        "title": f"Time Series by {time_grouping.capitalize()}",
                        "description": f"Line chart showing the trend over time, grouped by {time_grouping}."
                    })
                except:
                    pass
        
        # 6. Correlation matrix for numeric columns
        if len(self.numeric_columns) >= 3:
            columns = self.numeric_columns[:5]  # Limit to first 5
            
            recommendations.append({
                "type": "correlation_matrix",
                "columns": columns,
                "title": "Correlation Matrix",
                "description": "Heatmap showing the correlations between numeric variables."
            })
        
        # 7. Grouped bar charts for categorical vs categorical
        if len(self.categorical_columns) >= 2:
            col1 = self.categorical_columns[0]
            col2 = self.categorical_columns[1]
             
            if col1 in self.df.columns and col2 in self.df.columns:
                # Check if cardinality isn't too high
                if self.df[col1].nunique() <= 10 and self.df[col2].nunique() <= 10:
                    recommendations.append({
                        "type": "grouped_bar_chart",
                        "x_axis": col1,
                        "group_by": col2,
                        "title": f"{col1} by {col2}",
                        "description": f"Grouped bar chart showing the relationship between {col1} and {col2}."
                    })
        
        # 8. Box plot with categorical and numeric
        if self.categorical_columns and self.numeric_columns:
            cat_col = self.categorical_columns[0]
            num_col = self.numeric_columns[0]
            
            if cat_col in self.df.columns and num_col in self.df.columns:
                # Check if categorical column has reasonable cardinality
                if self.df[cat_col].nunique() <= 10:
                    recommendations.append({
                        "type": "box_plot_by_category",
                        "x_axis": cat_col,
                        "y_axis": num_col,
                        "title": f"{num_col} by {cat_col}",
                        "description": f"Box plot showing the distribution of {num_col} for each category in {cat_col}."
                    })
        
        return recommendations

    def generate_powerbi_recommendations(self):
        """
        Generate recommendations for PowerBI dashboard pages and visualizations.
        
        Returns:
            list: A list of dashboard page recommendations.
        """
        if self.df is None:
            return []
        
        dashboard_recommendations = []
        
        # 1. Overview Page
        overview_page = {
            "page_name": "Overview",
            "description": "A high-level summary of the dataset with key metrics and distributions.",
            "visualizations": [
                {
                    "type": "card",
                    "title": "Total Records",
                    "description": "Shows the total number of records in the dataset."
                },
                {
                    "type": "card",
                    "title": "Data Range",
                    "description": "Shows the time span of the dataset."
                }
            ]
        }
        
        # Add visualizations based on column types
        if self.numeric_columns:
            overview_page["visualizations"].append({
                "type": "column_chart",
                "title": f"Distribution of {self.numeric_columns[0]}",
                "description": f"Column chart showing the distribution of {self.numeric_columns[0]}."
            })
        
        if self.categorical_columns:
            overview_page["visualizations"].append({
                "type": "pie_chart",
                "title": f"Breakdown by {self.categorical_columns[0]}",
                "description": f"Pie chart showing the breakdown of records by {self.categorical_columns[0]}."
            })
        
        if self.datetime_columns:
            overview_page["visualizations"].append({
                "type": "area_chart",
                "title": "Trends Over Time",
                "description": "Area chart showing trends over time."
            })
        
        dashboard_recommendations.append(overview_page)
        
        # 2. Detailed Analysis Page
        if self.numeric_columns:
            details_page = {
                "page_name": "Detailed Analysis",
                "description": "Detailed analysis of key metrics and relationships between variables.",
                "visualizations": []
            }
            
            # Add scatter plot if we have multiple numeric columns
            if len(self.numeric_columns) >= 2:
                details_page["visualizations"].append({
                    "type": "scatter_plot",
                    "title": f"{self.numeric_columns[0]} vs {self.numeric_columns[1]}",
                    "description": f"Scatter plot showing relationship between {self.numeric_columns[0]} and {self.numeric_columns[1]}."
                })
            
            # Add box plot if we have categorical and numeric columns
            if self.categorical_columns and self.numeric_columns:
                details_page["visualizations"].append({
                    "type": "box_plot",
                    "title": f"{self.numeric_columns[0]} by {self.categorical_columns[0]}",
                    "description": f"Box plot showing distribution of {self.numeric_columns[0]} for each {self.categorical_columns[0]}."
                })
            
            # Add correlation matrix if we have multiple numeric columns
            if len(self.numeric_columns) >= 3:
                details_page["visualizations"].append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "description": "Heatmap showing correlations between numeric variables."
                })
            
            dashboard_recommendations.append(details_page)
        
        # 3. Temporal Analysis Page (if we have datetime columns)
        if self.datetime_columns:
            temporal_page = {
                "page_name": "Temporal Analysis",
                "description": "Analysis of trends and patterns over time.",
                "visualizations": [
                    {
                        "type": "line_chart",
                        "title": "Trend Over Time",
                        "description": "Line chart showing the trend over time."
                    },
                    {
                        "type": "column_chart",
                        "title": "Monthly Comparison",
                        "description": "Column chart comparing values by month."
                    }
                ]
            }
            
            # If we have numeric columns, add a combo chart
            if self.numeric_columns:
                temporal_page["visualizations"].append({
                    "type": "combo_chart",
                    "title": f"{self.numeric_columns[0]} Over Time",
                    "description": f"Combo chart showing {self.numeric_columns[0]} trends over time."
                })
            
            dashboard_recommendations.append(temporal_page)
        
        # 4. Categorical Breakdown Page (if we have categorical columns)
        if len(self.categorical_columns) >= 2:
            categorical_page = {
                "page_name": "Categorical Breakdown",
                "description": "Analysis of categorical variables and their relationships.",
                "visualizations": [
                    {
                        "type": "treemap",
                        "title": f"Treemap of {self.categorical_columns[0]}",
                        "description": f"Treemap showing the hierarchy of {self.categorical_columns[0]}."
                    },
                    {
                        "type": "stacked_bar_chart",
                        "title": f"{self.categorical_columns[0]} by {self.categorical_columns[1]}",
                        "description": f"Stacked bar chart showing the breakdown of {self.categorical_columns[0]} by {self.categorical_columns[1]}."
                    }
                ]
            }
            
            if len(self.categorical_columns) >= 3:
                categorical_page["visualizations"].append({
                    "type": "donut_chart",
                    "title": f"Distribution of {self.categorical_columns[2]}",
                    "description": f"Donut chart showing the distribution of {self.categorical_columns[2]}."
                })
            
            dashboard_recommendations.append(categorical_page)
        
        return dashboard_recommendations