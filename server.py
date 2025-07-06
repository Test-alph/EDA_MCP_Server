from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import Dict, List
import os

from asyncio import Lock
from typing import Dict, Literal
from pathlib import Path
import io, base64, tempfile, asyncio, json, re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import yaml

from mcp.server import FastMCP
from mcp.types import TextContent, ImageContent

from evidently import (
    DataDefinition, Dataset, Report,
    BinaryClassification, MulticlassClassification, Regression
)
from evidently.presets import (
    DataSummaryPreset, DataDriftPreset, RegressionPreset, ClassificationPreset,
)
from pydantic import BaseModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Try to import pyod for outlier detection
try:
    from pyod.models.iforest import IsolationForest
    from pyod.models.lof import LOF
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: pyod not available. Model-based outlier detection will not work.")

load_dotenv()

if "TAVILY_API_KEY" not in os.environ:
    raise Exception("TAVILY_API_KEY environment variable not set")
  
# Tavily API key
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# Initialize Tavily client
tavily_client = TavilyClient(TAVILY_API_KEY)

PORT = os.environ.get("PORT", 10000)

# Create an MCP server
mcp = FastMCP("data-science-eda", host="0.0.0.0", port=PORT)

# Global shared in-memory store and lock
data_store: Dict[str, pd.DataFrame] = {}
store_lock = Lock()

# Add a tool that uses Tavily
@mcp.tool()
def web_search(query: str) -> List[Dict]:
    """
    Use this tool to search the web for information.

    Args:
        query: The search query.

    Returns:
        The search results.
    """
    try:
        response = tavily_client.search(query)
        return response["results"]
    except:
        return "No results found"
#######################################################
# --- Schema inference models -------------------------------------------------
class SchemaColumn(BaseModel):
    name: str
    type: str  # "number", "datetime", "string"
    nullable: bool
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    max_length: int | None = None
    unique_count: int | None = None
    sample_values: list | None = None

class SchemaResult(BaseModel):
    dataset_name: str
    columns: list[SchemaColumn]
    total_rows: int
    total_columns: int

# --- Missing data analysis models -------------------------------------------------
class MissingDataStep(BaseModel):
    step_name: str
    description: str
    formula: str | None = None
    statistics: dict
    recommendations: list[str]

class MissingDataResult(BaseModel):
    dataset_name: str
    total_rows: int
    total_columns: int
    missing_summary: dict
    steps: list[MissingDataStep]
    image_uri: str
    human_checkpoint: str

# --- Outlier detection models -------------------------------------------------
class OutlierStep(BaseModel):
    step_name: str
    description: str
    formula: str
    parameters: dict
    statistics: dict
    recommendations: list[str]

class OutlierPayload(BaseModel):
    outliers: dict[str, list[int]]
    counts: dict[str, int]
    total_rows: int
    steps: list[OutlierStep]
    method_used: str

class OutlierResult(BaseModel):
    result: OutlierPayload
    image_uri: str
    human_checkpoint: str

# --- Target analysis models -------------------------------------------------
class TargetAnalysisResult(BaseModel):
    result: dict
    plots: dict[str, str]

# --- Feature transformation models -------------------------------------------------
class TransformationStep(BaseModel):
    step_name: str
    description: str
    formula: str
    parameters: dict
    statistics: dict
    recommendations: list[str]

class TransformationResult(BaseModel):
    dataset_name: str
    original_shape: tuple
    transformed_shape: tuple
    steps: list[TransformationStep]
    image_uri: str
    human_checkpoint: str

# --- Evidently helpers ------------------------------------------------------
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

def _ds(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, DataDefinition())   # auto-mapping works

def _ds_classification(df: pd.DataFrame) -> Dataset:
    # For classification, explicitly define the task and columns
    return Dataset.from_pandas(
        df,
        DataDefinition(
            classification=[MulticlassClassification(target="target", prediction="prediction")]
        ),
    )

def _ds_regression(df: pd.DataFrame) -> Dataset:
    # For regression, explicitly define the task and columns
    return Dataset.from_pandas(
        df,
        DataDefinition(
            regression=[Regression(target="target", prediction="prediction")]
        ),
    )

# ─────────────────────────── Resources / Helpers ──────────────────────────
def _fig_to_base64_png() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ────────────────────────────────── Tools ──────────────────────────────────
@mcp.tool()
async def load_data(file_path: str, name: str) -> str:
    """
    Load CSV/Excel/JSON into the shared store.
    Returns a confirmation string.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    async with store_lock:
        data_store[name] = df

    # Calculate memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    memory_mb = memory_usage / (1024 * 1024)
    
    return f"Loaded '{name}' with {len(df)} rows × {len(df.columns)} columns. Memory usage: {memory_mb:.2f} MB."


@mcp.tool()
async def basic_info(name: str) -> str:
    """
    Return shape, columns, dtypes and head() for a dataset.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    info = (
        f"Dataset: {name}\n"
        f"Shape: {df.shape}\n\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n\n"
        f"Head:\n{df.head().to_string(index=False)}"
    )
    return info


@mcp.tool()
async def missing_data_analysis(name: str) -> MissingDataResult:
    """
    Comprehensive missing data analysis with clustering, thresholded dropping, 
    median imputation, detailed math and human checkpoints.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    total_rows = len(df)
    total_columns = len(df.columns)
    
    # Step 1: Basic Missing Statistics
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / total_rows * 100).round(3)
    total_missing = missing_counts.sum()
    overall_missing_rate = (total_missing / (total_rows * total_columns) * 100).round(3)
    
    missing_summary = {
        "total_missing_values": int(total_missing),
        "overall_missing_rate_percent": overall_missing_rate,
        "columns_with_missing": int((missing_counts > 0).sum()),
        "missing_by_column": missing_counts.to_dict(),
        "missing_percentages": missing_percentages.to_dict()
    }
    
    steps = []
    
    # Step 2: Missing Data Clustering Analysis
    step2 = MissingDataStep(
        step_name="Missing Data Clustering",
        description="Analyze patterns of missing data across columns to identify systematic missingness",
        formula="Missing Pattern Matrix: M[i,j] = 1 if value is missing, 0 otherwise",
        statistics={
            "missing_patterns": {},
            "correlation_matrix": {},
            "systematic_missingness": False
        },
        recommendations=[]
    )
    
    # Calculate missing pattern correlation
    missing_matrix = df.isnull().astype(int)
    if missing_matrix.sum().sum() > 0:  # Only if there are missing values
        missing_corr = missing_matrix.corr()
        step2.statistics["correlation_matrix"] = missing_corr.round(3).to_dict()
        
        # Check for systematic missingness (high correlation between missing patterns)
        high_corr_pairs = []
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        "col1": missing_corr.columns[i],
                        "col2": missing_corr.columns[j],
                        "correlation": round(corr_val, 3)
                    })
        
        step2.statistics["systematic_missingness"] = len(high_corr_pairs) > 0
        step2.statistics["high_correlation_pairs"] = high_corr_pairs
        
        if len(high_corr_pairs) > 0:
            step2.recommendations.append("Systematic missingness detected - investigate data collection process")
            step2.recommendations.append("Consider if missing patterns indicate meaningful subgroups")
    
    steps.append(step2)
    
    # Step 3: Thresholded Column Dropping Analysis
    step3 = MissingDataStep(
        step_name="Thresholded Column Dropping Analysis",
        description="Identify columns that exceed missing data thresholds for potential removal",
        formula="Drop threshold: Drop column if missing_percentage > threshold (default: 50%)",
        statistics={
            "high_missing_columns": [],
            "recommended_drops": [],
            "threshold_used": 50.0
        },
        recommendations=[]
    )
    
    high_missing_cols = missing_percentages[missing_percentages > 50.0]
    step3.statistics["high_missing_columns"] = high_missing_cols.to_dict()
    
    if len(high_missing_cols) > 0:
        step3.recommendations.append(f"Consider dropping {len(high_missing_cols)} columns with >50% missing data")
        for col, pct in high_missing_cols.items():
            step3.recommendations.append(f"Column '{col}': {pct}% missing - evaluate if essential for analysis")
    
    steps.append(step3)
    
    # Step 4: Missingness Mechanism Analysis (Little's Test)
    step4 = MissingDataStep(
        step_name="Missingness Mechanism Analysis",
        description="Analyze missing data mechanism using Little's test to determine MCAR/MAR/MNAR",
        formula="Little's Test: χ² = Σ(observed - expected)²/expected\nMCAR if p > 0.05, else MAR/MNAR",
        statistics={
            "missingness_mechanism": "unknown",
            "littles_test_statistic": None,
            "littles_test_pvalue": None,
            "mechanism_confidence": "low"
        },
        recommendations=[]
    )
    
    # Simplified Little's test implementation
    if missing_matrix.sum().sum() > 0:
        try:
            from scipy.stats import chi2_contingency
            
            # Create contingency table for missing patterns
            missing_patterns = missing_matrix.value_counts()
            if len(missing_patterns) > 1:
                # Chi-square test for independence of missing patterns
                contingency_table = missing_matrix.T.value_counts().unstack(fill_value=0)
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    step4.statistics["littles_test_statistic"] = round(chi2, 3)
                    step4.statistics["littles_test_pvalue"] = round(p_value, 3)
                    
                    if p_value > 0.05:
                        step4.statistics["missingness_mechanism"] = "MCAR"
                        step4.statistics["mechanism_confidence"] = "high"
                        step4.recommendations.append("Missing data appears MCAR - standard imputation methods appropriate")
                    else:
                        step4.statistics["missingness_mechanism"] = "MAR/MNAR"
                        step4.statistics["mechanism_confidence"] = "medium"
                        step4.recommendations.append("Missing data appears MAR/MNAR - consider advanced imputation methods")
                else:
                    step4.recommendations.append("Insufficient missing pattern variation for Little's test")
            else:
                step4.recommendations.append("No missing data patterns to analyze")
        except Exception as e:
            step4.recommendations.append(f"Little's test failed: {str(e)}")
    
    steps.append(step4)
    
    # Step 5: Row-wise Missing Analysis
    step5 = MissingDataStep(
        step_name="Row-wise Missing Analysis",
        description="Analyze missing patterns by rows to identify problematic records",
        formula="Row missing rate = (missing values in row) / (total columns)",
        statistics={
            "rows_with_missing": 0,
            "high_missing_rows": 0,
            "row_missing_distribution": {}
        },
        recommendations=[]
    )
    
    row_missing_counts = df.isnull().sum(axis=1)
    rows_with_missing = (row_missing_counts > 0).sum()
    high_missing_rows = (row_missing_counts > total_columns * 0.5).sum()
    
    step5.statistics["rows_with_missing"] = int(rows_with_missing)
    step5.statistics["high_missing_rows"] = int(high_missing_rows)
    step5.statistics["row_missing_distribution"] = {
        "0_missing": int((row_missing_counts == 0).sum()),
        "1-25%_missing": int(((row_missing_counts > 0) & (row_missing_counts <= total_columns * 0.25)).sum()),
        "25-50%_missing": int(((row_missing_counts > total_columns * 0.25) & (row_missing_counts <= total_columns * 0.5)).sum()),
        "50-75%_missing": int(((row_missing_counts > total_columns * 0.5) & (row_missing_counts <= total_columns * 0.75)).sum()),
        "75-100%_missing": int((row_missing_counts > total_columns * 0.75).sum())
    }
    
    if high_missing_rows > 0:
        step5.recommendations.append(f"Consider dropping {high_missing_rows} rows with >50% missing values")
    
    steps.append(step5)
    
    # Step 6: Imputation Strategy Analysis with Categorical Support
    step6 = MissingDataStep(
        step_name="Imputation Strategy Analysis",
        description="Analyze data types and patterns to recommend appropriate imputation strategies",
        formula="Numeric: x_missing = median(x_non_missing) if |skew| > 1, else mean(x_non_missing)\nCategorical: x_missing = mode(x_non_missing)",
        statistics={
            "numeric_columns": [],
            "categorical_columns": [],
            "imputation_recommendations": {},
            "imputation_values": {}
        },
        recommendations=[]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    step6.statistics["numeric_columns"] = numeric_cols
    step6.statistics["categorical_columns"] = categorical_cols
    
    for col in df.columns:
        if col in numeric_cols:
            missing_pct = missing_percentages[col]
            if missing_pct > 0:
                # Check for skewness to recommend median vs mean
                non_missing_vals = df[col].dropna()
                if len(non_missing_vals) > 0:
                    skewness = non_missing_vals.skew()
                    if abs(skewness) > 1:
                        imputation_value = non_missing_vals.median()
                        step6.statistics["imputation_recommendations"][col] = "median (skewed data)"
                        step6.statistics["imputation_values"][col] = round(imputation_value, 3)
                        step6.recommendations.append(f"Column '{col}': Use median imputation ({imputation_value:.3f}) - skewness: {skewness:.3f}")
                    else:
                        imputation_value = non_missing_vals.mean()
                        step6.statistics["imputation_recommendations"][col] = "mean (normal data)"
                        step6.statistics["imputation_values"][col] = round(imputation_value, 3)
                        step6.recommendations.append(f"Column '{col}': Use mean imputation ({imputation_value:.3f}) - skewness: {skewness:.3f}")
        elif col in categorical_cols:
            missing_pct = missing_percentages[col]
            if missing_pct > 0:
                non_missing_vals = df[col].dropna()
                if len(non_missing_vals) > 0:
                    mode_value = non_missing_vals.mode().iloc[0] if len(non_missing_vals.mode()) > 0 else "Unknown"
                    step6.statistics["imputation_recommendations"][col] = "mode"
                    step6.statistics["imputation_values"][col] = str(mode_value)
                    step6.recommendations.append(f"Column '{col}': Use mode imputation ('{mode_value}') for categorical data")
    
    steps.append(step6)
    
    # Step 7: Data Quality Impact Assessment
    step7 = MissingDataStep(
        step_name="Data Quality Impact Assessment",
        description="Assess the impact of missing data on data quality and analysis validity",
        formula="Data Quality Score = (1 - overall_missing_rate) * 100",
        statistics={
            "data_quality_score": round((1 - overall_missing_rate / 100) * 100, 2),
            "analysis_impact": "low" if overall_missing_rate < 5 else "medium" if overall_missing_rate < 20 else "high",
            "recommended_actions": []
        },
        recommendations=[]
    )
    
    quality_score = step7.statistics["data_quality_score"]
    impact = step7.statistics["analysis_impact"]
    
    if quality_score >= 95:
        step7.recommendations.append("Excellent data quality - proceed with analysis")
    elif quality_score >= 80:
        step7.recommendations.append("Good data quality - consider targeted imputation")
    elif quality_score >= 60:
        step7.recommendations.append("Moderate data quality - implement comprehensive imputation strategy")
    else:
        step7.recommendations.append("Poor data quality - consider data collection improvements or alternative datasets")
    
    step7.statistics["recommended_actions"] = step7.recommendations.copy()
    steps.append(step7)
    
    # Generate visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Missing data matrix
    plt.subplot(2, 2, 1)
    msno.matrix(df, figsize=(8, 4))
    plt.title("Missing Data Pattern Matrix")
    
    # Subplot 2: Missing data bar chart
    plt.subplot(2, 2, 2)
    missing_percentages.plot(kind='bar')
    plt.title("Missing Data by Column (%)")
    plt.xlabel("Columns")
    plt.ylabel("Missing Percentage")
    plt.xticks(rotation=45)
    plt.axhline(y=50, color='red', linestyle='--', label='50% threshold')
    plt.legend()
    
    # Subplot 3: Row missing distribution
    plt.subplot(2, 2, 3)
    row_missing_pcts = (row_missing_counts / total_columns * 100)
    plt.hist(row_missing_pcts, bins=20, alpha=0.7, edgecolor='black')
    plt.title("Distribution of Row Missing Percentages")
    plt.xlabel("Missing Percentage per Row")
    plt.ylabel("Number of Rows")
    plt.axvline(x=50, color='red', linestyle='--', label='50% threshold')
    plt.legend()
    
    # Subplot 4: Data quality summary
    plt.subplot(2, 2, 4)
    categories = ['Complete', 'Missing']
    values = [100 - overall_missing_rate, overall_missing_rate]
    colors = ['green', 'red']
    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f"Overall Data Completeness\nQuality Score: {quality_score}%")
    
    plt.tight_layout()
    
    img_path = REPORT_DIR / f"missing_analysis_{name}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Human checkpoint message
    human_checkpoint = f"""
=== MISSING DATA ANALYSIS CHECKPOINT ===
Dataset: {name}
Total Rows: {total_rows}, Total Columns: {total_columns}
Overall Missing Rate: {overall_missing_rate}%
Data Quality Score: {quality_score}%

Key Findings:
- {len(high_missing_cols)} columns have >50% missing data
- {rows_with_missing} rows contain missing values
- {high_missing_rows} rows have >50% missing values
- Systematic missingness: {'Yes' if step2.statistics['systematic_missingness'] else 'No'}

Visualization saved to: {img_path}

Please review the missing data patterns and approve the recommended actions before proceeding.
================================================================
"""
    
    print(human_checkpoint)
    
    return MissingDataResult(
        dataset_name=name,
        total_rows=total_rows,
        total_columns=total_columns,
        missing_summary=missing_summary,
        steps=steps,
        image_uri=f"file://{img_path.resolve()}",
        human_checkpoint=human_checkpoint
    )


@mcp.tool()
async def create_visualization(
    name: str,
    kind: str,
    x: str | None = None,
    y: str | None = None,
) -> list:
    """
    kind = histogram | boxplot | scatter | correlation | missing
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    plt.figure(figsize=(10, 6))
    
    if kind == "histogram" and x:
        if x not in df.columns:
            raise KeyError(f"Column '{x}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[x]):
            raise ValueError(f"Column '{x}' must be numeric for histogram.")
        plt.hist(df[x].dropna(), bins=30, alpha=0.7)
        plt.title(f"Histogram – {x}")
        plt.xlabel(x)
        plt.ylabel("Frequency")
        
    elif kind == "boxplot":
        if not x:
            raise ValueError("Boxplot requires 'x' parameter.")
        if x not in df.columns:
            raise KeyError(f"Column '{x}' not found in dataset.")
        
        if y and y in df.columns:
            # Boxplot with categorical x and numeric y
            if not pd.api.types.is_numeric_dtype(df[y]):
                raise ValueError(f"Column '{y}' must be numeric for boxplot.")
            sns.boxplot(data=df, x=x, y=y)
            plt.title(f"Boxplot – {y} by {x}")
        else:
            # Simple boxplot of single column
            if not pd.api.types.is_numeric_dtype(df[x]):
                raise ValueError(f"Column '{x}' must be numeric for single-column boxplot.")
            plt.boxplot(df[x].dropna())
            plt.title(f"Boxplot – {x}")
            plt.ylabel(x)
            
    elif kind == "scatter" and x and y:
        if x not in df.columns or y not in df.columns:
            raise KeyError(f"Columns '{x}' and/or '{y}' not found in dataset.")
        if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError(f"Both columns '{x}' and '{y}' must be numeric for scatter plot.")
        plt.scatter(df[x], df[y], alpha=0.6)
        plt.title(f"Scatter – {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        
    elif kind == "correlation":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation matrix.")
        corr = numeric_df.corr()
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Correlation Matrix")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns)
        
        # Add correlation coefficient labels
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    
    elif kind == "missing":
        msno.matrix(df)
        plt.title(f"Missing Data Pattern – {name}")
        
    else:
        raise ValueError(f"Invalid visualization type '{kind}' or missing required parameters.")

    plt.tight_layout()
    img_b64 = _fig_to_base64_png()
    plt.close()
    
    return [
        TextContent(type="text", text=f"{kind.capitalize()} plot created."),
        ImageContent(type="image", data=img_b64, mimeType="image/png"),
    ]


@mcp.tool()
async def statistical_summary(name: str) -> str:
    """
    Return describe() and numeric correlation for a dataset.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    num = df.select_dtypes(np.number)
    if num.empty:
        return "No numeric columns."

    # Sample large datasets for correlation to avoid O(n²) cost
    sample_size = 10000
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        num_sample = df_sample.select_dtypes(np.number)
        correlation_note = f"\n\nNote: Correlation matrix computed on {sample_size:,} sample rows (from {len(df):,} total rows)"
    else:
        num_sample = num
        correlation_note = ""

    return (
        f"Descriptive statistics:\n{num.describe().to_string()}\n\n"
        f"Correlation matrix:{correlation_note}\n{num_sample.corr().to_string()}"
    )


@mcp.tool()
async def list_datasets() -> str:
    """List names and shapes of all datasets in memory."""
    async with store_lock:
        if not data_store:
            return "No datasets loaded."
        lines = [
            f"- {name}: {df.shape[0]} rows × {df.shape[1]} columns"
            for name, df in data_store.items()
        ]
    return "Datasets:\n" + "\n".join(lines)


@mcp.tool()
async def infer_schema(name: str) -> SchemaResult:
    """
    Infer column types, nullability, numeric ranges, patterns (email, URL, phone, date),
    ID heuristics (unique≥0.9n), precision/scale inference, and YAML contract checks.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    columns = []
    
    for col_name in df.columns:
        col_data = df[col_name]
        
        # Step 1: Basic Type Inference
        col_type = "unknown"
        nullable = col_data.isnull().any()
        
        # Step 2: Numeric Type Detection with Precision/Scale
        if pd.api.types.is_numeric_dtype(col_data):
            col_type = "number"
            min_value = float(col_data.min()) if not col_data.empty else None
            max_value = float(col_data.max()) if not col_data.empty else None
            
            # Precision and scale inference for numeric columns
            if col_data.dtype in ['int64', 'int32', 'int16', 'int8']:
                precision = len(str(abs(int(max_value)))) if max_value else 0
                scale = 0
            else:
                # For float columns, infer precision and scale
                sample_values = col_data.dropna().head(100)
                if len(sample_values) > 0:
                    # Convert to string and analyze decimal places
                    str_values = sample_values.astype(str)
                    max_decimal_places = 0
                    max_digits = 0
                    
                    for val in str_values:
                        if '.' in val:
                            decimal_places = len(val.split('.')[1])
                            max_decimal_places = max(max_decimal_places, decimal_places)
                        
                        # Count total digits (excluding decimal point)
                        digits = len(val.replace('.', '').replace('-', ''))
                        max_digits = max(max_digits, digits)
                    
                    precision = max_digits
                    scale = max_decimal_places
                else:
                    precision = 10  # Default
                    scale = 2       # Default
        else:
            min_value = None
            max_value = None
            precision = None
            scale = None
        
        # Step 3: ID Column Detection (unique≥0.9n heuristic)
        unique_count = col_data.nunique()
        total_count = len(col_data)
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        
        is_id_column = False
        if uniqueness_ratio >= 0.9:
            # Additional ID heuristics
            if col_type == "number":
                # Check if values are sequential or have ID-like patterns
                if min_value == 1 and max_value == total_count:
                    is_id_column = True
                elif col_name.lower() in ['id', 'id_', '_id', 'key', 'pk', 'primary_key']:
                    is_id_column = True
                elif uniqueness_ratio >= 0.95:  # Very high uniqueness
                    is_id_column = True
            elif col_type == "string":
                # Check for UUID patterns or other ID patterns
                if col_name.lower() in ['id', 'id_', '_id', 'key', 'pk', 'primary_key', 'uuid']:
                    is_id_column = True
                elif uniqueness_ratio >= 0.95:
                    is_id_column = True
        
        # Step 3.5: Datetime Inference for String Columns
        if col_type == "string" and uniqueness_ratio > 0.5:  # High uniqueness suggests potential datetime
            try:
                # Try to parse as datetime
                sample_dates = col_data.dropna().head(100)
                if len(sample_dates) > 0:
                    parsed_dates = pd.to_datetime(sample_dates, errors='coerce')
                    success_rate = parsed_dates.notna().sum() / len(sample_dates)
                    if success_rate > 0.8:  # 80% success rate
                        col_type = "datetime"
                        pattern = "datetime"
            except:
                pass  # If datetime parsing fails, keep as string
        
        # Step 4: Enhanced Pattern Detection
        pattern = None
        if col_type == "string" or pd.api.types.is_string_dtype(col_data):
            str_data = col_data.astype(str)
            
            # Email pattern with improved regex
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            email_matches = str_data.str.match(email_pattern, na=False)
            if email_matches.sum() > len(str_data) * 0.8:
                pattern = "email"
            
            # URL pattern with protocol detection
            elif str_data.str.contains(r'^https?://', na=False).sum() > len(str_data) * 0.8:
                pattern = "url"
            
            # Phone pattern with international support
            elif str_data.str.match(r'^[\d\s\-\(\)\+\.]+$', na=False).sum() > len(str_data) * 0.8:
                pattern = "phone"
            
            # Date patterns (multiple formats)
            elif (str_data.str.match(r'^\d{4}-\d{2}-\d{2}$', na=False).sum() > len(str_data) * 0.8 or
                  str_data.str.match(r'^\d{2}/\d{2}/\d{4}$', na=False).sum() > len(str_data) * 0.8 or
                  str_data.str.match(r'^\d{2}-\d{2}-\d{4}$', na=False).sum() > len(str_data) * 0.8):
                pattern = "date"
            
            # Credit card pattern
            elif str_data.str.match(r'\b(?:\d[ -]*?){13,16}\b', na=False).sum() > len(str_data) * 0.8:
                pattern = "credit_card"
            
            # Postal code pattern (US ZIP)
            elif str_data.str.match(r'^\d{5}(?:-\d{4})?$', na=False).sum() > len(str_data) * 0.8:
                pattern = "postal_code"
            
            # IP address pattern
            elif str_data.str.match(r'^(?:\d{1,3}\.){3}\d{1,3}$', na=False).sum() > len(str_data) * 0.8:
                pattern = "ip_address"
            
            # UUID pattern detection
            elif str_data.str.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$', na=False).sum() > len(str_data) * 0.8:
                pattern = "uuid"
        
        # Step 5: String Length Analysis
        max_length = None
        if col_type == "string" or pd.api.types.is_string_dtype(col_data):
            str_lengths = col_data.astype(str).str.len()
            max_length = int(str_lengths.max()) if not str_lengths.empty else None
        
        # Step 6: Sample Values (with smart sampling)
        sample_values = []
        if not col_data.empty:
            # For ID columns, show range
            if is_id_column and col_type == "number":
                sample_values = [f"Range: {min_value} to {max_value}"]
            else:
                # For categorical data, show most common values
                if col_data.nunique() <= 10:
                    sample_values = col_data.value_counts().head(5).index.tolist()
                else:
                    # For high-cardinality data, show diverse samples
                    sample_values = col_data.dropna().sample(min(5, len(col_data))).tolist()
        
        # Step 7: Data Quality Indicators
        data_quality = {
            "missing_percentage": round((col_data.isnull().sum() / len(col_data)) * 100, 2),
            "duplicate_percentage": round(((len(col_data) - col_data.nunique()) / len(col_data)) * 100, 2),
            "is_id_column": is_id_column,
            "uniqueness_ratio": round(uniqueness_ratio, 3)
        }
        
        # Step 8: Enhanced YAML Contract Generation
        yaml_contract = {
            "name": col_name,
            "type": col_type,
            "nullable": nullable,
            "constraints": {}
        }
        
        if min_value is not None and max_value is not None:
            yaml_contract["constraints"]["range"] = {"min": min_value, "max": max_value}
        
        if precision is not None and scale is not None:
            yaml_contract["constraints"]["precision"] = precision
            yaml_contract["constraints"]["scale"] = scale
        
        if max_length is not None:
            yaml_contract["constraints"]["max_length"] = max_length
        
        if pattern:
            yaml_contract["constraints"]["pattern"] = pattern
        
        if is_id_column:
            yaml_contract["constraints"]["unique"] = True
            yaml_contract["constraints"]["identifier"] = True
        
        # Add data quality constraints
        if data_quality["missing_percentage"] > 0:
            yaml_contract["constraints"]["missing_percentage"] = data_quality["missing_percentage"]
        
        if data_quality["duplicate_percentage"] > 0:
            yaml_contract["constraints"]["duplicate_percentage"] = data_quality["duplicate_percentage"]
        
        # Step 9: Enhanced Recommendations
        recommendations = []
        
        if is_id_column:
            recommendations.append("High uniqueness detected - likely ID column")
            if col_type == "number":
                recommendations.append("Consider using integer type for ID columns")
            recommendations.append("Verify ID column is not used in analysis (remove from features)")
        
        if data_quality["missing_percentage"] > 20:
            recommendations.append(f"High missing data ({data_quality['missing_percentage']}%) - investigate data collection")
        
        if data_quality["duplicate_percentage"] > 10:
            recommendations.append(f"High duplicate rate ({data_quality['duplicate_percentage']}%) - consider deduplication")
        
        if pattern == "email":
            recommendations.append("Email pattern detected - validate email format")
        elif pattern == "phone":
            recommendations.append("Phone pattern detected - consider standardization")
        elif pattern == "date":
            recommendations.append("Date pattern detected - ensure consistent format")
        elif pattern == "credit_card":
            recommendations.append("Credit card pattern detected - ensure PCI compliance")
        elif pattern == "postal_code":
            recommendations.append("Postal code pattern detected - validate geographic consistency")
        elif pattern == "ip_address":
            recommendations.append("IP address pattern detected - consider geolocation analysis")
        elif pattern == "uuid":
            recommendations.append("UUID pattern detected - consider data integrity checks")
        
        # Create enhanced schema column
        column = SchemaColumn(
            name=col_name,
            type=col_type,
            nullable=nullable,
            min_value=min_value,
            max_value=max_value,
            pattern=pattern,
            max_length=max_length,
            unique_count=unique_count,
            sample_values=sample_values
        )
        
        # Add additional metadata as attributes
        column.data_quality = data_quality
        column.is_id_column = is_id_column
        column.precision = precision
        column.scale = scale
        column.uniqueness_ratio = uniqueness_ratio
        column.yaml_contract = yaml_contract
        column.recommendations = recommendations
        
        columns.append(column)
    
    # Step 10: Dataset-level Analysis
    dataset_stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "id_columns": [col.name for col in columns if getattr(col, 'is_id_column', False)],
        "high_cardinality_columns": [col.name for col in columns if getattr(col, 'uniqueness_ratio', 0) > 0.8],
        "columns_with_missing": [col.name for col in columns if col.nullable],
        "pattern_columns": [col.name for col in columns if col.pattern],
        "data_quality_score": round((1 - sum(col.data_quality["missing_percentage"] for col in columns) / len(columns) / 100) * 100, 2)
    }
    
    # Generate YAML schema contract
    yaml_schema = {
        "dataset_name": name,
        "version": "1.0",
        "created_at": pd.Timestamp.now().isoformat(),
        "statistics": dataset_stats,
        "columns": [col.yaml_contract for col in columns]
    }
    
    # Save YAML contract
    yaml_path = REPORT_DIR / f"schema_contract_{name}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_schema, f, default_flow_style=False, sort_keys=False)
    
    # Human checkpoint message
    human_checkpoint = f"""
=== SCHEMA INFERENCE CHECKPOINT ===
Dataset: {name}
Shape: {dataset_stats['total_rows']} rows × {dataset_stats['total_columns']} columns
Data Quality Score: {dataset_stats['data_quality_score']}%

Key Findings:
- ID Columns: {len(dataset_stats['id_columns'])} ({', '.join(dataset_stats['id_columns']) if dataset_stats['id_columns'] else 'None'})
- High Cardinality: {len(dataset_stats['high_cardinality_columns'])} columns
- Pattern Detection: {len(dataset_stats['pattern_columns'])} columns with patterns
- Missing Data: {len(dataset_stats['columns_with_missing'])} columns have missing values

YAML Schema Contract saved to: {yaml_path}

Please review the inferred schema and approve before proceeding.
========================================
"""
    
    print(human_checkpoint)
    
    return SchemaResult(
        dataset_name=name,
        columns=columns,
        total_rows=dataset_stats['total_rows'],
        total_columns=dataset_stats['total_columns']
    )


@mcp.tool()
async def detect_outliers(
    name: str,
    method: Literal["iqr", "isolation_forest", "lof"] = "iqr",
    factor: float = 1.5,
    contamination: float = 0.05,
    sample_size: int = 10_000
) -> OutlierResult:
    """
    Detect outliers using concrete IQR bounds with equations, model-based scores 
    correlated per feature, dual KDE+boxplot visuals, and human checkpoints.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    # Step 1: Identify numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns to analyze")

    outliers = {}
    total_rows = len(df)
    steps = []

    # Step 2: Data Distribution Analysis
    step2 = OutlierStep(
        step_name="Data Distribution Analysis",
        description="Analyze distribution characteristics of numeric columns to understand data shape",
        formula="Skewness = E[(X-μ)³/σ³], Kurtosis = E[(X-μ)⁴/σ⁴] - 3",
        parameters={"columns_analyzed": len(num_cols)},
        statistics={},
        recommendations=[]
    )
    
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            step2.statistics[col] = {
                "mean": round(col_data.mean(), 3),
                "std": round(col_data.std(), 3),
                "skewness": round(col_data.skew(), 3),
                "kurtosis": round(col_data.kurtosis(), 3),
                "q1": round(col_data.quantile(0.25), 3),
                "q3": round(col_data.quantile(0.75), 3),
                "iqr": round(col_data.quantile(0.75) - col_data.quantile(0.25), 3)
            }
            
            # Check for extreme skewness or kurtosis
            skew = abs(col_data.skew())
            kurt = abs(col_data.kurtosis())
            if skew > 2:
                step2.recommendations.append(f"Column '{col}': Highly skewed ({skew:.3f}) - consider transformation")
            if kurt > 7:
                step2.recommendations.append(f"Column '{col}': Heavy-tailed ({kurt:.3f}) - expect many outliers")
    
    steps.append(step2)

    # Step 3: Outlier Detection by Method
    if method == "iqr":
        step3 = OutlierStep(
            step_name="IQR Outlier Detection",
            description="Detect outliers using Interquartile Range method with explicit bounds",
            formula="Lower Bound = Q1 - factor × IQR\nUpper Bound = Q3 + factor × IQR\nOutlier = x < Lower_Bound OR x > Upper_Bound",
            parameters={"factor": factor, "method": "iqr"},
            statistics={},
            recommendations=[]
        )
        
        for col in num_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1, Q3 = col_data.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                outliers[col] = outlier_indices
                
                step3.statistics[col] = {
                    "q1": round(Q1, 3),
                    "q3": round(Q3, 3),
                    "iqr": round(IQR, 3),
                    "lower_bound": round(lower_bound, 3),
                    "upper_bound": round(upper_bound, 3),
                    "outlier_count": len(outlier_indices),
                    "outlier_percentage": round(len(outlier_indices) / len(col_data) * 100, 2)
                }
                
                # Recommendations based on outlier percentage
                outlier_pct = len(outlier_indices) / len(col_data) * 100
                if outlier_pct > 10:
                    step3.recommendations.append(f"Column '{col}': High outlier rate ({outlier_pct:.1f}%) - investigate data quality")
                elif outlier_pct > 5:
                    step3.recommendations.append(f"Column '{col}': Moderate outlier rate ({outlier_pct:.1f}%) - consider robust methods")
        
        steps.append(step3)

    elif method in ["isolation_forest", "lof"]:
        if not PYOD_AVAILABLE:
            raise ValueError(f"Model-based outlier detection requires pyod. Install with: pip install pyod")
        
        step3 = OutlierStep(
            step_name=f"{method.upper()} Model-based Outlier Detection",
            description=f"Detect outliers using {method} algorithm with contamination parameter",
            formula="Isolation Forest: anomaly_score = path_length / avg_path_length\nLOF: local_outlier_factor = avg_reachability_ratio",
            parameters={"contamination": contamination, "method": method},
            statistics={},
            recommendations=[]
        )
        
        try:
            if method == "isolation_forest":
                model = IsolationForest(contamination=contamination, random_state=42)
                step3.formula = "Isolation Forest: anomaly_score = 2^(-avg_path_length / c(n))\nwhere c(n) = 2H(n-1) - 2(n-1)/n"
            else:  # lof
                model = LOF(n_neighbors=20, contamination=contamination)
                step3.formula = "LOF(x) = avg(reachability_dist(x,o)) / avg(reachability_dist(o,neighbors(o)))"
            
            # Fit and predict on numeric columns
            model.fit(df[num_cols])
            predictions = model.predict(df[num_cols])
            scores = model.decision_scores_ if hasattr(model, 'decision_scores_') else None
            
            # Get outlier indices (-1 = outlier)
            outlier_indices = np.where(predictions == -1)[0].tolist()
            
            # Attribute row-level flags to each column
            for col in num_cols:
                outliers[col] = outlier_indices
                
                # Calculate column-specific statistics
                col_data = df[col].dropna()
                outlier_values = df.loc[outlier_indices, col] if outlier_indices else pd.Series()
                
                step3.statistics[col] = {
                    "outlier_count": len(outlier_indices),
                    "outlier_percentage": round(len(outlier_indices) / len(col_data) * 100, 2) if len(col_data) > 0 else 0,
                    "contamination_used": contamination,
                    "model_type": method
                }
                
                if scores is not None:
                    step3.statistics[col]["avg_anomaly_score"] = round(np.mean(scores), 3)
                    step3.statistics[col]["max_anomaly_score"] = round(np.max(scores), 3)
                
                # Recommendations
                outlier_pct = len(outlier_indices) / len(col_data) * 100 if len(col_data) > 0 else 0
                if outlier_pct > contamination * 100 * 2:
                    step3.recommendations.append(f"Column '{col}': High outlier rate ({outlier_pct:.1f}%) - consider adjusting contamination")
                elif outlier_pct < contamination * 100 * 0.5:
                    step3.recommendations.append(f"Column '{col}': Low outlier rate ({outlier_pct:.1f}%) - model may be too conservative")
                    
        except Exception as e:
            step3.recommendations.append(f"Model-based detection failed: {str(e)}")
            # Return empty outlier sets
            for col in num_cols:
                outliers[col] = []
        
        steps.append(step3)

    # Step 4: Aggregate counts and create summary
    counts = {col: len(outliers[col]) for col in num_cols}
    total_outliers = sum(counts.values())
    
    step4 = OutlierStep(
        step_name="Outlier Summary and Impact Assessment",
        description="Summarize outlier detection results and assess impact on analysis",
        formula="Total Outliers = Σ(outliers_per_column)\nImpact Score = (total_outliers / total_cells) × 100",
        parameters={"total_columns": len(num_cols), "total_rows": total_rows},
        statistics={
            "total_outliers": total_outliers,
            "outliers_by_column": counts,
            "impact_score": round(total_outliers / (total_rows * len(num_cols)) * 100, 3),
            "columns_with_outliers": len([col for col, count in counts.items() if count > 0])
        },
        recommendations=[]
    )
    
    impact_score = total_outliers / (total_rows * len(num_cols)) * 100
    if impact_score > 5:
        step4.recommendations.append("High outlier impact - consider robust statistical methods")
        step4.recommendations.append("Investigate outlier causes and data collection process")
    elif impact_score > 2:
        step4.recommendations.append("Moderate outlier impact - use outlier-resistant methods")
    else:
        step4.recommendations.append("Low outlier impact - standard methods should work well")
    
    steps.append(step4)

    # Step 5: Generate dual visualization (Histogram + KDE + Boxplot)
    plt.figure(figsize=(15, 5 * len(num_cols)))
    
    for i, col in enumerate(num_cols):
        # Subplot 1: Histogram + KDE plot with outlier highlighting
        plt.subplot(len(num_cols), 2, 2*i + 1)
        col_data = df[col].dropna()
        outlier_data = df.loc[outliers[col], col] if outliers[col] else pd.Series()
        
        # Histogram for main distribution
        plt.hist(col_data, bins=30, density=True, alpha=0.7, color='blue', label='All Data')
        
        # Overlay KDE
        from scipy.stats import gaussian_kde
        if len(col_data) > 1:
            kde = gaussian_kde(col_data)
            x_range = np.linspace(col_data.min(), col_data.max(), 100)
            plt.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        
        # Highlight outliers
        if len(outlier_data) > 0:
            plt.hist(outlier_data, bins=10, density=True, alpha=0.8, color='red', label=f'Outliers ({len(outlier_data)})')
        
        plt.title(f"{col} - Distribution with Outliers")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        
        # Subplot 2: Boxplot with outlier points
        plt.subplot(len(num_cols), 2, 2*i + 2)
        bp = plt.boxplot(col_data, vert=False, showfliers=False)
        
        # Overlay outlier points
        if len(outlier_data) > 0:
            plt.scatter(outlier_data, [1] * len(outlier_data), 
                       color="red", alpha=0.6, s=20, label=f"Outliers ({len(outlier_data)})")
        
        plt.title(f"{col} - Boxplot with Outliers")
        plt.xlabel(col)
        plt.legend()
    
    plt.tight_layout()
    
    # Save to PNG
    img_path = REPORT_DIR / f"outliers_{name}_{method}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Step 6: Mahalanobis Distance Analysis (Multivariate Outliers)
    if len(num_cols) > 1 and method == "iqr":
        step6 = OutlierStep(
            step_name="Mahalanobis Distance Analysis",
            description="Detect multivariate outliers using Mahalanobis distance",
            formula="Mahalanobis Distance: d²(x,μ) = (x-μ)ᵀΣ⁻¹(x-μ)\nOutlier if d² > χ²_p(0.975) for p dimensions",
            parameters={"columns_analyzed": len(num_cols)},
            statistics={
                "mahalanobis_outliers": [],
                "mahalanobis_threshold": None,
                "multivariate_outlier_count": 0
            },
            recommendations=[]
        )
        
        try:
            from scipy.stats import chi2
            
            # Calculate Mahalanobis distance
            numeric_data = df[num_cols].dropna()
            if len(numeric_data) > len(num_cols):
                # Calculate covariance matrix and mean
                cov_matrix = numeric_data.cov()
                mean_vector = numeric_data.mean()
                
                # Calculate Mahalanobis distance for each point
                mahal_distances = []
                for idx, row in numeric_data.iterrows():
                    diff = row - mean_vector
                    mahal_dist = diff.T @ np.linalg.inv(cov_matrix) @ diff  # This is already squared
                    mahal_distances.append(mahal_dist)
                
                # Set threshold based on chi-square distribution
                p = len(num_cols)
                threshold = chi2.ppf(0.975, p)
                
                # Find multivariate outliers
                mahal_outliers = numeric_data[mahal_distances > threshold].index.tolist()
                
                step6.statistics["mahalanobis_threshold"] = round(threshold, 3)
                step6.statistics["multivariate_outlier_count"] = len(mahal_outliers)
                step6.statistics["mahalanobis_outliers"] = mahal_outliers
                
                if len(mahal_outliers) > 0:
                    step6.recommendations.append(f"Found {len(mahal_outliers)} multivariate outliers using Mahalanobis distance")
                    step6.recommendations.append("Consider these in addition to univariate outliers for comprehensive analysis")
                else:
                    step6.recommendations.append("No multivariate outliers detected")
                    
        except Exception as e:
            step6.recommendations.append(f"Mahalanobis analysis failed: {str(e)}")
        
        steps.append(step6)

    # Human checkpoint message
    human_checkpoint = f"""
=== OUTLIER DETECTION CHECKPOINT ===
Dataset: {name}
Method: {method.upper()}
Total Rows: {total_rows}
Numeric Columns: {len(num_cols)}

Detection Results:
- Total Outliers: {total_outliers}
- Impact Score: {step4.statistics['impact_score']}%
- Columns with Outliers: {step4.statistics['columns_with_outliers']}

Key Statistics:
"""
    
    for col in num_cols:
        if col in step3.statistics:
            stats = step3.statistics[col]
            human_checkpoint += f"- {col}: {stats.get('outlier_count', 0)} outliers ({stats.get('outlier_percentage', 0):.1f}%)\n"
    
    human_checkpoint += f"""
Visualization saved to: {img_path}

Please review the outlier patterns and approve the recommended actions before proceeding.
========================================
"""
    
    print(human_checkpoint)

    return OutlierResult(
        result=OutlierPayload(
            outliers=outliers,
            counts=counts,
            total_rows=total_rows,
            steps=steps,
            method_used=method
        ),
        image_uri=f"file://{img_path.resolve()}",
        human_checkpoint=human_checkpoint
    )


@mcp.tool()
async def feature_transformation(
    name: str,
    transformations: list[str] = ["boxcox", "log", "binning", "cardinality"],
    target_col: str | None = None
) -> TransformationResult:
    """
    Apply feature transformations: Box-Cox (x′=(x^λ−1)/λ), log shifts, 
    quantile binning, cardinality reduction with detailed formulas and checkpoints.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")
    
    original_shape = df.shape
    transformed_df = df.copy()
    steps = []
    
    # Step 1: Data Type Analysis
    step1 = TransformationStep(
        step_name="Data Type Analysis",
        description="Analyze data types to determine appropriate transformations",
        formula="Numeric columns: apply mathematical transformations\nCategorical columns: apply encoding transformations",
        parameters={"transformations_requested": transformations},
        statistics={},
        recommendations=[]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)  # Don't transform target column
    
    step1.statistics = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "total_numeric": len(numeric_cols),
        "total_categorical": len(categorical_cols)
    }
    
    if len(numeric_cols) == 0 and ("boxcox" in transformations or "log" in transformations):
        step1.recommendations.append("No numeric columns found for mathematical transformations")
    
    if len(categorical_cols) == 0 and "cardinality" in transformations:
        step1.recommendations.append("No categorical columns found for cardinality reduction")
    
    steps.append(step1)
    
    # Step 2: Box-Cox Transformation
    if "boxcox" in transformations and len(numeric_cols) > 0:
        step2 = TransformationStep(
            step_name="Box-Cox Transformation",
            description="Apply Box-Cox transformation to normalize skewed numeric data",
            formula="Box-Cox: x′ = (x^λ - 1) / λ if λ ≠ 0, else x′ = ln(x)\nOptimal λ found by maximizing log-likelihood",
            parameters={"columns_transformed": []},
            statistics={},
            recommendations=[]
        )
        
        from scipy.stats import boxcox
        
        for col in numeric_cols:
            col_data = transformed_df[col].dropna()
            if len(col_data) > 0:
                # Check if transformation is needed (skewness > 1)
                original_skew = col_data.skew()
                if abs(original_skew) > 1:
                    # Handle near-zero values by shifting
                    min_val = col_data.min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1e-6  # Small epsilon to avoid exact zero
                        col_data_shifted = col_data + shift
                    else:
                        shift = 0
                        col_data_shifted = col_data
                    
                    try:
                        # Apply Box-Cox transformation
                        transformed_data, lambda_param = boxcox(col_data_shifted)
                        
                        # Create new column name
                        new_col_name = f"{col}_boxcox"
                        transformed_df[new_col_name] = transformed_data
                        
                        # Calculate statistics
                        new_skew = transformed_df[new_col_name].skew()
                        improvement = abs(original_skew) - abs(new_skew)
                        
                        step2.parameters["columns_transformed"].append(col)
                        step2.statistics[col] = {
                            "original_skewness": round(original_skew, 3),
                            "transformed_skewness": round(new_skew, 3),
                            "lambda_parameter": round(lambda_param, 3),
                            "skewness_improvement": round(improvement, 3),
                            "shift_applied": round(shift, 6),
                            "new_column": new_col_name
                        }
                        
                        if improvement > 0.5:
                            step2.recommendations.append(f"Column '{col}': Significant improvement (skewness: {original_skew:.3f} → {new_skew:.3f})")
                        else:
                            step2.recommendations.append(f"Column '{col}': Minimal improvement - consider alternative transformation")
                            
                    except Exception as e:
                        step2.recommendations.append(f"Column '{col}': Box-Cox failed - {str(e)}")
                else:
                    step2.recommendations.append(f"Column '{col}': Not skewed enough (skewness: {original_skew:.3f}) - skipping")
            else:
                step2.recommendations.append(f"Column '{col}': No data available for transformation")
        
        steps.append(step2)
    
    # Step 3: Log Transformation
    if "log" in transformations and len(numeric_cols) > 0:
        step3 = TransformationStep(
            step_name="Log Transformation",
            description="Apply log transformation to handle right-skewed data",
            formula="Log transform: x′ = ln(x + shift)\nwhere shift = 1 - min(x) if min(x) ≤ 0",
            parameters={"columns_transformed": []},
            statistics={},
            recommendations=[]
        )
        
        for col in numeric_cols:
            col_data = transformed_df[col].dropna()
            if len(col_data) > 0:
                original_skew = col_data.skew()
                
                # Only apply if right-skewed
                if original_skew > 1:
                    # Calculate shift if needed
                    shift = 0
                    if col_data.min() <= 0:
                        shift = 1 - col_data.min()
                    
                    # Apply log transformation
                    transformed_data = np.log(col_data + shift)
                    
                    # Create new column name
                    new_col_name = f"{col}_log"
                    transformed_df[new_col_name] = transformed_data
                    
                    # Calculate statistics
                    new_skew = transformed_df[new_col_name].skew()
                    improvement = original_skew - new_skew
                    
                    step3.parameters["columns_transformed"].append(col)
                    step3.statistics[col] = {
                        "original_skewness": round(original_skew, 3),
                        "transformed_skewness": round(new_skew, 3),
                        "shift_applied": round(shift, 3),
                        "skewness_improvement": round(improvement, 3),
                        "new_column": new_col_name
                    }
                    
                    if improvement > 0.5:
                        step3.recommendations.append(f"Column '{col}': Good improvement (skewness: {original_skew:.3f} → {new_skew:.3f})")
                    else:
                        step3.recommendations.append(f"Column '{col}': Limited improvement - consider other transformations")
                else:
                    step3.recommendations.append(f"Column '{col}': Not right-skewed (skewness: {original_skew:.3f}) - skipping")
        
        steps.append(step3)
    
    # Step 4: Quantile Binning
    if "binning" in transformations and len(numeric_cols) > 0:
        step4 = TransformationStep(
            step_name="Quantile Binning",
            description="Convert numeric variables to categorical using quantile-based binning",
            formula="Quantile binning: bin_i = Q(i/n_bins)\nwhere Q is the quantile function and n_bins is the number of bins",
            parameters={"n_bins": 5, "columns_transformed": []},
            statistics={},
            recommendations=[]
        )
        
        n_bins = 5
        for col in numeric_cols:
            col_data = transformed_df[col].dropna()
            if len(col_data) > n_bins:
                # Create quantile-based bins
                bins = pd.qcut(col_data, q=n_bins, labels=[f"Q{i+1}" for i in range(n_bins)], duplicates='drop')
                
                # Create new column name
                new_col_name = f"{col}_binned"
                transformed_df[new_col_name] = bins
                
                # Calculate statistics
                bin_counts = bins.value_counts()
                bin_distribution = (bin_counts / len(bins) * 100).round(2)
                
                step4.parameters["columns_transformed"].append(col)
                step4.statistics[col] = {
                    "n_bins_created": len(bin_counts),
                    "bin_distribution": bin_distribution.to_dict(),
                    "new_column": new_col_name
                }
                
                # Check for balanced bins
                max_bin_pct = bin_distribution.max()
                min_bin_pct = bin_distribution.min()
                balance_ratio = max_bin_pct / min_bin_pct if min_bin_pct > 0 else float('inf')
                
                if balance_ratio < 2:
                    step4.recommendations.append(f"Column '{col}': Well-balanced bins (ratio: {balance_ratio:.2f})")
                else:
                    step4.recommendations.append(f"Column '{col}': Unbalanced bins (ratio: {balance_ratio:.2f}) - consider different binning strategy")
            else:
                step4.recommendations.append(f"Column '{col}': Insufficient data for {n_bins} bins")
        
        steps.append(step4)
    
    # Step 5: Cardinality Reduction
    if "cardinality" in transformations and len(categorical_cols) > 0:
        step5 = TransformationStep(
            step_name="Cardinality Reduction",
            description="Reduce high-cardinality categorical variables by grouping rare categories",
            formula="Cardinality reduction: group categories with frequency < threshold\nNew category = 'Other' if freq < min_freq",
            parameters={"min_frequency": 0.005, "columns_transformed": []},  # 0.5% threshold
            statistics={},
            recommendations=[]
        )
        
        min_freq = 0.005  # 0.5% threshold per book recommendation
        for col in categorical_cols:
            col_data = transformed_df[col].dropna()
            if len(col_data) > 0:
                # Calculate value counts and frequencies
                value_counts = col_data.value_counts()
                frequencies = value_counts / len(col_data)
                
                # Identify rare categories
                rare_categories = frequencies[frequencies < min_freq].index.tolist()
                rare_count = len(rare_categories)
                rare_freq = frequencies[frequencies < min_freq].sum()
                
                if rare_count > 0:
                    # Create new column with reduced cardinality
                    new_col_name = f"{col}_reduced"
                    reduced_data = col_data.copy()
                    
                    # Replace rare categories with 'Other'
                    reduced_data = reduced_data.replace(rare_categories, 'Other')
                    
                    transformed_df[new_col_name] = reduced_data
                    
                    # Calculate statistics
                    new_value_counts = reduced_data.value_counts()
                    new_cardinality = len(new_value_counts)
                    original_cardinality = len(value_counts)
                    reduction = original_cardinality - new_cardinality
                    
                    step5.parameters["columns_transformed"].append(col)
                    step5.statistics[col] = {
                        "original_cardinality": original_cardinality,
                        "new_cardinality": new_cardinality,
                        "categories_reduced": reduction,
                        "rare_categories_found": rare_count,
                        "rare_frequency_total": round(rare_freq, 3),
                        "new_column": new_col_name
                    }
                    
                    if reduction > 0:
                        step5.recommendations.append(f"Column '{col}': Reduced from {original_cardinality} to {new_cardinality} categories")
                        if rare_freq < 0.1:
                            step5.recommendations.append(f"Column '{col}': Low information loss ({rare_freq:.1%} of data)")
                        else:
                            step5.recommendations.append(f"Column '{col}': Moderate information loss ({rare_freq:.1%} of data) - review threshold")
                    else:
                        step5.recommendations.append(f"Column '{col}': No reduction needed - all categories above threshold")
                else:
                    step5.recommendations.append(f"Column '{col}': No rare categories found - cardinality reduction not needed")
        
        steps.append(step5)
    
    # Step 6: Transformation Summary
    transformed_shape = transformed_df.shape
    new_columns = [col for col in transformed_df.columns if col not in df.columns]
    
    step6 = TransformationStep(
        step_name="Transformation Summary",
        description="Summarize all transformations applied and their impact",
        formula="Shape change: (rows, cols) → (rows, cols + new_features)\nFeature increase = new_columns_count",
        parameters={"total_transformations": len(steps) - 1},
        statistics={
            "original_shape": original_shape,
            "transformed_shape": transformed_shape,
            "new_columns_created": len(new_columns),
            "new_columns": new_columns,
            "shape_change": f"{original_shape} → {transformed_shape}",
            "post_transform_skewness": {}
        },
        recommendations=[]
    )
    
    # Record post-transform skewness for transformed numeric columns
    for col in new_columns:
        if col.endswith('_boxcox') or col.endswith('_log'):
            original_col = col.replace('_boxcox', '').replace('_log', '')
            if original_col in numeric_cols:
                transformed_data = transformed_df[col].dropna()
                if len(transformed_data) > 0:
                    post_skew = transformed_data.skew()
                    step6.statistics["post_transform_skewness"][col] = round(post_skew, 3)
                    step6.recommendations.append(f"Column '{col}': Post-transform skewness = {post_skew:.3f}")
    
    if len(new_columns) > 0:
        step6.recommendations.append(f"Created {len(new_columns)} new features through transformations")
        step6.recommendations.append("Consider feature selection to avoid multicollinearity")
        step6.recommendations.append("Validate transformations improve model performance")
    else:
        step6.recommendations.append("No transformations applied - review transformation parameters")
    
    steps.append(step6)
    
    # Generate visualization with before/after plots
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Original vs Transformed distributions (for first numeric column)
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        original_data = df[col].dropna()
        
        # Original distribution
        plt.subplot(2, 2, 1)
        plt.hist(original_data, bins=30, alpha=0.7, label='Original', color='blue')
        plt.title(f"Original Distribution - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        
        # Show transformed version if available
        transformed_cols = [c for c in new_columns if col in c]
        if transformed_cols:
            plt.subplot(2, 2, 2)
            transformed_data = transformed_df[transformed_cols[0]].dropna()
            plt.hist(transformed_data, bins=30, alpha=0.7, label='Transformed', color='red')
            plt.title(f"Transformed Distribution - {transformed_cols[0]}")
            plt.xlabel(transformed_cols[0])
            plt.ylabel("Frequency")
            plt.legend()
            
            # Before vs After comparison
            plt.subplot(2, 2, 3)
            plt.scatter(original_data, transformed_data, alpha=0.6)
            plt.plot([original_data.min(), original_data.max()], [transformed_data.min(), transformed_data.max()], 'r--', label='y=x')
            plt.title(f"Before vs After Transformation - {col}")
            plt.xlabel(f"Original {col}")
            plt.ylabel(f"Transformed {col}")
            plt.legend()
    
    # Subplot 4: Transformation summary
    plt.subplot(2, 2, 4)
    transformation_types = [step.step_name for step in steps[1:-1]]  # Exclude analysis and summary steps
    if transformation_types:
        plt.bar(range(len(transformation_types)), [1] * len(transformation_types))
        plt.title("Transformations Applied")
        plt.xlabel("Transformation Type")
        plt.ylabel("Applied (1) / Not Applied (0)")
        plt.xticks(range(len(transformation_types)), transformation_types, rotation=45)
    
    plt.tight_layout()
    
    img_path = REPORT_DIR / f"feature_transformation_{name}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Human checkpoint message
    human_checkpoint = f"""
=== FEATURE TRANSFORMATION CHECKPOINT ===
Dataset: {name}
Original Shape: {original_shape}
Transformed Shape: {transformed_shape}
New Features Created: {len(new_columns)}

Transformations Applied:
"""
    
    for step in steps[1:-1]:  # Exclude analysis and summary steps
        if step.parameters.get("columns_transformed"):
            human_checkpoint += f"- {step.step_name}: {len(step.parameters['columns_transformed'])} columns\n"
    
    human_checkpoint += f"""
New Columns: {', '.join(new_columns) if new_columns else 'None'}

Visualization saved to: {img_path}

Please review the transformations and approve before proceeding to modeling.
===============================================
"""
    
    print(human_checkpoint)
    
    return TransformationResult(
        dataset_name=name,
        original_shape=original_shape,
        transformed_shape=transformed_shape,
        steps=steps,
        image_uri=f"file://{img_path.resolve()}",
        human_checkpoint=human_checkpoint
    )


# ─── Evidently Tools ──────────────────────────────────────────────────────

class DataQualityReportResult(BaseModel):
    html_uri: str

@mcp.tool()
async def data_quality_report(
    name: str,
    title: str | None = None,
) -> DataQualityReportResult:
    """
    Run Evidently DataSummaryPreset on a single dataset.
    Returns {"html_uri": "file://..."}.
    """
    async with store_lock:
        df = data_store.get(name)
    if df is None:
        raise KeyError(f"Dataset '{name}' not found.")

    metadata = {"title": title} if title else None
    rpt = Report([DataSummaryPreset()], metadata=metadata)
    snapshot = await asyncio.to_thread(rpt.run, _ds(df))

    html = REPORT_DIR / f"dq_{name}.html"
    snapshot.save_html(html)

    # DataSummaryPreset may not have 'quality_score', so just return the HTML URI
    return DataQualityReportResult(html_uri=f"file://{html.resolve()}")


class DriftAnalysisResult(BaseModel):
    drift_count: float
    drift_share: float
    html_uri: str

@mcp.tool()
async def drift_analysis(
    baseline: str,
    current: str,
) -> DriftAnalysisResult:
    async with store_lock:
        ref = data_store.get(baseline)
        cur = data_store.get(current)
    if ref is None or cur is None:
        raise KeyError("Datasets not found")

    rpt = Report([DataDriftPreset()])
    snap = await asyncio.to_thread(rpt.run, _ds(cur), _ds(ref))

    html = REPORT_DIR / f"drift_{baseline}_vs_{current}.html"
    snap.save_html(html)

    summary = json.loads(snap.json())
    # Find the DriftedColumnsCount metric
    drift_metric = next(
        (m["value"] for m in summary["metrics"]
         if m["metric_id"].startswith("DriftedColumnsCount")),
        {"count": 0.0, "share": 0.0},
    )

    return DriftAnalysisResult(
        drift_count=drift_metric["count"],
        drift_share=drift_metric["share"],
        html_uri=f"file://{html.resolve()}",
    )


class ModelPerformanceReportResult(BaseModel):
    metrics: dict
    html_uri: str

@mcp.tool()
async def model_performance_report(
    y_true: list[float | int],
    y_pred: list[float | int],
    model_type: str = "classification",
) -> ModelPerformanceReportResult:

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch.")

    # First try Evidently for regression only
    if model_type.startswith("reg"):
        df = pd.DataFrame({"target": y_true, "prediction": y_pred})
        rpt = Report([RegressionPreset()])
        snap = await asyncio.to_thread(rpt.run, _ds_regression(df))
        html = REPORT_DIR / f"perf_regression_{len(df)}.html"
        snap.save_html(html)

        summary = json.loads(snap.json())
        # Extract common regression metrics: RMSE, MAE, R2
        metrics = {}
        for m in summary["metrics"]:
            mid = m["metric_id"]
            val = m["value"]
            if mid.startswith("RMSE"):
                metrics["rmse"] = val
            elif mid.startswith("MeanError"):
                metrics["mae"] = val["mean"]
            elif mid.startswith("R2"):
                metrics["r2"] = val

    else:
        # Classification → fallback to sklearn metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_weighted": f1,
        }
        # No Evidently HTML for classification fallback
        html = REPORT_DIR / f"perf_classification_{len(y_true)}.html"
        with open(html, "w") as f:
            f.write(
                "<html><body>"
                f"<h1>Classification Metrics</h1>"
                f"<ul>"
                f"<li>Accuracy: {acc:.3f}</li>"
                f"<li>Precision: {prec:.3f}</li>"
                f"<li>Recall: {rec:.3f}</li>"
                f"<li>F1 (weighted): {f1:.3f}</li>"
                "</ul>"
                "</body></html>"
            )

    return ModelPerformanceReportResult(
        metrics=metrics,
        html_uri=f"file://{html.resolve()}",
    )


# ─── DEBUG DUMP FOR SUMMARY KEYS ─────────────────────────────────────────

class DebugSummaryResult(BaseModel):
    html_uri: str
    keys: list[str]
    summary: dict

@mcp.tool()
async def debug_drift_summary(
    baseline: str,
    current: str,
) -> DebugSummaryResult:
    """
    TEMPORARY: Run DataDrift and return summary keys.
    """
    async with store_lock:
        ref = data_store.get(baseline)
        cur = data_store.get(current)
    if ref is None or cur is None:
        raise KeyError("datasets not found")

    rpt = Report([DataDriftPreset()])
    snap = await asyncio.to_thread(rpt.run, _ds(cur), _ds(ref))

    html = REPORT_DIR / "drift_debug.html"
    snap.save_html(html)

    summary = json.loads(snap.json())
    return DebugSummaryResult(
        html_uri=f"file://{html.resolve()}",
        keys=list(summary.keys()),
        summary=summary,
    )

@mcp.tool()
async def debug_perf_summary(
    y_true: list[int | float],
    y_pred: list[int | float],
    model_type: str = "classification",
) -> DebugSummaryResult:
    """
    TEMPORARY: Run PerformancePreset and return summary keys.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")

    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    preset = ClassificationPreset() if model_type.startswith("class") else RegressionPreset()
    rpt = Report([preset])
    snap = await asyncio.to_thread(rpt.run, _ds(df))

    html = REPORT_DIR / "perf_debug.html"
    snap.save_html(html)

    summary = json.loads(snap.json())
    return DebugSummaryResult(
        html_uri=f"file://{html.resolve()}",
        keys=list(summary.keys()),
        summary=summary,
    )


######################################################
# Run the server
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=10000)
