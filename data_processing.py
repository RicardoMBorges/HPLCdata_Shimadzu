# data_processing.py

import pandas as pd
import numpy as np
from scipy.signal import correlate

### USE
# import data_processing as dp

# ex.: normalized_df = dp.min_max_normalize(combined_df)

  # Ensure that the Python file data_processing.py is in the same directory as your Jupyter Notebook or in a directory that's on the Python path.
  # If you make changes to data_processing.py, you might need to reload the module in your Jupyter Notebook. 
  # You can use the %load_ext autoreload and %autoreload 2 magic commands at the start of your notebook for automatic reloading.


### REMOVE UNWANTED REGIONS
def remove_unwanted_regions(df, start_value, end_value):
    """
    Remove unwanted regions by substituting sample values with zeros between specified 
    start and end values in the RT(min) axis.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    start_value (float/int): Starting value of the RT(min) range.
    end_value (float/int): Ending value of the RT(min) range.

    Returns:
    pd.DataFrame: DataFrame with substituted values.
    
    # Example usage
    start = 2  # Starting value of RT(min) for substitution
    end = 5    # Ending value of RT(min) for substitution
    modified_df = remove_unwanted_regions(combined_df.copy(), start, end)
    """
    # Identify the rows where RT(min) is within the specified range
    rows_to_substitute = df['RT(min)'].between(start_value, end_value)

    # Columns to be modified (excluding RT(min))
    sample_columns = [col for col in df.columns if col != 'RT(min)']

    # Substitute values with zeros in the specified range for all sample columns
    df.loc[rows_to_substitute, sample_columns] = 0

    return df
##


### ALIGNMENT FUNCTIONS
def align_samples(df, reference_column):
    """
    Function to align samples to a Reference Sample
    This method assumes linear shifts and may not work well with non-linear distortions.
    The choice of reference sample can affect the results.
    The data should be appropriately scaled or normalized if necessary.
    """
    ref_signal = df[reference_column]
    max_shifts = {}
    for column in df.columns:
        if column != 'RT(min)' and column != reference_column:
            shift = np.argmax(correlate(df[column], ref_signal)) - (len(ref_signal) - 1)
            max_shifts[column] = shift
            df[column] = np.roll(df[column], -shift)
    return df, max_shifts

# Align samples
aligned_df, shifts = align_samples(combined_df, 'Sample1')
##


# Function to align samples to a median
def align_samples_to_median(df):
    """
    We calculate the median profile across all samples for each time point.
    Each sample is then aligned to this median profile using cross-correlation.
    The shifts are calculated and applied to each sample to align them.
    """
    median_profile = df.drop('RT(min)', axis=1).median(axis=1)
    max_shifts = {}
    for column in df.columns:
        if column != 'RT(min)':
            shift = np.argmax(correlate(df[column], median_profile)) - (len(median_profile) - 1)
            max_shifts[column] = shift
            df[column] = np.roll(df[column], -shift)
    return df, max_shifts

# Align samples to the median profile
aligned_df, shifts = align_samples_to_median(combined_df)
##    We calculate the standard deviation profile across all samples.


# Function to align samples using Standard Deviation
def align_samples_using_std(df):
    """
    Less common for direct alignment purposes, but it can be a useful method for identifying the degree of variability or inconsistency among your samples.
    We identify a stable region around the point of lowest standard deviation.
    We then align each sample to a reference sample (Sample1 in this case) but only focusing on this stable region.
    The shifts are then calculated and applied.
    """
    std_profile = df.drop('RT(min)', axis=1).std(axis=1)
    # Identify stable regions (low standard deviation)
    # For simplicity, let's assume we use the overall lowest std value
    stable_point = std_profile.idxmin()
    max_shifts = {}
    for column in df.columns:
        if column != 'RT(min)':
            # Align using the stable point
            stable_region = df.loc[stable_point-5:stable_point+5, column] # Adjust the range as needed
            shift = np.argmax(correlate(stable_region, df.loc[stable_point-5:stable_point+5, 'Sample1'])) - 5
            max_shifts[column] = shift
            df[column] = np.roll(df[column], -shift)
    return df, max_shifts

# Align samples using standard deviation
aligned_df, shifts = align_samples_using_std(combined_df)
##


### NORMALIZATION FUNCTIONS
    """
Min-max normalization scales the data so that it fits within a specific range, typically 0 to 1.
    """
def min_max_normalize(df):
    for column in df.columns:
        if column != 'RT(min)':
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

normalized_df = min_max_normalize(combined_df.copy())
##


# Z-score normalization.
def z_score_normalize(df):
    """
    Z-score normalization transforms the data to have a mean of 0 and a standard deviation of 1.
    """
    for column in df.columns:
        if column != 'RT(min)':
            mean_val = df[column].mean()
            std_val = df[column].std()
            df[column] = (df[column] - mean_val) / std_val
    return df

normalized_df = z_score_normalize(combined_df.copy())
##


# Normalization by a Control.
def normalize_by_control(df, control_column):
    """
    Normalization by a control is specific to experimental designs where a control feature is available.
    """
    control = df[control_column]
    for column in df.columns:
        if column != 'RT(min)' and column != control_column:
            df[column] = df[column] / control
    return df

normalized_df = normalize_by_control(combined_df.copy(), 'ControlSample')
##


### SCALING FUNCTIONS
def min_max_scale(df, new_min=0, new_max=1):
    """
    This is similar to min-max normalization but can be used to scale the data to a range different from 0 to 1.
    """
    for column in df.columns:
        if column != 'RT(min)':
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return df

scaled_df = min_max_scale(combined_df.copy(), 0, 1)  # Example range 0 to 1
##


# Standard Scaling (Z-Score Scaling)
def standard_scale(df):
    """
    Standard Scaling (Z-Score Scaling) involves scaling the data to have a mean of 0 and a standard deviation of 1, similar to Z-score normalization.
    It's ideal for algorithms that assume the data is centered around zero and has a standard deviation of one.
    """
    for column in df.columns:
        if column != 'RT(min)':
            mean_val = df[column].mean()
            std_val = df[column].std()
            df[column] = (df[column] - mean_val) / std_val
    return df

scaled_df = standard_scale(combined_df.copy())
##


# Robust scaling uses the median and the interquartile range, making it effective in cases where the data contains outliers.
def robust_scale(df):
    """
    Robust scaling uses the median and the interquartile range, making it effective in cases where the data contains outliers.
    """
    for column in df.columns:
        if column != 'RT(min)':
            median_val = df[column].median()
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df[column] = (df[column] - median_val) / IQR
    return df

scaled_df = robust_scale(combined_df.copy())
##
