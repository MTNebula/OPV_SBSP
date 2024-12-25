#!/usr/bin/env python3
# %% loading in things

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import logging
import warnings
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns

import pandas as pd

import copy
from itertools import chain

from base_component import BaseComponent

colors = sns.color_palette("viridis", 50)
sns.set_palette("viridis")




from config import (NUM_OF_TOTAL_YEARS, NUM_OF_ITERATIONS, DISCOUNT_RATE,
                    DISCOUNT_RATE_SD, EXCEL_PATH, SHEET_NAME, SHEET_NAMES,
                    values_of_scenarios)




def extract_base_components(excel_path: str, sheet_name: str):
    """
    Extract base components from an Excel file.

    Args:
        excel_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to read.

    Returns:
        dict: Dictionary of base components.
    """

    data = pd.read_excel(excel_path, sheet_name, skiprows=2)  # Skip the first two rows

    # Filter relevant columns and rows
    data = data.rename(columns={
        'Unnamed: 0': 'Include',
        'Unnamed: 1': 'Primary',
        'Unnamed: 2': 'Secondary',
        'Unnamed: 3': 'Tertiary',
        'Unnamed: 4': 'Quaternary',
        'Unnamed: 5': 'Quinary',
        'Unnamed: 6': 'Units',
        'Unnamed: 7': 'Distribution',
        'Unnamed: 8': 'Time for determination (Year)',
        'Unnamed: 9': 'Lower Limit',  # Assuming 'Unnamed: 9' is the current name
        'Unnamed: 10': 'Upper Limit',  # Assuming 'Unnamed: 10' is the current name
        'Unnamed: 11': 'SD',  # Assuming 'Unnamed: 11' is the current name
        'Unnamed: 12': 'Scale',  # Assuming 'Unnamed: 12' is the current name
        'Unnamed: 13': 'Count',  # Assuming 'Unnamed: 13' is the current name
        'Unnamed: 14': 'Shape',
        'Unnamed: 15': 'Mean',
    })
    try:
        # Read the Excel file
        df = data
        
        # Check if 'Include' column exists
        if 'Include' not in df.columns:
            raise KeyError("'Include' column not found in the Excel file.") #Some values on the excel sheet are relevant there to calculate the parameters but are no longer needed here.
        
        # Filter the relevant rows and columns
        filtered_df = df[df['Include'] == 'Include']
        
        # Create a dictionary to store the base components
        base_components = {}
        
        for _, row in filtered_df.iterrows():
            # Concatenate the primary to quinary fields to form the component name as on the excel, components may have multiple dependencies
            name = ' > '.join(filter(pd.notna, [row['Primary'], row['Secondary'], row['Tertiary'], row['Quaternary'], row['Quinary']]))
            parents = ' > '.join(name.split(' > ')[:-1])
            name = name.split(' > ')[-1]
            unit = row['Units']
            distribution = row['Distribution']
            time_for_determination = row['Time for determination (Year)']
            low= row['Lower Limit']
            high = row['Upper Limit']
            sd = row['SD']
            mean = row['Mean']
            scale = row['Scale']
            count = row['Count']
            shape = row['Shape']

            
            base_components[name] = BaseComponent(name, parents, unit, distribution, time_for_determination, low, high, sd, mean, shape, scale, count)
        
        return base_components
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

#Let the Monte Carlo fun begin! (If you're reading this, I am tired and I'm sorry for my jokes - or not)
def generate_random_value(distribution, mean=0, sd=1, low=0, high=1, shape=1, scale=1, count=1):
    """
    Generate a random value based on the specified distribution.
    
    Parameters:
        distribution (str): The name of the distribution (e.g., 'Uniform', 'Normal', 'Exponential').
        mean (float): The mean value (used for 'Normal', 'Gamma', 'Log-normal' distributions).
        sd (float): The standard deviation (used for 'Normal', 'Log-normal' distributions).
        low (float): The lower bound (used for 'Uniform' distribution).
        high (float): The upper bound (used for 'Uniform' distribution).
        shape (float): The shape parameter (used for 'Gamma', 'Weibull' distributions).
        scale (float): The scale parameter (used for 'Gamma', 'Exponential', 'Weibull' distributions).
        count (int): The number of occurrences (used for 'Poisson' distribution).
    
    Returns:
        float: A random value based on the specified distribution.
    """
#The function generates random values based on the specified distribution. The function takes the distribution type and the parameters for the distribution as input and returns a random value based on the distribution.
#Originally, more distributions were planned to be implemented, but only a few were implemented due to time constraints and the complexity of the distributions (values returned were not making much sense)
    if distribution == 'Uniform':
        return np.random.uniform(low, high)
    elif distribution == 'Normal' or distribution == 'nORMAL':  # Handling case sensitivity
        return np.random.normal(mean, sd)
    elif distribution == 'Exponential':
        return np.random.exponential(scale)
    elif distribution == 'Poisson':
        return np.random.poisson(count)
    elif distribution == 'Gamma':
        return np.random.gamma(shape, scale)
    elif distribution == 'Beta':
        return np.random.beta(shape, scale)
    elif distribution == 'Weibull':
        return np.random.weibull(shape) * scale
    elif distribution == 'Log-normal':
        return np.random.lognormal(mean, sd)
    elif distribution == 'Linear':  # Assuming Linear as a uniform distribution
        return np.random.uniform(low, high)
    elif distribution == 'Bernoulli':
        return 365 + np.random.binomial(1, sd)
    elif distribution == 'Exact': # this will mean it will be a umber
        return mean
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
 