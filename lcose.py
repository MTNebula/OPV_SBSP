#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import logging
import warnings
import time


#Number of years is equal to Years of Mission + 1 for pre-launch costs 
NUM_OF_TOTAL_YEARS = 21
NUM_OF_ITERATIONS = 10
DISCOUNT_RATE = 0.1
DISCOUNT_RATE_SD = 0.02
EXCEL_PATH = 'data/LCOE_Parameters.xlsx'
SHEET_NAME = 'Sheet2'

class CostComponent:
    def __init__(self, cost):
        self.costs = cost
        self.is_discounted_cost = False

    @property
    def mean(self):
        non_zero_cost = self.costs[self.costs != 0]
        if non_zero_cost.size > 0:
            return np.mean(non_zero_cost)
        else:
            return None

    @property
    def sd(self):
        non_zero_cost = self.costs[self.costs != 0]
        if non_zero_cost.size > 0:
            return np.std(non_zero_cost)
        else:
            return None

    @property
    def min(self):
        non_zero_cost = self.costs[self.costs != 0]
        if non_zero_cost.size > 0:
            return np.min(non_zero_cost)
        else:
            return None

    @property
    def max(self):
        non_zero_cost = self.costs[self.costs != 0]
        if non_zero_cost.size > 0:
            return np.max(non_zero_cost)
        else:
            return None

    def __add__(self, other):
        if isinstance(other, CostComponent):
            return CostComponent(self.costs + other.costs)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __mul__(self, other):
        if isinstance(other, CostComponent):
            return CostComponent(self.costs * other.costs)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

    def __neg__(self):
        """Define unary negation (i.e., -self)."""
        return CostComponent(-self.costs)

    def __truediv__(self, other):
        """Define division (i.e., self / other)."""
        if isinstance(other, CostComponent):
            return CostComponent(self.costs / other.costs)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

    def __sub__(self, other):
        """Define subtraction (i.e., self - other)."""
        if isinstance(other, CostComponent):
            return CostComponent(self.costs - other.costs)
        else:
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")


    def __repr__(self):
        return f"Cost={self.costs})"
    
    def get_discounted_cost(self):
        if self.is_discounted_cost == True:
            return self.discounted_cost
        else:
            UserWarning("Please discount it first")

    def apply_discounted_rate(self, d):
        if self.is_discounted_cost is False:
            cost = np.zeros(self.costs.shape)
            for i in range(self.costs.shape[0]):
                for j in range(self.costs.shape[1]):
                    cost[i, j] = self.costs[i, j]/(1+d) ** j
            self.costs = cost
            self.is_discounted_cost = True
        else:
            print("ERROR: it is already discounted please make sure that you want to do this")

        return self.costs

class BaseComponent:
    def __init__(self, name, parents, unit, distribution, time_for_determination, low, high, sd, mean, shape, scale, count):
        """
        Initialize a BaseComponent object with specified attributes.

        Args:
        - name (str): Name of the component.
        - parents (str): Parents of the component.
        - unit (str): Unit of measurement for the component.
        - distribution (str): Distribution type for cost calculation.
        - time_for_determination (str or int): Time specification for cost determination.

        Attributes:
        - name (str): Name of the component.
        - parents (str): Parents of the component.
        - unit (str): Unit of measurement for the component.
        - distribution (str): Distribution type for cost calculation.
        - time_for_determination (str or int): Time specification for cost determination.
        - cost (np.ndarray): Array of costs calculated based on time_for_determination.
        """
        self.name = name
        self.parents = parents
        self.unit = unit
        self.distribution = distribution
        self.time_for_determination = time_for_determination
        self.low = low
        self.high = high
        self.sd = sd
        self.mean = mean
        self.shape = shape
        self.scale = scale
        self.count = count
        self.cost_component = CostComponent(self._generate_cost_array())

    def __repr__(self):
        """
        Return a string representation of the BaseComponent object.
        """
        return f"{self.name}', unit='{self.unit}', distribution='{self.distribution}', parents='{self.parents}')"

    def __add__(self, other):
        """
        Define addition behavior for BaseComponent objects.
        
        Args:
        - other (BaseComponent): Another BaseComponent object to add.

        Returns:
        - list: Combined cost array element-wise.

        Raises:
        - TypeError: If 'other' is not an instance of BaseComponent.
        """
        if isinstance(other, BaseComponent):
            combined_cost =  self.cost_component + other.cost_component
            return combined_cost
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __mul__(self, other):
        """
        Define multiplication behavior for BaseComponent objects.
        
        Args:
        - other (BaseComponent): Another BaseComponent object to multiply.

        Returns:
        - list: Combined cost array element-wise.

        Raises:
        - TypeError: If 'other' is not an instance of BaseComponent.
        """
        if isinstance(other, BaseComponent):
            combined_cost = self.cost * other.cost
            return combined_cost
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")
        
    def get_cost(self):
        return self.cost_component

    def _generate_cost_array(self):
        """
        Generate an array of costs based on time_for_determination and distribution.
        
        Returns:
        - np.ndarray: Array of costs calculated based on time_for_determination.
        """
        cost = np.zeros((NUM_OF_ITERATIONS, NUM_OF_TOTAL_YEARS+1))
        try:
            for j in range(cost.shape[0]):
                if self.time_for_determination in [0, 'Y0', 'YO', 'y0']:
                    cost[j, 0] = generate_random_value(self.distribution, mean=self.mean, sd=self.sd, low=self.low, high=self.high, shape=self.shape, scale=self.scale, count=self.count)
                elif 'Yearly' in self.time_for_determination:
                    for i in range(1, cost.shape[1]):
                        cost[j, i] = generate_random_value(self.distribution, mean=self.mean, sd=self.sd, low=self.low, high=self.high, shape=self.shape, scale=self.scale, count=self.count)
                else:
                    raise ValueError(f"Unrecognized time_for_determination value: '{self.time_for_determination}'")
        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            logging.info('Normal distribution is assumed.')
            cost[:] = generate_random_value('Normal')
        except TypeError as te:
            logging.error(f"TypeError occurred: {te}")
            logging.info('Normal distribution is assumed.')
            cost[:] = generate_random_value('Normal')
        except Exception as e:
            logging.error(f"Error while calculating cost for component '{self.name}': {e}")
            logging.info('Normal distribution is assumed.')
            cost[:] = generate_random_value('Normal')
        return cost
    
    

class CollectorClass:
    """
    Represents a collector class for managing parts and their costs.
    """

    def __init__(self, name):
        """
        Initialize a CollectorClass instance.

        Args:
            name (str): The name of the collector class.
        """
        self.name = name
        self.parts = {}
        self.cost_component = None



    def add_part(self, parts, operation="+"):
        """
        Add parts to the collector class with specified operation on costs.
        
        Args:
            parts (list or BaseComponent or CollectorClass): Parts to add.
            operation (str, optional): Operation to perform on costs ('+', '*', '-', '/'). Defaults to '+'.
        Raises:
            TypeError: If parts are not instances of BaseComponent or CollectorClass.
            ValueError: If parts have different length cost arrays.

        """
        if isinstance(parts, BaseComponent) or isinstance(parts, CollectorClass):
            parts = [parts]

        for part in parts:
            if isinstance(part, BaseComponent) or isinstance(part, CollectorClass):
                self.parts[part.name] = part
            else:
                raise TypeError("Only instances of BaseComponent or CollectorClass can be added as parts.")

        # Initialize cost_component if it's None
        if self.cost_component is None:
            shape = parts[0].get_cost().costs.shape  # Copy the cost array from the first part
            if operation == "+" or operation == "-":
                self.cost_component = CostComponent(np.zeros(shape))
            elif operation == "*" or operation == "/":
                self.cost_component = CostComponent(np.ones(shape))

        # Perform specified operation on costs element-wise
        if operation == "+":
            for part in parts:
                self.cost_component += part.get_cost()
        elif operation == "*":
            for part in parts:
                self.cost_component *= part.get_cost()
        elif operation == "-":
            for part in parts:
                self.cost_component -= part.get_cost()
        elif operation == "/":
            for part in parts:
                part_cost = part.get_cost().costs
                part_cost[part_cost == 0.] = 1.
                self.cost_component /= part.get_cost()

    def set_cost(self, cost):
        self.cost_component = CostComponent(cost)

    def get_cost(self):
        return self.cost_component

    def __repr__(self):
        return f"CollectorClass(name='{self.name}', parts={list(self.parts.keys())})"
    

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
        'Unnamed: 5': 'Quintenary',
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
            raise KeyError("'Include' column not found in the Excel file.")
        
        # Filter the relevant rows and columns
        filtered_df = df[df['Include'] == 'Include']
        
        # Create a dictionary to store the base components
        base_components = {}
        
        for _, row in filtered_df.iterrows():
            # Concatenate the primary to quintenary fields to form the component name
            name = ' > '.join(filter(pd.notna, [row['Primary'], row['Secondary'], row['Tertiary'], row['Quaternary'], row['Quintenary']]))
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
    

if __name__ == "__main__":

    # Example usage
    base_components = extract_base_components(EXCEL_PATH, SHEET_NAME)
    repairs = CollectorClass("Repairs")
    parts = [base_components["Material Repairs"], base_components["Material Repairs Cost"]]
    repairs.add_part(parts, "*")


    launch = CollectorClass("Launch")
    parts = [base_components["Repair Number of Launches"], base_components["Repair Launching cost"]]
    launch.add_part(parts, "*")

    maintenance = CollectorClass("Maintenance")
    parts = [launch, repairs]
    maintenance.add_part(parts, "*")

    emission = CollectorClass("Emission")
    parts = [base_components["Total emissions"], base_components["Emission Cost"]]
    emission.add_part(parts, "*")

    fuel = CollectorClass("Fuel")
    parts = [base_components["Fuel use"], base_components["Fuel cost"]]
    fuel.add_part(parts, "*")


    energy = CollectorClass("Energy")
    parts = [base_components["Installed Capacity"], base_components["Efficiency (n)"], base_components["#days in year x"], base_components["Hours in a day"]]
    energy.add_part(parts, "*")
    energy.add_part(base_components["Energy consumption"], "-")

    transport = CollectorClass("Transport")
    parts = [base_components["Distance of transport"], base_components["Cost of Fuel"]]
    transport.add_part(parts, "*")

    manufacture_panels = CollectorClass("Manufacture Panels")
    parts = [base_components['Number of panels'], base_components['Raw Materials'], base_components['Processing']]
    manufacture_panels.add_part(parts, "*")
    manufacture_panels.add_part(transport)
    manufacture = manufacture_panels

    # seems to be a lot
    deployment = CollectorClass("Deployment")
    deployment.add_part(base_components['Logistics Costs'])

    # seems to be a lot as well
    launch = CollectorClass("Launch")
    parts = [base_components['Number of Launches'], base_components['Launching cost']]
    launch.add_part(parts)


    # Cost overall should be checked and looked over
    cost = CollectorClass("Cost")
    parts = [launch, deployment, manufacture, fuel, maintenance]
    cost.add_part(parts)

    cost.cost_component.apply_discounted_rate(DISCOUNT_RATE)
    energy.cost_component.apply_discounted_rate(DISCOUNT_RATE)


    lcoe = CollectorClass("LCOSE")
    parts = [cost, energy]
    lcoe.add_part(parts, "/")
    print("#########################LCOSE#####################################")