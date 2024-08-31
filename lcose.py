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

colors = sns.color_palette("viridis", 50)
sns.set_palette("viridis")


#Number of years is equal to Years of Mission + 1 for pre-launch costs 
NUM_OF_TOTAL_YEARS = 21
NUM_OF_ITERATIONS = 1000
DISCOUNT_RATE = 0.1
DISCOUNT_RATE_SD = 0.02
EXCEL_PATH = 'data/LCOE_Parameters.xlsb.xlsx'
SHEET_NAME = 'OPV Scenario 2'
SHEET_NAMES = ['PV Scenario 1', 'PV Scenario 2', 'OPV Scenario 1', 'OPV Scenario 2']

# Values for the scenarios are calculated directly in the excel file based on the technology and original efficiency.

values_of_scenarios = { 'PV Scenario 1':[], 'PV Scenario 2':[], 'OPV Scenario 1':[], 'OPV Scenario 2':[]}


class CostComponent:
    def __init__(self, cost, name, unit):
        self.name = name
        self.costs = cost
        self.unit = unit
        self.is_discounted_cost = False
        self.costs_per_iteration = None
        try: 
            # self.plot_and_save_histogram()
            # self.plot_and_save_histogram_per_iteration()
            # despine the top and right axis
            sns.despine()
            #add a gray grid
            plt.grid(color='gray', linestyle='-', linewidth=0.5)
           
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"There was a problem with plotting {self.name}")
            return None

# Define properties for mean, standard deviation, min, and max for each variable used to run the simulation

    @property
    def mean(self):
        costs = self.get_cost_per_iteration()
        non_zero_costs = costs[costs != 0]
        return np.mean(non_zero_costs)

    @property
    def sd(self):
        costs = self.get_cost_per_iteration()
        non_zero_costs = costs[costs != 0]
        return np.std(non_zero_costs)

    @property
    def min(self):
        costs = self.get_cost_per_iteration()
        non_zero_costs = costs[costs != 0]
        return np.min(non_zero_costs)

    @property
    def max(self):
        costs = self.get_cost_per_iteration()
        non_zero_costs = costs[costs != 0]
        return np.max(non_zero_costs)



# Defines the mathematical operations for the CostComponent class to be able to add, subtract, multiply, and divide cost components.

    def __add__(self, other):
        if isinstance(other, CostComponent):
            return CostComponent(self.costs + other.costs, self.name, self.unit)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __mul__(self, other):
        if isinstance(other, CostComponent):
            return CostComponent(self.costs * other.costs, self.name, self.unit)
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

    def __neg__(self):
        """Define unary negation (i.e., -self)."""
        return CostComponent(-self.costs, self.name, self.unit)

    def __truediv__(self, other):
        """Define division (i.e., self / other)."""
        if isinstance(other, CostComponent):
            return CostComponent(self.costs / other.costs, self.name, self.unit)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")

    def __sub__(self, other,):
        """Define subtraction (i.e., self - other)."""
        if isinstance(other, CostComponent):
            return CostComponent(self.costs - other.costs, self.name, self.unit)
        else:
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self)}' and '{type(other)}'")

# Define the string representation of the CostComponent class to create the discounted value.

    def __repr__(self):
        return f"Cost={self.costs})"
    
    def get_discounted_cost(self):
        if self.is_discounted_cost == True:
            return copy.deepcopy(self.discounted_cost)
        else:
            UserWarning("Please discount it first")

    def apply_discounted_rate(self, d):
        if self.is_discounted_cost is False:
            cost = np.zeros(self.costs.shape)
            for i in range(self.costs.shape[0]):
                for j in range(self.costs.shape[1]):
                    cost[i, j] = self.costs[i, j]/(1+d) ** j #discounting the cost according to year and discount rate
            self.costs = cost
            self.is_discounted_cost = True
        else:
            print("ERROR: it is already discounted please make sure that you want to do this")

        return self.costs
    
    def get_cost_per_iteration(self):
        if self.costs_per_iteration is None and self.costs.ndim == 2:
            self.costs_per_iteration = np.sum(self.costs, axis=1)
            return copy.deepcopy(self.costs_per_iteration)
        else:
            return copy.deepcopy(self.costs) 
    
    def get_non_zero_cost(self):
        return copy.deepcopy(self.costs[self.costs != 0])

    # Histograms are created to show the distribution of costs for each variable in the simulation and make sure that the values are within the expected range.
    def plot_and_save_histogram(self, folder='plots'):
        # Flatten the cost matrix to get all individual cost values
        flat_costs = self.get_non_zero_cost().flatten()

        # Create a folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Plot the histogram with 20 bins
        plt.hist(flat_costs, bins=20)
        plt.title(f'{self.name}') 
        plt.xlabel(f'{self.unit}')
        plt.ylabel('Frequency')
        sns.despine()
        plt.grid(color='gray', linestyle='--', linewidth=0.5)

        # Save the plot in the specified folder with the filename based on the variable name
        plt.savefig(f'{folder}/{self.name}.png')

        # # Show the plot
        # plt.show()
        plt.close()

    def plot_and_save_histogram_per_iteration(self, folder='plots_per_iteration'):
        # Flatten the cost matrix to get all individual cost values
        flat_costs = self.get_cost_per_iteration()

        # Create a folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Plot the histogram with 20 bins
        plt.hist(flat_costs, bins=20, edgecolor='black')
        plt.title(f'{self.name}') 
        plt.xlabel(f'{self.unit}')
        plt.ylabel('Frequency')
        sns.despine()
        plt.grid(color='gray', linestyle='-', linewidth=0.5)

        # Save the plot in the specified folder with the filename based on the variable name
        plt.savefig(f'{folder}/{self.name}.png')

        # # Show the plot
        # plt.show()
        plt.close()

# base components are used for the simulation and are the building blocks for the cost calculations
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
        self.cost_component = CostComponent(self._generate_cost_array(), name, self.unit)

# Define the base components and what they are made of. The cost of each component is calculated based on the distribution and time for determination, parents are used to show relationships between components.
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
        return copy.deepcopy(self.cost_component)
    
    def set_cost(self, cost):
        self.cost_component = CostComponent(cost, self.name, self.unit)

    def _generate_cost_array(self):
        """
        Generate an array of costs based on time_for_determination and distribution.
        
        Returns:
        - np.ndarray: Array of costs calculated based on time_for_determination.
      
          """
        # This is where the Monte Carlo simulation is run to generate the costs for each component based on the distribution and time for determination for each iteration.
        cost = np.zeros((NUM_OF_ITERATIONS, NUM_OF_TOTAL_YEARS+1))
        try:
            for j in range(cost.shape[0]):
                if self.time_for_determination in [0, 'Y0', 'YO', 'y0']: #Makes sure that the cost is only calculated once for these components
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

    def __init__(self, name, unit="unit"):
        """
        Initialize a CollectorClass instance.

        Args:
            name (str): The name of the collector class.
        """
        self.name = name
        self.unit = unit
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
            shape = parts[0].get_cost().costs.shape  # Get the shape of the cost array
            if operation == "+" or operation == "-":
                self.cost_component = CostComponent(np.zeros(shape), self.name, self.unit)
            elif operation == "*":
                self.cost_component = CostComponent(np.ones(shape), self.name, self.unit)

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
            self.cost_component = parts[0].get_cost() / parts[1].get_cost()

    def set_cost(self, cost):
        self.cost_component = CostComponent(cost, self.name, self.unit)

    def get_cost(self):
        return copy.deepcopy(self.cost_component)

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
    

class  CostCalculator:
    def __init__(self, excel_path, sheet_name):
        print(excel_path)
        print(sheet_name)
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.base_components = extract_base_components(excel_path, sheet_name)
        self.repairsmaterial = None
        self.launchrepair = None
        self.repairtotal = None
        self.maintenance = None
        self.emissiontotal = None
        self.emissiontotalsystem = None
        self.emissiontotallaunch = None
        self.emission = None
        self.fuel = None
        self.efficiencyboost = None
        self.efficiencyd = None
        self.efficiencybd = None
        self.efficiencyt = None
        self.energy = None
        self.manufacture_panels = None
        self.manufacture = None
        self.launch = None
        self.investment = None
        self.cost = None
        self.costwithoutemission = None
        self.totalcost = None
        self.totalcostwithoutemission = None
        self.totalenergy = None
        self.lcoe = None
        self.lcoewithoutemission = None

        self.calculate_costs()

# Calculates the repairs cost accounting for the material, which is assumed to be 5% of the original materials costs, launching costs for those materials and also a 5% of the original assembly cost. The total cost is then calculated based on the probability of material repairs and the total cost of repairs.
#This accounts for the fact that material repairs may or may not happen   
    def calculate_costs(self):
        self.repairsmaterial = CollectorClass("Repairs Material", "USD")
        parts = [self.base_components["Material Repairs Cost"], self.base_components["Repair Assembly"]]
        self.repairsmaterial.add_part(parts, "+")

        self.launchrepair = CollectorClass("Launch Repair", 'USD')
        parts = [self.base_components["Repair Number of Launches"], self.base_components["Repair Launching cost"]]
        self.launchrepair.add_part(parts, "*")

        self.repairtotal = CollectorClass("Repairs Assembly", 'USD')
        parts = [self.repairsmaterial, self.launchrepair]
        self.repairtotal.add_part(parts, "+")

        self.maintenance = CollectorClass("Maintenance", 'USD')
        parts = [self.base_components["Probability of Material Repairs"], self.repairtotal]
        self.maintenance.add_part(parts, "*")

# We are now calculating the emissions cost based on the energy system emissions and the launching emissions. The total emissions cost is then calculated based on the emissions cost, the area of the panels, and the total emissions.
        self.emissiontotalsystem = CollectorClass("Emissions for Energy System", 'TCO2e')
        parts = [self.base_components["Energy system emissions"], self.base_components["Area of panels"]]
        self.emissiontotalsystem.add_part(parts, "*")

        self.emissiontotallaunch = CollectorClass("Emissions for Launching", 'TCO2e')
        parts = [self.base_components["Number of Launches"], self.base_components["Launching emissions"]]
        self.emissiontotallaunch.add_part(parts, "*")

        self.emissiontotal = CollectorClass("Total Emission", 'TCO2e')
        parts = [self.emissiontotalsystem, self.emissiontotallaunch]
        self.emissiontotal.add_part(parts, "+")

        self.emission = CollectorClass("Emission Cost", 'USD')
        parts = [self.emissiontotal, self.base_components["Emission Cost"]]
        self.emission.add_part(parts, "*")

# We are now calculating the fuel cost based on the fuel use and the fuel cost. The total fuel cost is then calculated based on the fuel cost and the fuel use.

        self.fuel = CollectorClass("Fuel", 'USD')
        parts = [self.base_components["Fuel use"], self.base_components["Fuel cost"]]
        self.fuel.add_part(parts, "*")

# This section calculates the total energy generated by the system throughout the mission's life

#First the efficiency for each year is calculated:
# The efficiency boost depends on whether there is a repair or not. If there is a repair, the efficiency boost is calculated and will be added to that year's efficiencyt. If there is no repair, the efficiency boost is 0%. 
        self.efficiencyboost = CollectorClass("Efficiency boost", '%')
        parts = [self.base_components["Efficiency boost repair"], self.base_components["Probability of Material Repairs"]]
        self.efficiencyboost.add_part(parts, "*")

        self.efficiencyd = CollectorClass("Efficiency Degradation Rate", '%')
        parts = [self.base_components["Efficiency Degradation Rate"]]
        self.efficiencyd.add_part(parts)

# We now calculate the degradation per year t following a formula that mirrors that of depreciation but substrating the degradation rate from 1. The efficiency degradation is then calculated based on the original efficiency and the degradation rate. 
        costs = self.efficiencyd.get_cost().costs 
        for iteration in range(costs.shape[0]):
            for year in range(costs.shape[1]):
                costs[iteration, year] = self.base_components["Original Efficiency"].cost_component.costs[iteration, year] * (1 - self.base_components["Efficiency Degradation Rate"].cost_component.costs[iteration, year])**year

        self.efficiencyd = CollectorClass("Efficiency degradation", '%')
        self.efficiencyd.set_cost(costs)

        self.efficiencybd = CollectorClass("Efficiency bd", '%') #The efficiency boost is multiplied by the new degraded efficiency to get the proportional boost
        parts = [self.efficiencyboost, self.efficiencyd]
        self.efficiencybd.add_part(parts, "*")

        self.efficiencyt = CollectorClass("Efficiency t", '%') #These efficiencies are then summed to account for potential boosts due to repairs
        parts = [self.efficiencybd, self.efficiencyd]
        self.efficiencyt.add_part(parts, "+")

#Finally, we are able to calculate the total energy generated by the system throughout the mission's life based on the installed capacity, the efficiencyt, the number of days in a year, and the hours in a day.
        self.energy = CollectorClass("Energy", 'MWh')
        parts = [self.base_components["Installed Capacity"], self.efficiencyt, self.base_components["#days in year t"], self.base_components["Hours in a day"]]
        self.energy.add_part(parts, "*")

#We calculate the manufacture of the panels costs
        self.manufacture_panels = CollectorClass("Manufacture Panels", 'USD/m2')
        parts = [self.base_components['Raw Materials'], self.base_components['Processing']]
        self.manufacture_panels.add_part(parts, "+")

        self.manufacture = CollectorClass("Manufacture Cost", 'USD')
        parts = [self.base_components['Area of panels'], self.manufacture_panels]
        self.manufacture.add_part(parts, "*")

#We calculate the launch costs based on the number of launches and the launching cost. The number of launches depends on the mass and stowability of the panels.
        self.launch = CollectorClass("Launch", 'USD')
        parts = [self.base_components['Number of Launches'], self.base_components['Launching cost']]
        self.launch.add_part(parts, "*")

# we start bringing the costs together into the actual terms for the LCOE formula. In this case, investment. 

        self.investment = CollectorClass("Investment Costs", 'USD')
        parts = [self.base_components['Assembly'], self.launch, self.manufacture]
        self.investment.add_part(parts, "+")

#These are the total costs for the system.
        self.cost = CollectorClass("Cost", 'USD')
        parts = [self.investment, self.emission, self.fuel, self.maintenance]
        self.cost.add_part(parts, "+")

#Because carbon is not normally accounted for in LCOE, a version without the emissions is calculated.
        self.costwithoutemission = CollectorClass("Cost", 'USD')
        parts = [self.investment, self.fuel, self.maintenance]
        self.costwithoutemission.add_part(parts, "+")

#It's depreciation time! This is done to show the value of the asset in present terms. 
        costs = self.base_components["Depreciation of Assets"].get_cost().costs
        for iteration in range(costs.shape[0]):
            for year in range(costs.shape[1]):
                costs[iteration, year] = (1 + costs[iteration, year]) ** year

        self.base_components["Depreciation of Assets"].set_cost(costs)

        self.totalcost = CollectorClass("Cost Total", 'USD')
        parts = [self.cost, self.base_components["Depreciation of Assets"]]
        self.totalcost.add_part(parts, "/")
        self.totalcost.set_cost(self.totalcost.get_cost().get_cost_per_iteration())
        # self.totalcost.get_cost().plot_and_save_histogram()
        print(self.totalcost)
        print(self.totalcost.get_cost().costs)

        self.totalcostwithoutemission = CollectorClass("Cost Total", 'USD')
        parts = [self.costwithoutemission, self.base_components["Depreciation of Assets"]]
        self.totalcostwithoutemission.add_part(parts, "/")
        self.totalcostwithoutemission.set_cost(self.totalcostwithoutemission.get_cost().get_cost_per_iteration())
        # self.totalcostwithoutemission.get_cost().plot_and_save_histogram()
        print(self.totalcostwithoutemission)
        print(self.totalcostwithoutemission.get_cost().costs)

        self.totalenergy = CollectorClass("Energy Total", 'MWh')
        parts = [self.energy, self.base_components["Depreciation of Assets"]]
        self.totalenergy.add_part(parts, "/")
        self.totalenergy.set_cost(self.totalenergy.get_cost().get_cost_per_iteration())
        print(self.totalenergy)
        print(self.totalenergy.get_cost().costs)

# We can FINALLY calculate the LCOSE and LCOSE without emissions. 

        self.lcoe = CollectorClass("LCOSE", 'USD/Mwh')
        parts = [self.totalcost, self.totalenergy]
        self.lcoe.add_part(parts, "/")
        print(self.lcoe.get_cost().costs)

        self.lcoewithoutemission = CollectorClass("LCOSE", 'USD/Mwh')
        parts = [self.totalcostwithoutemission, self.totalenergy]
        self.lcoewithoutemission.add_part(parts, "/")

        print(self.lcoewithoutemission.get_cost().costs)

# This is just to show the results of the calculations
def create_table(scenarios):
    data = []
    for scenario, calculator in scenarios.items():
        calculator.base_components['lcoe'] = calculator.lcoe
        calculator.base_components['emissiontotallaunch'] = calculator.emissiontotallaunch
        calculator.base_components['emissiontotalsystem'] = calculator.emissiontotalsystem
        calculator.base_components['maintenance'] = calculator.maintenance
        calculator.base_components['emission'] = calculator.emission
        calculator.base_components['fuel'] = calculator.fuel
        calculator.base_components['manufacture'] = calculator.manufacture
        calculator.base_components['launch'] = calculator.launch
        calculator.base_components['energy'] = calculator.energy


        for component in calculator.base_components.values():
            data.append({
                'Scenario': scenario,
                'Component': component.name,
                'Mean': component.cost_component.mean,
                'Max': component.cost_component.max,
                'Min': component.cost_component.min,
                'SD': component.cost_component.sd
            })

    df = pd.DataFrame(data)
    # Assuming `table` is the DataFrame you want to save
    df.to_csv('data/results/Results_per_scenario.csv', index=False)

    return df

# Plots to show results

def plot_error_bars(scenarios, exclude_components=[]):
    fig, ax = plt.subplots()

    for scenario, calculator in scenarios.items():
        components = [component for component in calculator.base_components.keys() if component not in exclude_components]
        means = [calculator.base_components[component].cost_component.mean for component in components]
        errors = [calculator.base_components[component].cost_component.sd for component in components]

        x = range(len(components))
        ax.errorbar(x, means, yerr=errors, label=scenario, fmt='o')

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=90)
    ax.set_xlabel('Component')
    ax.set_ylabel('Mean Cost')
    ax.set_title('Mean Cost with Error Bars for Each Component in Each Scenario')
    ax.legend()
    # Use log scale for y-axis
    ax.set_yscale('log')
    sns.despine()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #get the labels to not overlap with the graph
    plt.tight_layout()
    # Save the plot
    plt.savefig(f'data/results/plot/error_bars_{scenario}.png')
    plt.close()


if __name__ == "__main__":


    for sheet_name in values_of_scenarios.keys():
        print(f"Calculating {sheet_name}")
        values_of_scenarios[sheet_name] = CostCalculator(EXCEL_PATH, sheet_name)
        print(f"Finished Calculating {sheet_name}")

# %% Create the table
    # Usage
    table = create_table(values_of_scenarios)
    print(table)

# %% Create error bars
    exclude_components = ['#days in year t', 'Original Efficiency','Hours in a day','Mission Life', 'Depreciation of Assets', 'Probability of Material Repairs','Emission Cost','Launching cost','Repair Launching cost','Fuel cost','Energy system emissions', 'Launching emissions', 'Efficiency Degradation Rate','Efficiency boost repair', 'Raw Materials','Processing']
    plot_error_bars(values_of_scenarios, exclude_components)
    #use viridis color map
    sns.set_palette("viridis")
    # move labels to the right so they do not overlap
    plt.tight_layout()



# # %% Create plot of opv and pv
#     degradation_comp_scenarios = [values_of_scenarios['PV Scenario 1'], values_of_scenarios['OPV Scenario 1']] 
#     plt.figure(figsize=(10, 6))
#     for scenario in degradation_comp_scenarios:
#         print(f"Degradation calc for the following scenario:{scenario.sheet_name}")
#         print(scenario.efficiencyt.get_cost())
#         mean_per_year = np.mean(scenario.efficiencyt.get_cost().costs, axis=0)
#         print(f"mean per year {mean_per_year}")

#             # Create a range for the x-axis (years)
#         years = range(1, len(mean_per_year))

#         # Plot the data

#         plt.plot(years, mean_per_year[1:], label=scenario.sheet_name)

#     # Add labels and title
#     plt.xlabel('Years')
#     plt.ylabel('Mean Cost per Year')
#     plt.title('Comparison of Mean Costs per Year for PV and OPV Scenarios')
#     plt.show()

#     # Add a legend
#     plt.legend()

#     # Show the plot
#     plt.savefig('data/results/mean_costs_per_year.png')


# %%calculate lcoe
    for name, information in values_of_scenarios.items():
        lcoe = information.lcoe.get_cost().costs[:5]
        print(f"LCOE for {name}: {lcoe}")

# %% lcoe plot
    # Create a list to store the means and standard deviations for each scenario
    means = []
    stds = []
    scenario_names = []

    # Iterate over the scenarios
    for name, information in values_of_scenarios.items():
        lcoe = information.lcoe.get_cost().costs 
        mean = np.mean(lcoe)  # Calculate the mean
        std = np.std(lcoe)  # Calculate the standard deviation
        means.append(mean)
        stds.append(std)
        scenario_names.append(name)

    # Create a list for the x-axis positions
    x_pos = range(1, len(scenario_names) + 1)
    # Create the plot
    plt.figure(figsize=(10, 6))
    # Plot the error bars
    plt.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, label='LCOSE with error bars')
    # Plot the means as horizontal lines
    plt.hlines(means, xmin=[x - 0.2 for x in x_pos], xmax=[x + 0.2 for x in x_pos], colors='green', label='Mean LCOSE')
    sns.despine()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # Add labels and title
    plt.xlabel('Scenario')
    plt.ylabel('LCoSE (USD/MWh)')
    # Set the x-axis ticks and labels
    plt.xticks(x_pos, scenario_names)
    # Add a legend without a frame
    plt.legend(frameon=False)
    # Show the plot
    plt.savefig('data/results/lcoe_per_scenario.png')
    plt.close()

# %% lcoe without emissions plot
    # Create a list to store the means and standard deviations for each scenario
    means = []
    stds = []
    scenario_names = []
    # Iterate over the scenarios
    for name, information in values_of_scenarios.items():
        lcoe = information.lcoewithoutemission.get_cost().costs  
        mean = np.mean(lcoe)  # Calculate the mean
        std = np.std(lcoe)  # Calculate the standard deviation
        means.append(mean)
        stds.append(std)
        scenario_names.append(name)
    # Create a list for the x-axis positions
    x_pos = range(1, len(scenario_names) + 1)
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # Plot the error bars
    plt.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, label='LCOE with error bars excluding emissions')
    # Plot the means as horizontal lines
    plt.hlines(means, xmin=[x - 0.2 for x in x_pos], xmax=[x + 0.2 for x in x_pos],colors='green', label='Mean LCOE')
    # Add labels and title
    plt.xlabel('Scenario')
    plt.ylabel('LCoSE (USD/MWh)')
    sns.despine()
    # Set the x-axis ticks and labels
    plt.xticks(x_pos, scenario_names)
    # Add a legend without frame
    plt.legend(frameon=False)
    # Show the plot
    plt.savefig('data/results/lcoewithouthemissions_per_scenario.png')



# %% costs plot
    plt.close()
    # Initialize empty lists to store the costs
    maintenance_costs = []
    emission_costs = []
    fuel_costs = []
    manufacture_costs = []
    launch_costs = []

    # Loop through the scenarios
    for name, information in values_of_scenarios.items():
        maintenance_costs.append(np.mean(information.maintenance.get_cost().get_cost_per_iteration()))
        emission_costs.append(np.mean(information.emission.get_cost().get_cost_per_iteration()))
        fuel_costs.append(np.mean(information.fuel.get_cost().get_cost_per_iteration()))
        manufacture_costs.append(np.mean(information.manufacture.get_cost().get_cost_per_iteration()))
        launch_costs.append(np.mean(information.launch.get_cost().get_cost_per_iteration()))

    # Create a range for the x-axis
    x = np.arange(len(values_of_scenarios))

    # Plot the stacked histogram
    plt.bar(x, maintenance_costs, label='Maintenance')
    plt.bar(x, emission_costs, bottom=maintenance_costs, label='Emission')
    plt.bar(x, fuel_costs, bottom=np.array(maintenance_costs)+np.array(emission_costs), label='Fuel')
    plt.bar(x, manufacture_costs, bottom=np.array(maintenance_costs)+np.array(emission_costs)+np.array(fuel_costs), label='Manufacture')
    plt.bar(x, launch_costs, bottom=np.array(maintenance_costs)+np.array(emission_costs)+np.array(fuel_costs)+np.array(manufacture_costs), label='Launch')

    # Add labels and title
    sns.despine

    plt.xlabel('Scenarios')
    plt.ylabel('Costs')
    plt.title('Mean costs for each scenario')
    plt.xticks(x, values_of_scenarios.keys())
    plt.legend(frameon=False)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()



 # plt.yscale('log')

    # Show the plot
    plt.savefig('data/results/Mean Costs for each scenario.png')
    plt.show()

    # %%
    # Initialize empty lists to store the emissions
    emission_system = []
    emission_launch = []

    # Loop through the scenarios
    for name, information in values_of_scenarios.items():
        emission_system.append(np.mean(information.emissiontotalsystem.get_cost().get_cost_per_iteration()))
        emission_launch.append(np.mean(information.emissiontotallaunch.get_cost().get_cost_per_iteration()))

    # Create a range for the x-axis
    x = np.arange(len(values_of_scenarios))

    # Plot the stacked histogram
    plt.bar(x, emission_system, label='Emission System')
    plt.bar(x, emission_launch, bottom=np.array(emission_system), label='Emission Launch')

    # Add labels and title
    

    # plt.xlabel('Scenarios')
    plt.ylabel('Emissions (ton C02e)')
    plt.xticks(x, values_of_scenarios.keys())
    plt.legend(frameon=False)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    # Show the plot
    plt.savefig('data/results/emissions.png')
    plt.show()
