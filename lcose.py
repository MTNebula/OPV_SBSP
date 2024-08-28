#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import logging
import warnings
import time
import matplotlib.pyplot as plt
import os


#Number of years is equal to Years of Mission + 1 for pre-launch costs 
NUM_OF_TOTAL_YEARS = 20
NUM_OF_ITERATIONS = 10
DISCOUNT_RATE = 0.1
DISCOUNT_RATE_SD = 0.02
EXCEL_PATH = 'data/LCOE_Parameters.xlsx'
SHEET_NAME = 'Sheet2'

class CostComponent:
    def __init__(self, cost, name, unit):
        self.name = name
        self.costs = cost
        self.unit = unit
        self.is_discounted_cost = False
        self.costs_per_iteration = None
        try: 
            self.plot_and_save_histogram()
            self.plot_and_save_histogram_per_iteration()
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"There was a problem with plotting {self.name}")
            return None

    @property
    def mean(self):
        self.get_cost_per_iteration()
        mean = []
        for costs in [self.costs, self.costs_per_iteration]:
            non_zero_cost = costs[costs != 0]
            if non_zero_cost.size > 0:
                mean.append(np.mean(non_zero_cost))
            else:
                print(f"ERROR: creating mean for some values because all alues are zero for variable: {self.name}")
        return mean

    @property
    def sd(self):
        self.get_cost_per_iteration()
        sd = []
        for costs in [self.costs, self.costs_per_iteration]:
            non_zero_cost = costs[costs != 0]
            if non_zero_cost.size > 0:
                sd.append(np.std(non_zero_cost))
            else:
                print(f"ERROR: creating std for some values because all alues are zero for variable: {self.name}")
        return sd

    @property
    def min(self):
        self.get_cost_per_iteration()
        min = []
        for costs in [self.costs, self.costs_per_iteration]:
            non_zero_cost = costs[costs != 0]
            if non_zero_cost.size > 0:
                min.append(np.min(non_zero_cost))
            else:
                print(f"ERROR: creating min for some values because all alues are zero for variable: {self.name}")
        return min

    @property
    def max(self):
        self.get_cost_per_iteration()
        max = []
        for costs in [self.costs, self.costs_per_iteration]:
            non_zero_cost = costs[costs != 0]
            if non_zero_cost.size > 0:
                max.append(np.min(non_zero_cost))
            else:
                print(f"ERROR: creating max for some values because all alues are zero for variable: {self.name}")
        return max

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
    
    def get_cost_per_iteration(self):
        if self.costs_per_iteration is None:
            self.costs_per_iteration = np.sum(self.costs, axis=1)
        return self.costs_per_iteration
    
    def get_non_zero_cost(self):
        return self.costs[self.costs != 0]

    
    def plot_and_save_histogram(self, folder='plots'):
        # Flatten the cost matrix to get all individual cost values
        flat_costs = self.get_non_zero_cost().flatten()

        # Create a folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Plot the histogram with 20 bins
        plt.hist(flat_costs, bins=20, edgecolor='black')
        plt.title(f'{self.name}') 
        plt.xlabel(f'{self.unit}')
        plt.ylabel('Frequency')
        plt.grid(True)

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
        plt.grid(True)

        # Save the plot in the specified folder with the filename based on the variable name
        plt.savefig(f'{folder}/{self.name}.png')

        # # Show the plot
        # plt.show()
        plt.close()

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
    
    def set_cost(self, cost):
        self.cost_component = CostComponent(cost, self.name, self.unit)

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
            shape = parts[0].get_cost().costs.shape  # Copy the cost array from the first part
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
            raise KeyError("'Include' column not found in the Excel file.")
        
        # Filter the relevant rows and columns
        filtered_df = df[df['Include'] == 'Include']
        
        # Create a dictionary to store the base components
        base_components = {}
        
        for _, row in filtered_df.iterrows():
            # Concatenate the primary to quinary fields to form the component name
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

    # Calculate Maintenance Costs

    base_components = extract_base_components(EXCEL_PATH, SHEET_NAME)
    repairsmaterial = CollectorClass("Repairs Material", "USD")
    parts = [ base_components["Material Repairs Cost"], base_components["Repair Assembly"]]
    repairsmaterial.add_part(parts, "+")


    launchrepair = CollectorClass("Launch Repair", 'USD')
    parts = [base_components["Repair Number of Launches"], base_components["Repair Launching cost"]]
    launchrepair.add_part(parts, "*")

    repairtotal = CollectorClass("Repairs Assembly",'USD')
    parts = [repairsmaterial, launchrepair]
    repairtotal.add_part(parts, "+")

    maintenance = CollectorClass("Maintenance",'USD')
    parts = [base_components["Probability of Material Repairs"], repairtotal]
    maintenance.add_part(parts, "*")

    # Calculate Emissions Costs

    emissiontotal = CollectorClass("Emission", 'TCO2e/m2')
    parts = [base_components["Energy system emissions"], base_components["Launching emissions"]]
    emissiontotal.add_part(parts, "+")

    emission = CollectorClass("Emission", 'USD')
    parts = [emissiontotal, base_components["Emission Cost"], base_components["Area of panels"]]
    emission.add_part(parts, "*")

    # Calculate Fuel Costs

    fuel = CollectorClass("Fuel",'USD')
    parts = [base_components["Fuel use"], base_components["Fuel cost"]]
    fuel.add_part(parts, "*")

# Calculate Energy Generated

    efficiencyboost = CollectorClass("Efficiency boost", '%')
    parts = [base_components["Efficiency boost repair"], base_components["Probability of Material Repairs"]]
    efficiencyboost.add_part(parts, "*")

    costs = efficiencyboost.get_cost().costs
    for iteration in range(costs.shape[0]):
        for year in range(costs.shape[1]):
            costs[iteration, year] = base_components["Original Efficiency"].cost_component.costs[iteration, year] * (1 - base_components["Efficiency Degradation Rate"].cost_component.costs[iteration, year])**year 

    efficiencyd = CollectorClass("Efficiency d",'%')
    efficiencyd.set_cost(costs)


    efficiencybd = CollectorClass("Efficiency bd", '%')
    parts = [efficiencyboost,efficiencyd]
    efficiencybd.add_part(parts, "*")


    efficiencyt = CollectorClass("Efficiency t", '%')
    parts = [efficiencybd,efficiencyd]
    efficiencyt.add_part(parts, "+")

    energy = CollectorClass("Energy", 'MWh')
    parts = [base_components["Installed Capacity"], efficiencyt, base_components["#days in year t"], base_components["Hours in a day"]]
    energy.add_part(parts, "*")


    
#Calculate Manufacture Costs

    manufacture_panels = CollectorClass("Manufacture Panels", 'USD/m2')
    parts = [base_components['Raw Materials'], base_components['Processing']]
    manufacture_panels.add_part(parts, "+")

    manufacture = CollectorClass("Manufacture Cost", 'USD')
    parts = [base_components['Area of panels'], manufacture_panels]
    manufacture.add_part(parts, "*")
    
    
    # Launch Costs
    launch = CollectorClass("Launch", 'USD')
    parts = [base_components['Number of Launches'], base_components['Launching cost']]
    launch.add_part(parts, "*")

     # Investment Costs
    investment = CollectorClass("Investment Costs", 'USD')
    parts = [base_components['Assembly'], launch, manufacture]
    investment.add_part(parts, "+")


    # Cost overall should be checked and looked over
    cost = CollectorClass("Cost", 'USD')
    parts = [investment, emission, fuel, maintenance]
    cost.add_part(parts, "+")

    costwithoutemission = CollectorClass("Cost", 'USD')
    parts = [investment, fuel, maintenance]
    costwithoutemission.add_part(parts, "+")

    costs = base_components["Depreciation of Assets"].get_cost().costs
    for iteration in range(costs.shape[0]):
        for year in range(costs.shape[1]):
            costs[iteration, year] = (1 + costs[iteration, year]) ** year

    base_components["Depreciation of Assets"].set_cost(costs)

    totalcost = CollectorClass("Cost Total", 'USD')
    parts = [cost, base_components["Depreciation of Assets"]]
    totalcost.add_part(parts, "/")
    totalcost.set_cost(totalcost.get_cost().get_cost_per_iteration())
    totalcost.get_cost().plot_and_save_histogram()
    print(totalcost)
    print(totalcost.get_cost().costs)
    

    totalcostwithoutemission = CollectorClass("Cost Total", 'USD')
    parts = [costwithoutemission, base_components["Depreciation of Assets"]]
    totalcostwithoutemission.add_part(parts, "/")
    totalcostwithoutemission.set_cost(totalcostwithoutemission.get_cost().get_cost_per_iteration())
    totalcostwithoutemission.get_cost().plot_and_save_histogram()
    print(totalcostwithoutemission)
    print(totalcostwithoutemission.get_cost().costs)
    
    totalenergy = CollectorClass("Energy Total", 'MWh')
    parts = [energy, base_components["Depreciation of Assets"]]
    totalenergy.add_part(parts, "/")
    totalenergy.set_cost(totalenergy.get_cost().get_cost_per_iteration())
    print(totalenergy)
    print(totalenergy.get_cost().costs)

    lcoe = CollectorClass("LCOSE", 'USD/Mwh')
    parts = [totalcost, totalenergy]
    lcoe.add_part(parts, "/")
    print(lcoe.get_cost().costs)

    lcoewithoutemission = CollectorClass("LCOSE", 'USD/Mwh')
    parts = [totalcostwithoutemission, totalenergy]
    lcoewithoutemission.add_part(parts, "/")

    print(lcoewithoutemission.get_cost().costs)