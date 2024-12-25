#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

from config import (NUM_OF_TOTAL_YEARS, NUM_OF_ITERATIONS, DISCOUNT_RATE,
                    DISCOUNT_RATE_SD, EXCEL_PATH, SHEET_NAME, SHEET_NAMES,
                    values_of_scenarios)


class CostComponent:
    def __init__(self, cost, name, unit):
        self.name = name
        self.costs = cost
        self.unit = unit
        self.is_discounted_cost = False
        self.costs_per_iteration = None
        try: 
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