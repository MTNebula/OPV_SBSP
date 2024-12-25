#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import logging



import cost_component as CostComponent

import utils

from config import (NUM_OF_TOTAL_YEARS, NUM_OF_ITERATIONS, DISCOUNT_RATE,
                    DISCOUNT_RATE_SD, EXCEL_PATH, SHEET_NAME, SHEET_NAMES,
                    values_of_scenarios)

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
                    cost[j, 0] = utils.generate_random_value(self.distribution, mean=self.mean, sd=self.sd, low=self.low, high=self.high, shape=self.shape, scale=self.scale, count=self.count)
                elif 'Yearly' in self.time_for_determination:
                    for i in range(1, cost.shape[1]):
                        cost[j, i] = utils.generate_random_value(self.distribution, mean=self.mean, sd=self.sd, low=self.low, high=self.high, shape=self.shape, scale=self.scale, count=self.count)
                else:
                    raise ValueError(f"Unrecognized time_for_determination value: '{self.time_for_determination}'")
        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            logging.info('Normal distribution is assumed.')
            cost[:] = utils.generate_random_value('Normal')
        except TypeError as te:
            logging.error(f"TypeError occurred: {te}")
            logging.info('Normal distribution is assumed.')
            cost[:] = utils.generate_random_value('Normal')
        except Exception as e:
            logging.error(f"Error while calculating cost for component '{self.name}': {e}")
            logging.info('Normal distribution is assumed.')
            cost[:] = utils.generate_random_value('Normal')
        return cost