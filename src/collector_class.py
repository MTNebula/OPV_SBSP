import copy
import numpy as np

from base_component import BaseComponent
from cost_component import CostComponent

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
    