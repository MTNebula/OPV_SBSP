#!/usr/bin/env python3

import unittest
import numpy as np
import os
from cost_component import CostComponent  # Assuming the class is in a file named cost_component.py

class TestCostComponent(unittest.TestCase):

    def setUp(self):
        self.costs = np.array([[100, 200, 300], [150, 250, 350]])
        self.name = "TestCost"
        self.unit = "USD"
        self.cost_component = CostComponent(self.costs, self.name, self.unit)

    def test_initialization(self):
        self.assertEqual(self.cost_component.name, self.name)
        self.assertTrue(np.array_equal(self.cost_component.costs, self.costs))
        self.assertEqual(self.cost_component.unit, self.unit)
        self.assertFalse(self.cost_component.is_discounted_cost)

    def test_mean(self):
        expected_mean = np.mean(self.costs[self.costs != 0])
        self.assertAlmostEqual(self.cost_component.mean, expected_mean)

    def test_sd(self):
        expected_sd = np.std(self.costs[self.costs != 0])
        self.assertAlmostEqual(self.cost_component.sd, expected_sd)

    def test_min(self):
        expected_min = np.min(self.costs[self.costs != 0])
        self.assertAlmostEqual(self.cost_component.min, expected_min)

    def test_max(self):
        expected_max = np.max(self.costs[self.costs != 0])
        self.assertAlmostEqual(self.cost_component.max, expected_max)

    def test_addition(self):
        other_costs = np.array([[50, 100, 150], [75, 125, 175]])
        other_component = CostComponent(other_costs, "OtherCost", "USD")
        result = self.cost_component + other_component
        expected_costs = self.costs + other_costs
        self.assertTrue(np.array_equal(result.costs, expected_costs))

    def test_subtraction(self):
        other_costs = np.array([[50, 100, 150], [75, 125, 175]])
        other_component = CostComponent(other_costs, "OtherCost", "USD")
        result = self.cost_component - other_component
        expected_costs = self.costs - other_costs
        self.assertTrue(np.array_equal(result.costs, expected_costs))

    def test_multiplication(self):
        other_costs = np.array([[2, 2, 2], [2, 2, 2]])
        other_component = CostComponent(other_costs, "OtherCost", "USD")
        result = self.cost_component * other_component
        expected_costs = self.costs * other_costs
        self.assertTrue(np.array_equal(result.costs, expected_costs))

    def test_division(self):
        other_costs = np.array([[2, 2, 2], [2, 2, 2]])
        other_component = CostComponent(other_costs, "OtherCost", "USD")
        result = self.cost_component / other_component
        expected_costs = self.costs / other_costs
        self.assertTrue(np.array_equal(result.costs, expected_costs))

    def test_apply_discounted_rate(self):
        discount_rate = 0.1
        expected_costs = np.zeros(self.costs.shape)
        for i in range(self.costs.shape[0]):
            for j in range(self.costs.shape[1]):
                expected_costs[i, j] = self.costs[i, j] / (1 + discount_rate) ** j
        self.cost_component.apply_discounted_rate(discount_rate)
        self.assertTrue(np.array_equal(self.cost_component.costs, expected_costs))
        self.assertTrue(self.cost_component.is_discounted_cost)

    def test_get_cost_per_iteration(self):
        expected_costs_per_iteration = np.sum(self.costs, axis=1)
        self.assertTrue(np.array_equal(self.cost_component.get_cost_per_iteration(), expected_costs_per_iteration))

    def test_get_non_zero_cost(self):
        expected_non_zero_costs = self.costs[self.costs != 0]
        self.assertTrue(np.array_equal(self.cost_component.get_non_zero_cost(), expected_non_zero_costs))

    def test_plot_and_save_histogram(self):
        # This test will check if the plot is saved without errors
        self.cost_component.plot_and_save_histogram(folder='test_plots')
        self.assertTrue(os.path.exists('test_plots'))
        self.assertTrue(os.path.exists(f'test_plots/{self.name}.png'))

    def test_plot_and_save_histogram_per_iteration(self):
        # This test will check if the plot is saved without errors
        self.cost_component.plot_and_save_histogram_per_iteration(folder='test_plots_per_iteration')
        self.assertTrue(os.path.exists('test_plots_per_iteration'))
        self.assertTrue(os.path.exists(f'test_plots_per_iteration/{self.name}.png'))

if __name__ == '__main__':
    unittest.main()
