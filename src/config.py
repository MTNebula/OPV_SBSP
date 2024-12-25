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