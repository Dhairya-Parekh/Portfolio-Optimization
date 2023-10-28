#%%
import numpy as np
import pandas as pd
#%%
closing_prices = pd.read_csv('../Data/data.csv')
# Make sure that the data is float (it might be string with commas) for all columns except the date
def convert_to_float(value):
    try:
        return float(value.replace(',', ''))
    except AttributeError:
        return value

for column in closing_prices.columns:
    if column != 'Date':
        closing_prices[column] = closing_prices[column].apply(convert_to_float)
#%%
# Create a new column for the date in datetime format
closing_prices['Date'] = pd.to_datetime(closing_prices['Date'])
#%%
# Split the data into train and test sets
test_size = 0.2
test_split = int(len(closing_prices) * test_size)
train_set = closing_prices.iloc[test_split:]
test_set = closing_prices.iloc[:test_split]
#%%
# Save the train and test sets
train_set.to_csv('../Data/train.csv', index=False)
test_set.to_csv('../Data/test.csv', index=False)
# %%
