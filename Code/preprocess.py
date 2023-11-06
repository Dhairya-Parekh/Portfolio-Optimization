#%%
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
# Split the data into train, validation and test sets
train_fraction = 0.6
validation_fraction = 0.2
test_fraction = 0.2

train_cutoff = int(len(closing_prices) * train_fraction)
validation_cutoff = int(len(closing_prices) * (train_fraction + validation_fraction))

train_data = closing_prices.iloc[:train_cutoff]
validation_data = closing_prices.iloc[train_cutoff:validation_cutoff]
test_data = closing_prices.iloc[validation_cutoff:]

train_data.to_csv('../Data/train.csv', index=False)
validation_data.to_csv('../Data/validation.csv', index=False)
test_data.to_csv('../Data/test.csv', index=False)