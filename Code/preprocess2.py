import pandas as pd
import numpy as np
from datetime import datetime as dt

BASE_PATH = "../Data/New-Data/raw/"

if __name__=='__main__':
    df = pd.read_csv(BASE_PATH+'^NSEI.csv')
    ignore_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.drop(columns=ignore_cols+['Adj Close'])
    df = df.set_index('Date')
    assets1 = ['^NSEI', 'RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS',\
            'ICICIBANK.NS', 'INFY.NS', 'SBIN.NS', 'BAJFINANCE.NS',\
                'INDHOTEL.NS', 'ONGC.NS', 'NAUKRI.NS', 'INR=X', 'JETAIRWAYS.NS', 'IDEA.NS']
    assets2 = ['GOLD-MMIc1.MCX', 'SILVER-MSDc1.MCX', 'IND-10Y.BOND', 'NIFTY.FUT']

    for asset in assets1:
        temp_df = pd.read_csv(BASE_PATH+asset+'.csv')
        temp_df = temp_df.drop(columns=ignore_cols)
        temp_df = temp_df.rename(columns={'Date': "Date", 'Adj Close': asset})
        temp_df = temp_df.set_index('Date')
        df = df.join(temp_df)

    for asset in assets2:
        temp_df = pd.read_csv(BASE_PATH+asset+'.csv')
        cols = temp_df.columns
        cols = [x for x in cols if x != 'Price' and x != 'Date']
        temp_df = temp_df.drop(columns=cols)
        temp_df['Price'] = temp_df['Price'].astype(str).apply(lambda x: float(x.replace(',','')))
        temp_df['Date'] = temp_df['Date'].apply(lambda x: dt.strptime(x,'%b %d, %Y').strftime('%Y-%m-%d'))
        temp_df = temp_df.rename(columns={'Date': "Date", 'Price': asset})
        temp_df = temp_df.set_index('Date')
        df = df.join(temp_df)
    
    # df['CASH'] = np.ones((len(df.index)))
    df.info()
    # df[:len(df)*4//5].to_csv(BASE_PATH+'../train.csv')
    # df[len(df)*4//5:].to_csv(BASE_PATH+'../test.csv')
    # Split the data into train, validation and test sets
    train_fraction = 0.6
    validation_fraction = 0.2
    test_fraction = 0.18

    train_cutoff = int(len(df) * train_fraction)
    validation_cutoff = int(len(df) * (train_fraction + validation_fraction))
    test_cutoff = int(len(df) * (train_fraction + validation_fraction + test_fraction))

    train_data = df.iloc[:train_cutoff]
    validation_data = df.iloc[train_cutoff:validation_cutoff]
    test_data = df.iloc[validation_cutoff:test_cutoff]

    train_data.to_csv(BASE_PATH+'../train.csv')
    validation_data.to_csv(BASE_PATH+'../validation.csv')
    test_data.to_csv(BASE_PATH+'../test.csv')