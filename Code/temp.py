if __name__ == '__main__':
    # Get returns DataFrame
    train_returns_df, test_returns_df = get_returns_df()
    strategies = {
        'Random': random_weights,
        'Uniform': uniform_weights,
        'Markowitz': markowitz_weights,
        'RL': RL_weights
    }
    train_returns = train_returns_df.iloc[:, 1:].values
    train_dates = train_returns_df['Date'].values
    test_returns = test_returns_df.iloc[:, 1:].values
    test_dates = test_returns_df['Date'].values
    # Evaluate each strategy
    results = {}
    for name, func in strategies.items():
        weights = func(train_returns)
        results[name] = evaluate_portfolio(train_returns, weights)
        results[name]['dates'] = train_dates
    print("Evaluation done now plotting")
    # Plot portfolio statistics
    for metric  in ['Mean Return', 'Volatility', 'Sharpe', 'Max Drawdown', 'Cumulative Return']:
        plot_portfolio_statistics(results, metric, type='bar')
    for metric in ['Portfolio Value', 'Portfolio Returns']:
        plot_portfolio_statistics(results, metric, type='line')