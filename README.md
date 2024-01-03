# Market Regime and Sentiment-Mitigated Algorithmic Trading Strategy

## Overview
This project integrates advanced techniques in U.S. equities trading, combining pairs trading, Hidden Markov Models (HMM) for market regime classification, sentiment analysis, and robust risk management.

## Key Components
- **Pairs Trading Strategy**: Stock pair selection through unsupervised machine learning.
- **Regime Classification with HMM**: Identifies market regimes and allocates assets.
- **Sentiment Analysis**: NLP techniques to analyze news streams.
- **Risk Management**: Includes Triple Barrier method and Equal Contribution to Risk approach.

## Backtesting Results
- **In Sample (Jan 1, 2017, to June 1, 2023)**: 17.28% profit, Sharpe Ratio of 1.248, max drawdown 6.4%.
- **Out of Sample (Jan 1, 2022, to Jan 1, 2023)**: 31.99% profit, Sharpe Ratio of 1.476, max drawdown 9%.
- **Stress Test - March 2020**: 1.64% profit, Sharpe Ratio of .937, max drawdown 4.5%.
- **Blind OOS (Jan 1, 2023, to Apr 1, 2023)**: 11.47% profit, Sharpe Ratio of 1.529, max drawdown 8.2%.
- **Live Paper Trading Result**: 0.05% profit.
For detailed results please check the report
## Installation and Usage
Python and QuantConnect Platform - see the [QuantConnect](https://www.quantconnect.com/) for details.

## Contributing
This project was developed as part of an academic final project for the Algorithmic Trading Course. Contributions were made solely by Dev Patel, Andres Caicedo, Cindy Chiu, and Nathan Luksik.

## License
This project is licensed under the MIT License - see the [MIT License](https://opensource.org/licenses/MIT) for details.

## Acknowledgement
Gratitude to David Ye and the course staff for their invaluable guidance.
