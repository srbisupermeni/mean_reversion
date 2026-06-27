# NVDA mean-reversion prototype

This project tests whether an unusually large daily move tends to reverse. It computes daily adjusted-close returns over a rolling 20-session window:

`return_z = (today_return - rolling_mean_return) / rolling_std_return`

Default rules:

- `return_z <= -2`: target a long position.
- `return_z >= +2`: target a short position.
- Exit when the z-score returns within `+/-0.5` of zero.
- Calculate the signal after today's close and apply it to the next close-to-close return.
- Charge 5 basis points for each unit of turnover.

The output also includes `price_z`, the close's deviation from its rolling 20-day mean, so the two common definitions can be compared.
Historical data comes from Yahoo Finance. If it temporarily throttles requests, wait a few minutes and rerun the command.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python mean_reversion.py --ticker NVDA --start 2020-01-01
```

Compare a price-level signal:

```powershell
python mean_reversion.py --ticker NVDA --start 2020-01-01 --signal price_z
```

Run the offline unit tests:

```powershell
python -m unittest -v
```

This is a research backtest, not investment advice. A single volatile trending stock is not necessarily mean-reverting; validate on out-of-sample data and include realistic execution assumptions before considering live trading.
# mean_reversion
