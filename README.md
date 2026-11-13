# Vehicle Specs & Pricing Analysis (Pandas + SQL)

A data-cleaning and analysis project on the [UCI Automobile dataset](https://archive.ics.uci.edu/dataset/10/automobile): 205 vehicles from 22 manufacturers, with 26 attributes covering specs, insurance risk and price.

The goal is to take messy vehicle data, clean it into a reliable table, and answer practical questions: **what drives a vehicle's price, how makes and body styles compare, and where the best value is**. Key results are re-checked in SQL so the Pandas and SQL answers have to match.

## What the pipeline does

| Step | What happens | Pandas / SQL used |
|---|---|---|
| 1. Profile | Data-quality report: missing values, types, unique counts | `isna`, `nunique`, `dtypes` |
| 2. Clean | Fix make-name typos (`alfa-romero`, `peugot`), drop rows with no price, fill missing door counts by body style, fill numeric gaps with the median for the same make, set category and integer types | `groupby().transform`, `fillna`, `astype("category")` |
| 3. Engineer | Power-to-weight, price per horsepower, combined MPG, price band (quartiles), insurance risk level | `qcut`, `cut`, vectorised maths |
| 4. Analyse | Averages by make, median price by body style × drive type, price correlations, price band profiles, risk vs insurance losses, best value per body style | `groupby().agg`, `pivot_table`, `corr`, `idxmin` |
| 5. SQL check | Loads the clean data into SQLite and runs `GROUP BY`/`HAVING`, a subquery `JOIN` and segment queries. It asserts that SQL and Pandas give the same average price per make | `to_sql`, `read_sql`, SQLite |
| 6. Export | Clean dataset, every result table as CSV, three charts | `to_csv`, matplotlib |

## Key findings

- **Data quality:** 20% of `normalized_losses` was missing. Price, bore, stroke, horsepower and RPM each had 1–2% missing. After cleaning, 201 of the 205 rows are usable.
- **Engine size drives price.** Engine size has the strongest link to price (r = 0.87), then curb weight (0.83) and horsepower (0.81). Fuel economy moves the other way: cheaper cars use less fuel.
- **Drive type matters a lot.** Rear-wheel-drive sedans have a median price of about $18.4k, against about $8.6k for front-wheel-drive sedans. That's more than double.
- **Price bands are clear.** The premium quartile (median $20.3k) has about twice the horsepower of the budget quartile (141 vs 68 hp) and gets about 12 fewer MPG.
- **Insurance risk:** Cars rated "riskier" have higher average insurance losses (142) than "safer" cars (88).
- **Best value per horsepower:** Toyota (hardtop, $73/hp), Subaru (hatchback, $74/hp) and Mitsubishi (sedan, $79/hp).
- **Diesel vs gas:** Non-turbo diesels average about 40 MPG, against about 28 MPG for non-turbo gas cars.

## Charts

![Average price by make](outputs/charts/avg_price_by_make.png)
![Engine size vs price](outputs/charts/engine_size_vs_price.png)
![Price correlations](outputs/charts/price_correlations.png)

## Run it

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `outputs/` (CSV tables and charts).

## Project structure

```
├── analysis.py            # full pipeline: load → clean → features → analysis → SQL check → export
├── data/imports-85.csv    # raw UCI Automobile dataset
├── outputs/               # clean data, result tables (CSV), charts (PNG)
└── requirements.txt
```

## Ideas to extend

- Join with a parts or VIN dataset to analyse parts fitment across makes and models
- Build a price prediction model (e.g. linear regression on engine size, weight and drive type)
- Swap SQLite for PostgreSQL
