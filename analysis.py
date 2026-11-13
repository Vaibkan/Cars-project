"""
Vehicle Specs & Pricing Analysis
================================
A Pandas + SQL analysis of the UCI Automobile dataset (205 vehicles, 26 attributes).

Pipeline:
  1. Load raw data and profile data quality
  2. Clean: fix make names, handle missing values, enforce types
  3. Engineer features (power-to-weight, price per hp, combined mpg, price band)
  4. Analyse with Pandas (groupby, pivot_table, correlation, cut/qcut)
  5. Re-run key questions in SQL (SQLite) to cross-check the Pandas results
  6. Export tables and charts to outputs/

Run:  python analysis.py
"""

from pathlib import Path
import sqlite3

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "imports-85.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
OUT.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- 1. Load + profile
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "?"])
    df.columns = df.columns.str.replace("-", "_")
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(1),
            "unique": df.nunique(),
        }
    )
    return report[report["missing"] > 0].sort_values("missing", ascending=False)


# ---------------------------------------------------------------- 2. Clean
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Typos in the source data
    df["make"] = df["make"].replace({"alfa-romero": "alfa-romeo", "peugot": "peugeot"})

    # Rows without a price can't be used for pricing analysis
    df = df.dropna(subset=["price"])

    # Doors: fill from the most common value for that body style
    df["num_of_doors"] = df.groupby("body_style")["num_of_doors"].transform(
        lambda s: s.fillna(s.mode().iloc[0])
    )

    # Numeric gaps: impute with the median of the same make, then overall median
    for col in ["normalized_losses", "bore", "stroke", "horsepower", "peak_rpm"]:
        df[col] = df.groupby("make")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

    # Categorical text -> category dtype, counts -> int
    for col in ["make", "fuel_type", "aspiration", "num_of_doors", "body_style",
                "drive_wheels", "engine_location", "engine_type", "fuel_system"]:
        df[col] = df[col].astype("category")
    for col in ["horsepower", "peak_rpm", "price", "curb_weight", "engine_size"]:
        df[col] = df[col].round().astype(int)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 3. Features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["power_to_weight"] = (df["horsepower"] / df["curb_weight"] * 1000).round(1)  # hp per 1000 lb
    df["price_per_hp"] = (df["price"] / df["horsepower"]).round(1)
    df["combined_mpg"] = (0.55 * df["city_mpg"] + 0.45 * df["highway_mpg"]).round(1)
    df["price_band"] = pd.qcut(df["price"], q=4, labels=["Budget", "Mid", "Upper-mid", "Premium"])
    df["risk_level"] = pd.cut(df["symboling"], bins=[-3, -1, 1, 3],
                              labels=["Safer", "Neutral", "Riskier"], include_lowest=True)
    return df


# ---------------------------------------------------------------- 4. Pandas analysis
def analyse(df: pd.DataFrame) -> dict:
    results = {}

    results["by_make"] = (
        df.groupby("make", observed=True)
        .agg(models=("price", "size"),
             avg_price=("price", "mean"),
             avg_hp=("horsepower", "mean"),
             avg_mpg=("combined_mpg", "mean"),
             avg_price_per_hp=("price_per_hp", "mean"))
        .round(1)
        .sort_values("avg_price", ascending=False)
    )

    results["body_x_drive_price"] = df.pivot_table(
        index="body_style", columns="drive_wheels", values="price",
        aggfunc="median", observed=True
    ).round(0)

    results["price_correlations"] = (
        df.select_dtypes("number").corr()["price"]
        .drop("price").sort_values(ascending=False).round(2).to_frame("corr_with_price")
    )

    results["band_profile"] = (
        df.groupby("price_band", observed=True)[
            ["price", "horsepower", "curb_weight", "engine_size", "combined_mpg"]
        ].median().round(0)
    )

    results["risk_vs_losses"] = (
        df.groupby("risk_level", observed=True)
        .agg(vehicles=("price", "size"),
             avg_normalized_losses=("normalized_losses", "mean"),
             avg_price=("price", "mean"))
        .round(0)
    )

    # Value-for-money: cheapest horsepower within each body style
    results["best_value_per_body"] = (
        df.loc[df.groupby("body_style", observed=True)["price_per_hp"].idxmin(),
               ["body_style", "make", "price", "horsepower", "price_per_hp"]]
        .set_index("body_style")
    )
    return results


# ---------------------------------------------------------------- 5. SQL cross-check
def sql_checks(df: pd.DataFrame) -> dict:
    con = sqlite3.connect(":memory:")
    df.astype({c: str for c in df.select_dtypes("category").columns}).to_sql("vehicles", con, index=False)

    queries = {
        "sql_top_makes_by_price": """
            SELECT make, COUNT(*) AS models, ROUND(AVG(price), 1) AS avg_price
            FROM vehicles
            GROUP BY make
            HAVING COUNT(*) >= 3
            ORDER BY avg_price DESC
            LIMIT 5;""",
        "sql_fuel_aspiration": """
            SELECT fuel_type, aspiration, COUNT(*) AS vehicles,
                   ROUND(AVG(combined_mpg), 1) AS avg_mpg,
                   ROUND(AVG(price), 0) AS avg_price
            FROM vehicles
            GROUP BY fuel_type, aspiration
            ORDER BY fuel_type, aspiration;""",
        "sql_above_make_average": """
            SELECT v.make, v.body_style, v.price, ROUND(m.avg_price, 0) AS make_avg_price
            FROM vehicles v
            JOIN (SELECT make, AVG(price) AS avg_price FROM vehicles GROUP BY make) m
              ON v.make = m.make
            WHERE v.price > 1.5 * m.avg_price
            ORDER BY v.price DESC;""",
    }
    out = {name: pd.read_sql(q, con) for name, q in queries.items()}

    # Cross-check: SQL and Pandas should agree on average price per make
    pandas_avg = df.groupby("make", observed=True)["price"].mean().round(1)
    sql_avg = pd.read_sql("SELECT make, AVG(price) AS p FROM vehicles GROUP BY make", con) \
        .set_index("make")["p"].round(1)
    assert (pandas_avg.sort_index() == sql_avg.sort_index()).all(), "Pandas and SQL disagree"
    con.close()
    return out


# ---------------------------------------------------------------- 6. Charts
def charts(df: pd.DataFrame, results: dict) -> None:
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

    top = results["by_make"]["avg_price"].head(10).sort_values()
    ax = top.plot.barh(color="#2a6fdb", figsize=(7, 4.5))
    ax.set(title="Average price by make (top 10)", xlabel="Average price (USD)", ylabel="")
    plt.tight_layout(); plt.savefig(CHARTS / "avg_price_by_make.png"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fuel, grp in df.groupby("fuel_type", observed=True):
        ax.scatter(grp["engine_size"], grp["price"], label=fuel, alpha=0.7)
    ax.set(title="Engine size vs price", xlabel="Engine size (cubic inches)", ylabel="Price (USD)")
    ax.legend(title="Fuel")
    plt.tight_layout(); plt.savefig(CHARTS / "engine_size_vs_price.png"); plt.close()

    ax = results["price_correlations"]["corr_with_price"].plot.bar(color="#2a6fdb", figsize=(8, 4))
    ax.set(title="Correlation of each numeric feature with price", ylabel="Pearson r")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout(); plt.savefig(CHARTS / "price_correlations.png"); plt.close()


# ---------------------------------------------------------------- main
def main() -> None:
    raw = load_raw(DATA)
    qr = quality_report(raw)
    df = add_features(clean(raw))

    results = analyse(df)
    results.update(sql_checks(df))

    qr.to_csv(OUT / "data_quality_report.csv")
    df.to_csv(OUT / "vehicles_clean.csv", index=False)
    for name, table in results.items():
        table.to_csv(OUT / f"{name}.csv")
    charts(df, results)

    pd.set_option("display.width", 120)
    print(f"Raw rows: {len(raw)} | Clean rows: {len(df)} | Columns: {df.shape[1]}\n")
    print("Missing values in raw data:\n", qr, "\n")
    for name, table in results.items():
        print(f"--- {name} ---\n{table.head(10)}\n")


if __name__ == "__main__":
    main()
