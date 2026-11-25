"""Reproduce the main county-level results from the team's semester study."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
MEASURES = {
    "SLEEP": "sleep",
    "OBESITY": "obesity",
    "DIABETES": "diabetes",
    "STROKE": "stroke",
    "CHD": "chd",
    "CSMOKING": "smoking",
    "LPA": "inactive",
    "ACCESS2": "uninsured",
    "DEPRESSION": "depression",
}
OUTCOMES = {
    "obesity": "Obesity",
    "diabetes": "Diagnosed diabetes",
    "stroke": "Stroke",
    "chd": "Coronary heart disease",
}
REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
COVARIATES = ["smoking", "inactive", "uninsured", "depression"]


def county_sample(path):
    columns = ["year", "locationid", "stateabbr", "measureid", "data_value", "totalpopulation"]
    raw = pd.read_csv(path, dtype={"locationid": str})
    raw.columns = raw.columns.str.strip().str.lower()
    if "data_value_type" in raw:
        raw = raw[raw["data_value_type"] == "Age-adjusted prevalence"]
    raw = raw[columns].copy()
    raw["locationid"] = raw["locationid"].str.zfill(5)
    raw["measure"] = raw["measureid"].map(MEASURES)
    is_sleep = raw["measure"].eq("sleep")
    raw = raw[raw["measure"].notna() & ((is_sleep & raw.year.eq(2022)) | (~is_sleep & raw.year.eq(2023)))]
    state_to_region = {state: region for region, states in REGIONS.items() for state in states.split()}
    raw = raw[raw["stateabbr"].isin(state_to_region)]
    county = raw.pivot(index=["locationid", "stateabbr"], columns="measure", values="data_value")
    population = raw[raw["measure"] == "obesity"].set_index(["locationid", "stateabbr"])["totalpopulation"]
    county["log_pop"] = np.log(population.where(population.gt(0)))
    county = county.reset_index()
    county["region"] = county["stateabbr"].map(state_to_region)
    return county.dropna(subset=["sleep", *OUTCOMES, *COVARIATES, "log_pop", "region"]).copy()


def analyze(county):
    county["quartile"] = pd.qcut(county["sleep"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grouped = county.groupby("quartile", observed=True)
    quartiles = grouped[list(OUTCOMES)].mean()
    quartiles.insert(0, "counties", grouped.size())

    adjusted = []
    controls = " + ".join([*COVARIATES, "log_pop", "C(region)"])
    for outcome, label in OUTCOMES.items():
        model = smf.ols(f"{outcome} ~ sleep + {controls}", data=county).fit(cov_type="HC3")
        low, high = model.conf_int().loc["sleep"]
        adjusted.append({"outcome": label, "estimate": model.params["sleep"],
                         "ci_low": low, "ci_high": high, "counties": int(model.nobs)})
    return quartiles, pd.DataFrame(adjusted)


def plot_quartiles(quartiles, path):
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.3), layout="constrained")
    figure.suptitle("County health prevalence by short-sleep quartile", fontweight="bold")
    figure.supxlabel("2022 short-sleep prevalence quartile (Q1 lowest, Q4 highest)")
    figure.supylabel("Mean 2023 age-adjusted prevalence (%)")
    for ax, (outcome, label) in zip(axes.flat, OUTCOMES.items()):
        ax.plot(quartiles.index.astype(str), quartiles[outcome], "o-", color="#146877", lw=2)
        ax.set_title(label, loc="left")
        ax.grid(axis="y", alpha=.25)
        ax.margins(x=.17, y=.30)
        for index, value in enumerate(quartiles[outcome]):
            ax.annotate(f"{value:.2f}", (index, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_adjusted(adjusted, path):
    figure, ax = plt.subplots(figsize=(8.4, 4.25), layout="constrained")
    ax.errorbar(adjusted["estimate"], adjusted["outcome"], fmt="o", color="#146877", capsize=4,
                xerr=[adjusted["estimate"] - adjusted["ci_low"], adjusted["ci_high"] - adjusted["estimate"]])
    ax.axvline(0, color="#606871", linestyle="--", lw=1)
    ax.set_title("Adjusted county-level associations with short sleep", loc="left", fontweight="bold")
    ax.set_xlabel("Outcome prevalence change (pp) per +1 pp in short sleep\nPoints show estimates. Bars show 95% HC3 confidence intervals.")
    ax.set_ylabel("2023 health outcome")
    ax.grid(axis="x", alpha=.25)
    ax.invert_yaxis()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "cdc-places-2025-extract.csv")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    results = args.output / "results"
    charts = args.output / "visualizations"
    results.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    county = county_sample(args.input)
    quartiles, adjusted = analyze(county)
    quartiles.to_csv(results / "quartiles.csv", float_format="%.6f")
    adjusted.to_csv(results / "adjusted.csv", index=False, float_format="%.6f")
    plot_quartiles(quartiles, charts / "short-sleep-quartiles.png")
    plot_adjusted(adjusted, charts / "adjusted-associations.png")
    print(f"Counties and county equivalents: {len(county):,}")
    print(quartiles.round(2).to_string())
    print(adjusted.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
