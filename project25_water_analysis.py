"""
Project 25: Water Usage & Conservation Analysis
Run:  python project25_water_analysis.py
Needs: pandas, numpy, matplotlib
Outputs: printed report + CSV tables + charts in ./output/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = "25_water_consumption_expanded.csv"   # or the original 25_water_consumption.csv
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- Task 1
section("TASK 1: Load and inspect")
df = pd.read_csv(DATA_FILE)
print("Shape (rows, columns):", df.shape)
print("Columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------------- Task 2
section("TASK 2: Missing values and duplicate Meter_Record_ID")
print("Missing values per column:")
print(df.isna().sum())
dup_ids = df[df["Meter_Record_ID"].duplicated(keep=False)].sort_values("Meter_Record_ID")
n_dup_extra = df["Meter_Record_ID"].duplicated().sum()
print(f"\nDuplicate Meter_Record_ID entries (extra copies): {n_dup_extra}")
print(f"Records involved (incl. first copies): {len(dup_ids)}")
print("Are the duplicates exact copies of the whole row?",
      df.duplicated().sum() == n_dup_extra)
print(dup_ids.head(10))

# Remove duplicates: a meter record ID must be unique, and these are exact copies,
# so keeping them would double count consumption.
df = df.drop_duplicates(subset="Meter_Record_ID", keep="first").reset_index(drop=True)
print("\nShape after dropping duplicates:", df.shape)

# ---------------------------------------------------------------- Task 3
section("TASK 3: Handle missing Water_Usage_Liters")
n_missing = df["Water_Usage_Liters"].isna().sum()
print("Missing usage values:", n_missing)
print("Mean:", round(df["Water_Usage_Liters"].mean(), 1),
      "| Median:", round(df["Water_Usage_Liters"].median(), 1))
# Method: fill with the MEDIAN usage of the same Household_Type.
# Why: median is not distorted by extreme readings (unlike the mean), and grouping by
# household type keeps the fill value relevant to the kind of household.
# Dropping rows would lose the other valid columns; a fill keeps totals comparable.
df["Water_Usage_Liters"] = df["Water_Usage_Liters"].fillna(
    df.groupby("Household_Type")["Water_Usage_Liters"].transform("median"))
print("Missing after imputation:", df["Water_Usage_Liters"].isna().sum())
print("Method used: group median by Household_Type")

# ---------------------------------------------------------------- Task 4
section("TASK 4: Reading_Date to datetime, extract month and weekday")
df["Reading_Date"] = pd.to_datetime(df["Reading_Date"])
df["Year"] = df["Reading_Date"].dt.year
df["Month"] = df["Reading_Date"].dt.month
df["Month_Name"] = df["Reading_Date"].dt.month_name()
df["Year_Month"] = df["Reading_Date"].dt.to_period("M")
df["Weekday"] = df["Reading_Date"].dt.day_name()
print(df[["Reading_Date", "Year", "Month", "Month_Name", "Weekday"]].head())
print("Date range:", df["Reading_Date"].min().date(), "to", df["Reading_Date"].max().date())

# ---------------------------------------------------------------- Task 5
section("TASK 5: Total and average consumption per household")
total_all = df["Water_Usage_Liters"].sum()
print(f"Total water consumption (all records): {total_all:,.1f} liters")
hh = df.groupby("Household_ID")["Water_Usage_Liters"].agg(
    Total_Liters="sum", Avg_Liters="mean", Readings="count")
print(f"Households: {len(hh)}")
print(f"Overall average per record: {df['Water_Usage_Liters'].mean():,.1f} L")
print(f"Average of household totals: {hh['Total_Liters'].mean():,.1f} L")
print(f"Average of household averages: {hh['Avg_Liters'].mean():,.1f} L")
print(hh.head())
hh.round(1).to_csv(f"{OUT_DIR}/household_totals_averages.csv")

# ---------------------------------------------------------------- Task 6
section("TASK 6: Month-to-month change per household")
monthly_hh = (df.groupby(["Household_ID", "Year_Month"])["Water_Usage_Liters"]
                .sum().reset_index().sort_values(["Household_ID", "Year_Month"]))
monthly_hh["Change_Liters"] = monthly_hh.groupby("Household_ID")["Water_Usage_Liters"].diff()
monthly_hh["Change_Pct"] = monthly_hh.groupby("Household_ID")["Water_Usage_Liters"].pct_change() * 100
print("Note: change is measured between a household's consecutive months WITH readings;")
print("a household is not read every month, so gaps between months can exist.\n")
print(monthly_hh.head(12).round({"Water_Usage_Liters": 1, "Change_Liters": 1, "Change_Pct": 1}))
valid_changes = monthly_hh["Change_Liters"].notna().sum()
print(f"\nHousehold month-pairs with a change value: {valid_changes}")
print("Average change between consecutive readings:",
      round(monthly_hh["Change_Liters"].mean(), 1), "L")
print("Households with the largest single increase:")
print(monthly_hh.nlargest(5, "Change_Liters").round({"Water_Usage_Liters": 1, "Change_Liters": 1, "Change_Pct": 1}))
monthly_hh.round({"Water_Usage_Liters": 1, "Change_Liters": 1, "Change_Pct": 1}).to_csv(f"{OUT_DIR}/household_month_to_month_change.csv", index=False)

# ---------------------------------------------------------------- Task 7
section("TASK 7: Rank household types by total usage")
ht = (df.groupby("Household_Type")["Water_Usage_Liters"]
        .agg(Total_Liters="sum", Avg_Liters="mean", Records="count")
        .sort_values("Total_Liters", ascending=False))
ht["Rank"] = range(1, len(ht) + 1)
print(ht.round(1))
ht.round(1).to_csv(f"{OUT_DIR}/household_type_ranking.csv")

# ---------------------------------------------------------------- Task 8
section("TASK 8: Average consumption by building type and zone")
bt = df.groupby("Building_Type")["Water_Usage_Liters"].agg(["mean", "median", "count"]).round(1)
zn = df.groupby("Zone")["Water_Usage_Liters"].agg(["mean", "median", "count"]).round(1)
print("By building type:\n", bt.sort_values("mean", ascending=False))
print("\nBy zone:\n", zn.sort_values("mean", ascending=False))
spread_bt = bt["mean"].max() - bt["mean"].min()
spread_zn = zn["mean"].max() - zn["mean"].min()
print(f"\nSpread between highest and lowest average: building type {spread_bt:,.0f} L, "
      f"zone {spread_zn:,.0f} L")
bt.to_csv(f"{OUT_DIR}/avg_by_building_type.csv")
zn.to_csv(f"{OUT_DIR}/avg_by_zone.csv")

# ---------------------------------------------------------------- Task 9
section("TASK 9: Top 10 households by total consumption")
top10 = hh.sort_values("Total_Liters", ascending=False).head(10)
print(top10.round(1))
top10.round(1).to_csv(f"{OUT_DIR}/top10_households.csv")

# ---------------------------------------------------------------- Task 10
section("TASK 10: Month with highest overall usage")
by_ym = df.groupby("Year_Month")["Water_Usage_Liters"].agg(["sum", "mean", "count"])
by_cal = df.groupby(["Month", "Month_Name"])["Water_Usage_Liters"].agg(["sum", "mean", "count"])
peak_ym = by_ym["sum"].idxmax()
peak_cal = by_cal["sum"].idxmax()
print(f"Highest total usage, year-month: {peak_ym} ({by_ym['sum'].max():,.1f} L, "
      f"{int(by_ym.loc[peak_ym, 'count'])} records)")
print("Caution: monthly totals rise and fall with the NUMBER of readings taken that month, "
      "so compare the per-record average as well.")
print(f"Highest total usage, calendar month (all years pooled): {peak_cal[1]} "
      f"({by_cal['sum'].max():,.1f} L)")
print(f"Highest AVERAGE per record, year-month: {by_ym['mean'].idxmax()} "
      f"({by_ym['mean'].max():,.1f} L)")
print("\nCalendar-month table:")
print(by_cal.round(1))
by_ym.round(1).to_csv(f"{OUT_DIR}/monthly_trend.csv")

# ---------------------------------------------------------------- Task 11
section("TASK 11: Dictionary of zone totals and set of unique zones")
zone_totals = {}
for zone, liters in zip(df["Zone"], df["Water_Usage_Liters"]):
    zone_totals[zone] = zone_totals.get(zone, 0) + liters
zone_totals = {k: round(v, 1) for k, v in sorted(zone_totals.items())}
unique_zones = set(df["Zone"])
print("Zone-wise totals (dict):")
for z, v in zone_totals.items():
    print(f"  {z}: {v:,.1f} L")
print("Unique zones (set):", sorted(unique_zones), "| count:", len(unique_zones))
zone_table = pd.DataFrame({"Total_Liters": zone_totals})
zone_table["Share_%"] = (zone_table["Total_Liters"] / zone_table["Total_Liters"].sum() * 100).round(1)
zone_table.to_csv(f"{OUT_DIR}/zone_totals.csv")

# ---------------------------------------------------------------- Task 12
section("TASK 12: Consumption classification function")
# Thresholds are taken from the data: bottom quarter = Low, top quarter = High.
LOW_LIMIT = float(np.percentile(df["Water_Usage_Liters"], 25))
HIGH_LIMIT = float(np.percentile(df["Water_Usage_Liters"], 75))


def classify_consumption(liters, low=LOW_LIMIT, high=HIGH_LIMIT):
    """Return 'Low' (< low), 'Normal' (low to high inclusive) or 'High' (> high)."""
    if liters < low:
        return "Low"
    if liters > high:
        return "High"
    return "Normal"


print(f"Low    : below {LOW_LIMIT:,.1f} L  (25th percentile)")
print(f"Normal : {LOW_LIMIT:,.1f} to {HIGH_LIMIT:,.1f} L")
print(f"High   : above {HIGH_LIMIT:,.1f} L  (75th percentile)")
df["Usage_Category"] = df["Water_Usage_Liters"].apply(classify_consumption)
print("\nRecords per category:")
print(df["Usage_Category"].value_counts())
print("\nCategory share by Household_Type (%):")
print((pd.crosstab(df["Household_Type"], df["Usage_Category"], normalize="index") * 100).round(1))

# ---------------------------------------------------------------- Task 13
section("TASK 13: NumPy statistics")
arr = df["Water_Usage_Liters"].to_numpy()
print(f"Mean              : {np.mean(arr):,.2f} L")
print(f"Median            : {np.median(arr):,.2f} L")
print(f"Variance (sample) : {np.var(arr, ddof=1):,.2f}")
print(f"Std dev (sample)  : {np.std(arr, ddof=1):,.2f} L")
print(f"Variance (pop.)   : {np.var(arr):,.2f}")
print(f"Std dev (pop.)    : {np.std(arr):,.2f} L")
print(f"Min / Max         : {arr.min():,.1f} / {arr.max():,.1f} L")

# ---------------------------------------------------------------- Task 14
section("TASK 14: Outliers by IQR, compare household types")
q1, q3 = np.percentile(arr, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1:,.1f}  Q3={q3:,.1f}  IQR={iqr:,.1f}")
print(f"Outlier fences: lower={lower:,.1f}  upper={upper:,.1f}")
outliers = df[(df["Water_Usage_Liters"] > upper) | (df["Water_Usage_Liters"] < lower)]
print(f"Records outside the fences: {len(outliers)}")
if len(outliers) == 0:
    print("No statistical outliers: usage is spread fairly evenly between the minimum and "
          "maximum, so the IQR fence lies beyond the largest reading.")
    p95 = np.percentile(arr, 95)
    heavy = df[df["Water_Usage_Liters"] > p95].sort_values("Water_Usage_Liters", ascending=False)
    print(f"\nFallback: 'very high' records above the 95th percentile ({p95:,.1f} L): {len(heavy)}")
    print(heavy[["Meter_Record_ID", "Household_ID", "Household_Type", "Zone",
                 "Water_Usage_Liters"]].head(10))
    heavy.to_csv(f"{OUT_DIR}/very_high_records_p95.csv", index=False)
else:
    print(outliers.sort_values("Water_Usage_Liters", ascending=False).head(10))
    outliers.to_csv(f"{OUT_DIR}/iqr_outliers.csv", index=False)

cmp_ht = (df.groupby("Household_Type")["Water_Usage_Liters"]
            .agg(Mean="mean", Median="median", Std="std", Max="max").round(1))
cmp_ht["Outliers"] = outliers.groupby("Household_Type").size().reindex(cmp_ht.index).fillna(0).astype(int)
print("\nAverage usage by household type:")
print(cmp_ht.sort_values("Mean", ascending=False))
cmp_ht.to_csv(f"{OUT_DIR}/household_type_comparison.csv")

# ---------------------------------------------------------------- Task 15
section("TASK 15: WaterMeter class and recommendations")


class WaterMeter:
    """A water meter identified by ID and linked to a household type."""

    def __init__(self, meter_id, household_type):
        self.meter_id = meter_id
        self.household_type = household_type

    def __repr__(self):
        return f"WaterMeter(meter_id='{self.meter_id}', household_type='{self.household_type}')"


meters = [WaterMeter(r.Meter_Record_ID, r.Household_Type)
          for r in df.head(5).itertuples()]
for m in meters:
    print(m)

top_zone = zone_table["Total_Liters"].idxmax()
top_ht = ht.index[0]
top_bt = bt["mean"].idxmax()
high_share = (df["Usage_Category"] == "High").mean() * 100
top10_share = top10["Total_Liters"].sum() / total_all * 100
print("\nFive water-conservation recommendations (based on the findings above):")
recs = [
    f"1. Target the heaviest households first: the top 10 households account for "
    f"{top10_share:.1f}% of all recorded usage. Send them usage reports and offer "
    f"free leak checks and efficient-fixture retrofits.",
    f"2. Run a High-usage alert: readings above {HIGH_LIMIT:,.0f} L (the top quarter, "
    f"{high_share:.0f}% of records) should trigger a notice and a follow-up check "
    f"for leaks or wasteful use.",
    f"3. Focus on {top_zone}, the zone with the largest total consumption (largely because "
    f"it has the most readings), with a pressure-management and leak-detection survey "
    f"of its pipes and meters; also check {zn['mean'].idxmax()}, which has the highest "
    f"average per reading.",
    f"4. Prioritise {top_bt} buildings (highest average usage) and {top_ht} households "
    f"(highest total) for efficiency audits: low-flow taps, dual-flush toilets, "
    f"and water-efficient appliances.",
    f"5. Time campaigns around the peak month ({peak_ym}) with awareness drives on "
    f"outdoor and seasonal use, and add tiered pricing so high users pay more per liter.",
]
for r in recs:
    print(r)
with open(f"{OUT_DIR}/recommendations.txt", "w") as f:
    f.write("\n".join(recs))

# ---------------------------------------------------------------- Charts
section("Charts")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(by_ym.index.astype(str), by_ym["sum"], marker="o")
ax.set_title("Monthly total water usage")
ax.set_ylabel("Liters")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/monthly_trend.png", dpi=130)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
zone_table["Total_Liters"].plot.bar(ax=axes[0], color="#2a7fb8")
axes[0].set_title("Total usage by zone")
axes[0].set_ylabel("Liters")
ht["Total_Liters"].plot.bar(ax=axes[1], color="#3a9a6a")
axes[1].set_title("Total usage by household type")
bt["mean"].plot.bar(ax=axes[2], color="#c87a2a")
axes[2].set_title("Average usage by building type")
axes[2].set_ylabel("Liters")
for a in axes:
    a.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparisons.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.boxplot(arr, orientation="horizontal")
ax.set_title("Water usage per record (boxplot, IQR method)")
ax.set_xlabel("Liters")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/usage_boxplot.png", dpi=130)
plt.close()
print("Saved charts and tables to ./" + OUT_DIR + "/")
