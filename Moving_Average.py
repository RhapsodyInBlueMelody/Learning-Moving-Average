import numpy as np
import csv
import matplotlib.pyplot as plt

# --- Load CSV ---
years = []
values = []
country = None

with open("Indonesia-GDP-2020-2025.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not country:
            country = row["Country"]
        years.append(int(row["Year"]))
        values.append(float(row["GDP"]))

n = 3  # window size

# --- Simple Moving Average ---
sma = []
for i in range(len(values)):
    if i+1 < n:
        sma.append(None)
    else:
        window = values[i-n+1:i+1]
        avg = sum(window) / n
        sma.append(avg)

# --- Weighted Moving Average ---
wma = []
weights = np.arange(1, n+1)
for i in range(len(values)):
    if i+1 < n:
        wma.append(None)
    else:
        window = values[i-n+1:i+1]
        weighted_avg = sum(x*w for x, w in zip(window, weights)) / weights.sum()
        wma.append(weighted_avg)

# --- Exponential Moving Average ---
ema = []
alpha = 2 / (n+1)
for i in range(len(values)):
    if i == 0:
        ema.append(values[0])  # seed with first value
    else:
        value = alpha * values[i] + (1-alpha) * ema[-1]
        ema.append(value)

# --- Print Table ---
print(f"{'Year':<6}{'Original':<15}{'SMA(3)':<15}{'WMA(3)':<15}{'EMA(3)':<15}")
for y, orig, s, w, e in zip(years, values, sma, wma, ema):
    print(f"{y:<6}{orig:<15.2f}{(s if s is not None else '-'): <15}{(w if w is not None else '-'): <15}{e:<15.2f}")

# --- Plot ---
plt.figure(figsize=(10,6))
plt.plot(years, values, label='Original Data', marker='o')
plt.plot(years, sma, label='SMA (3)', linestyle='--')
plt.plot(years, wma, label='WMA (3)', linestyle='--')
plt.plot(years, ema, label='EMA (3)', linestyle='--')

plt.title(f"GDP {country} per Tahun")
plt.xlabel("Year")
plt.ylabel("GDP Value")
plt.legend()
plt.grid(True)
plt.show()

