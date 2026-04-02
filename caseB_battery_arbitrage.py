import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt

# ================== LOAD DATA ==================
df = pd.read_csv('caseB_grid_battery_market_hourly.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])          # FIX: convert to datetime
print("Columns:", df.columns.tolist())
print(f"Data points: {len(df)}")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

price = df['day_ahead_price_gbp_per_mwh'].values           # £/MWh
ancillary_pay = df['ancillary_availability_gbp_per_mw_per_h'].values

T = len(df)
dt = 1.0   # hours

# ================== PARAMETERS ==================
E = 2.0              # MWh (2000 kWh)
P_max = 1.0          # MW
eta_rt = 0.88
eta_ch = np.sqrt(eta_rt)   # 0.938
eta_dis = np.sqrt(eta_rt)  # 0.938
SOC_init = 1.0       # MWh (50% of 2 MWh)

print(f"Price range: {price.min():.2f} to {price.max():.2f} £/MWh")
print(f"Unit check: {price[0]} £/MWh = {price[0]/1000:.4f} £/kWh\n")

# ================== BASE CASE (Pure Arbitrage) ==================
prob = LpProblem("Battery_Arbitrage", LpMaximize)
P_ch = LpVariable.dicts("P_ch", range(T), 0, P_max)
P_dis = LpVariable.dicts("P_dis", range(T), 0, P_max)
SOC = LpVariable.dicts("SOC", range(T+1), 0, E)

# Objective
prob += lpSum((P_dis[t] * price[t] - P_ch[t] * price[t]) * dt for t in range(T))

# SOC dynamics
for t in range(T):
    prob += SOC[t+1] == SOC[t] + (P_ch[t] * eta_ch - P_dis[t] / eta_dis) * dt

# Prevent simultaneous charge/discharge (optional but clean)
for t in range(T):
    prob += P_ch[t] + P_dis[t] <= P_max

# Boundary conditions
prob += SOC[0] == SOC_init
prob += SOC[T] == SOC_init

prob.solve(PULP_CBC_CMD(msg=0))
print("Base case solver status:", LpStatus[prob.status])
base_profit = value(prob.objective)
print(f"Base case profit (£): {base_profit:.2f}")

# Extract results
P_ch_base = np.array([value(P_ch[t]) for t in range(T)])
P_dis_base = np.array([value(P_dis[t]) for t in range(T)])
SOC_base = np.array([value(SOC[t]) for t in range(T+1)])

# ================== EXTENSION (Market Stacking) ==================
prob_ext = LpProblem("Battery_Stacking", LpMaximize)
P_ch_ext = LpVariable.dicts("P_ch_ext", range(T), 0, P_max)
P_dis_ext = LpVariable.dicts("P_dis_ext", range(T), 0, P_max)
R = LpVariable.dicts("R", range(T), 0, P_max)          # reserved for ancillary
SOC_ext = LpVariable.dicts("SOC_ext", range(T+1), 0, E)

# Objective: arbitrage + ancillary availability payment
prob_ext += lpSum((P_dis_ext[t] * price[t] - P_ch_ext[t] * price[t]) * dt +
                  R[t] * ancillary_pay[t] * dt for t in range(T))

for t in range(T):
    prob_ext += SOC_ext[t+1] == SOC_ext[t] + (P_ch_ext[t] * eta_ch - P_dis_ext[t] / eta_dis) * dt
    prob_ext += P_ch_ext[t] <= P_max - R[t]
    prob_ext += P_dis_ext[t] <= P_max - R[t]
    prob_ext += P_ch_ext[t] + P_dis_ext[t] <= P_max - R[t]

prob_ext += SOC_ext[0] == SOC_init
prob_ext += SOC_ext[T] == SOC_init

prob_ext.solve(PULP_CBC_CMD(msg=0))
print("Stacking case solver status:", LpStatus[prob_ext.status])
ext_profit = value(prob_ext.objective)
print(f"Stacking case profit (£): {ext_profit:.2f}")

P_ch_ext_vals = np.array([value(P_ch_ext[t]) for t in range(T)])
P_dis_ext_vals = np.array([value(P_dis_ext[t]) for t in range(T)])
R_vals = np.array([value(R[t]) for t in range(T)])
SOC_ext_vals = np.array([value(SOC_ext[t]) for t in range(T+1)])

# ================== VERIFICATION CHECKS ==================
print("\n=== VERIFICATION ===")
print("Base case:")
print(f"  End SOC == Initial SOC? {abs(SOC_base[-1] - SOC_init) < 1e-6}")
print(f"  Energy balance (MWh, should be ~0): {sum(P_ch_base*eta_ch - P_dis_base/eta_dis):.6f}")
print(f"  Simultaneous charge+discharge hours: {np.sum((P_ch_base > 0.01) & (P_dis_base > 0.01))}")

print("Stacking case:")
print(f"  End SOC == Initial SOC? {abs(SOC_ext_vals[-1] - SOC_init) < 1e-6}")
print(f"  Energy balance (MWh): {sum(P_ch_ext_vals*eta_ch - P_dis_ext_vals/eta_dis):.6f}")
print(f"  Hours where R[t] > 0.01: {np.sum(R_vals > 0.01)}")

# ================== ADDITIONAL INSIGHTS ==================
# 1. Daily profit comparison
net_base = P_dis_base - P_ch_base
net_stack = P_dis_ext_vals - P_ch_ext_vals

daily_profit_base = []
daily_profit_stack = []
for d in range(60):   # 60 days
    idx = slice(d*24, (d+1)*24)
    daily_profit_base.append(np.sum(net_base[idx] * price[idx]))
    daily_profit_stack.append(np.sum(net_stack[idx] * price[idx] + R_vals[idx] * ancillary_pay[idx]))

print("\n=== ADDITIONAL INSIGHTS ===")
print(f"Total energy charged (base): {sum(P_ch_base):.2f} MWh")
print(f"Total energy discharged (base): {sum(P_dis_base):.2f} MWh")
print(f"Energy throughput (base): {sum(P_ch_base + P_dis_base):.2f} MWh")
print(f"Profit per MWh throughput (base): {base_profit / max(sum(P_ch_base + P_dis_base), 1e-6):.2f} £/MWh")
print(f"Same for stacking: {ext_profit / max(sum(P_ch_ext_vals + P_dis_ext_vals), 1e-6):.2f} £/MWh")

# 2. Price vs net power (scatter) – shows arbitrage threshold
plt.figure(figsize=(8,5))
plt.scatter(price, net_base, alpha=0.5, s=5, label='Base arbitrage')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Day-ahead price (£/MWh)')
plt.ylabel('Net power (MW)')
plt.title('Battery dispatch vs price (base case)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('caseB_price_vs_power.png', dpi=300)
plt.show()

# 3. Daily profit boxplot comparison
plt.figure(figsize=(6,6))
bp = plt.boxplot([daily_profit_base, daily_profit_stack], tick_labels=['Base arbitrage', 'Market stacking'],
                 patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
plt.ylabel('Daily profit (£)')
plt.title('Daily profit comparison (60 days)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('caseB_daily_profit_boxplot.png', dpi=300)
plt.show()

# ================== MAIN PLOTS (full time series) ==================
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Top: Price
ax[0].plot(df['timestamp'], price, color='blue', linewidth=1, label='Day-ahead price')
ax[0].set_ylabel('Price (£/MWh)')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Middle: Net power (base case only – stacking looks similar)
ax[1].plot(df['timestamp'], net_base, color='green', linewidth=0.8, label='Net power (base)')
ax[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
ax[1].set_ylabel('Net power (MW)')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

# Bottom: SOC comparison
ax[2].plot(df['timestamp'], SOC_base[:-1], color='purple', linewidth=1, label='SOC (base)')
ax[2].plot(df['timestamp'], SOC_ext_vals[:-1], color='orange', linestyle='--', linewidth=1, label='SOC (stacking)')
ax[2].set_ylabel('SOC (MWh)')
ax[2].legend()
ax[2].grid(True, alpha=0.3)

plt.xlabel('Date (60 days)')
plt.suptitle('Case B – Grid Battery Arbitrage vs Market Stacking')
plt.tight_layout()
plt.savefig('caseB_full_timeseries.png', dpi=300)
plt.show()

# ================== ZOOM (first 7 days) ==================
zoom = 7*24
fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax2[0].plot(df['timestamp'][:zoom], price[:zoom], color='blue', label='Price')
ax2[0].set_ylabel('Price (£/MWh)')
ax2[0].legend()
ax2[0].grid(True, alpha=0.3)
ax2[1].plot(df['timestamp'][:zoom], net_base[:zoom], color='green', label='Net power (base)')
ax2[1].axhline(0, color='black', linestyle='--')
ax2[1].set_ylabel('Net power (MW)')
ax2[1].set_xlabel('Date (first 7 days)')
ax2[1].legend()
ax2[1].grid(True, alpha=0.3)
plt.suptitle('Zoom: first 7 days')
plt.tight_layout()
plt.savefig('caseB_zoom_7days.png', dpi=300)
plt.show()

# ================== OPTIONAL EXTENSION PLOT: Reserve Utilisation ==================
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], R_vals, color='gray', linewidth=1.2, label='Ancillary Reserve (MW)')
plt.fill_between(df['timestamp'], 0, R_vals, color='gray', alpha=0.3)
plt.ylabel('Reserved Capacity (MW)')
plt.xlabel('Date (60 days)')
plt.title('Market Stacking – Ancillary Reserve Utilisation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('caseB_reserve_utilisation.png', dpi=300)
plt.show()

print("\nAll plots saved: caseB_full_timeseries.png, caseB_zoom_7days.png,")
print("caseB_price_vs_power.png, caseB_daily_profit_boxplot.png")
print("Extra plot saved: caseB_reserve_utilisation.png (optional for appendix)")