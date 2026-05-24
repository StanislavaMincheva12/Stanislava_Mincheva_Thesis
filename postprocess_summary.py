import pandas as pd
from pathlib import Path

summary_path = Path('results/summary.csv')
out_path = Path('results/summary_interpretable.csv')

if not summary_path.exists():
    raise FileNotFoundError('results/summary.csv not found')

df = pd.read_csv(summary_path)

total_L = df['n_L_accept_Y'] + df['n_L_accept_S']
total_H = df['n_H_accept_Y'] + df['n_H_accept_S']

df['share_L_Y'] = (df['n_L_accept_Y'] / total_L).fillna(0.0)
df['share_L_S'] = (df['n_L_accept_S'] / total_L).fillna(0.0)
df['share_H_Y'] = (df['n_H_accept_Y'] / total_H).fillna(0.0)
df['share_H_S'] = (df['n_H_accept_S'] / total_H).fillna(0.0)
df['delay_gap_H_minus_L'] = df['avg_delay_H'].fillna(0.0) - df['avg_delay_L'].fillna(0.0)
df['L_delays_less_than_H'] = df['delay_gap_H_minus_L'] > 0
df['firm_switches'] = df['switch_time_Y_to_S'].notna()

nice = df[[
    'bar_v', 'delta_v_over_bar_v', 'beta_H', 'gamma', 'P0', 'p_L', 'p_Y',
    'avg_delay_H', 'avg_delay_L', 'delay_gap_H_minus_L', 'L_delays_less_than_H',
    'switch_time_Y_to_S', 'firm_switches',
    'n_L_accept_Y', 'n_L_accept_S', 'share_L_Y', 'share_L_S',
    'all_adopted'
]].copy()

for col in ['avg_delay_H', 'avg_delay_L', 'delay_gap_H_minus_L', 'share_L_Y', 'share_L_S']:
    nice[col] = nice[col].round(3)

nice.to_csv(out_path, index=False)

print('\nSaved:', out_path)
print('\nFirst 20 rows of summary_interpretable.csv\n')
print(nice.head(20).to_string(index=False))

print('\nQuestion (i): avg_delay_H')
print(nice[['bar_v', 'delta_v_over_bar_v', 'beta_H', 'gamma', 'P0', 'p_L', 'p_Y', 'avg_delay_H']].head(20).to_string(index=False))

print('\nQuestion (ii): whether low-bias delays less')
print(nice[['avg_delay_H', 'avg_delay_L', 'delay_gap_H_minus_L', 'L_delays_less_than_H']].head(20).to_string(index=False))

print('\nQuestion (iii): firm switch timing')
print(nice[['gamma', 'P0', 'p_L', 'p_Y', 'firm_switches', 'switch_time_Y_to_S']].head(20).to_string(index=False))

print('\nQuestion (iv): low-bias contract shares')
print(nice[['n_L_accept_Y', 'n_L_accept_S', 'share_L_Y', 'share_L_S']].head(20).to_string(index=False))