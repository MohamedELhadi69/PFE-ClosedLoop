import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from data_utils import load_or_fetch_dataframe, parse_col
pd.set_option('display.max_columns', None)
OUTPUT_DIR = Path('graphs_output')
OUTPUT_DIR.mkdir(exist_ok=True)

def save_plot(filename):
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
print('Loading dataset...')
df = load_or_fetch_dataframe()
df['statistics'] = df['statistics'].apply(parse_col)
df['anomalies'] = df['anomalies'].apply(parse_col)
df['labels'] = df['labels'].apply(parse_col)
print(f'Total rows: {len(df)}')
print('Analyzing entire dataset...\n')
anomaly_types = []
anomaly_exists = []
zones = []
applications = []
all_means = {kpi: [] for kpi in df.iloc[0]['statistics'].keys()}
all_variances = {kpi: [] for kpi in df.iloc[0]['statistics'].keys()}
all_trends = {kpi: [] for kpi in df.iloc[0]['statistics'].keys()}
for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f'Processing row {idx}/32000...')
    anomaly = row['anomalies']
    anomaly_types.append(anomaly['type'])
    anomaly_exists.append(anomaly['exists'])
    zones.append(row['labels']['zone'])
    applications.append(row['labels']['application'])
    stats = row['statistics']
    for kpi in all_means.keys():
        all_means[kpi].append(stats[kpi]['mean'])
        all_variances[kpi].append(stats[kpi]['variance'])
        all_trends[kpi].append(stats[kpi]['trend'])
print('Data collection complete!\n')
print('Creating graphs...')
anomaly_type_counts = Counter(anomaly_types)
fig = plt.figure(figsize=(16, 8))
colors_anomaly = plt.cm.Set3(np.linspace(0, 1, len(anomaly_type_counts)))
sizes = list(anomaly_type_counts.values())
ax = fig.add_subplot(121)
wedges, texts, autotexts = ax.pie(sizes, colors=colors_anomaly, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)
ax.set_title(f'Anomaly Types Distribution', fontsize=12, fontweight='bold')
ax2 = fig.add_subplot(122)
ax2.axis('off')
info_text = 'Anomaly Type Counts\n' + '━' * 35 + '\n\n' + '\n'.join([f'{atype:30s}: {count:6d}' for atype, count in sorted(anomaly_type_counts.items(), key=lambda x: x[1], reverse=True)])
ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace', fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=2, linewidth=2))
fig.suptitle(f'Distribution of Anomaly Types (Entire Dataset - {len(df)} windows)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save_plot('graph1_anomaly_types.png')
plt.close()
fig, ax = plt.subplots(figsize=(10, 8))
anomaly_counts = Counter(anomaly_exists)
labels_tf = ['Anomaly Detected', 'Normal (No Anomaly)']
sizes_tf = [anomaly_counts[True], anomaly_counts[False]]
colors_tf = ['#d62728', '#2ca02c']
wedges, texts, autotexts = ax.pie(sizes_tf, labels=labels_tf, colors=colors_tf, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
ax.set_title(f'Anomaly Presence Distribution (Entire Dataset)', fontsize=14, fontweight='bold', pad=20)
info_text = f'Anomaly: {anomaly_counts[True]}\nNormal: {anomaly_counts[False]}'
ax.text(1.2, 0.5, info_text, transform=ax.transAxes, fontsize=11, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
save_plot('graph2_anomaly_tf.png')
plt.close()
fig, ax = plt.subplots(figsize=(10, 8))
zone_counts = Counter(zones)
colors_zones = plt.cm.Set2(np.linspace(0, 1, len(zone_counts)))
zone_labels = list(zone_counts.keys())
zone_sizes = list(zone_counts.values())
wedges, texts, autotexts = ax.pie(zone_sizes, labels=zone_labels, colors=colors_zones, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title(f'Network Zones Distribution (Entire Dataset)', fontsize=14, fontweight='bold', pad=20)
info_text = '\n'.join([f'Zone {z}: {c}' for z, c in zone_counts.items()])
ax.text(1.2, 0.5, info_text, transform=ax.transAxes, fontsize=11, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
plt.tight_layout()
save_plot('graph3_zones.png')
plt.close()
fig, ax = plt.subplots(figsize=(12, 8))
app_counts = Counter(applications)
colors_apps = plt.cm.Pastel1(np.linspace(0, 1, len(app_counts)))
app_labels = list(app_counts.keys())
app_sizes = list(app_counts.values())
wedges, texts, autotexts = ax.pie(app_sizes, labels=app_labels, colors=colors_apps, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)
ax.set_title(f'Application Types Distribution (Entire Dataset)', fontsize=14, fontweight='bold', pad=20)
info_text = '\n'.join([f'{app}: {c}' for app, c in app_counts.items()])
ax.text(1.2, 0.5, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
save_plot('graph4_applications.png')
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
kpi_names = list(all_means.keys())
overall_means = {kpi: np.mean(all_means[kpi]) for kpi in kpi_names}
overall_vars = {kpi: np.mean(all_variances[kpi]) for kpi in kpi_names}
signal_kpis = ['RSRP', 'UL_SNR']
signal_means = [overall_means[k] for k in signal_kpis]
ax = axes[0, 0]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(signal_kpis)))
bars = ax.bar(signal_kpis, signal_means, color=colors, edgecolor='black', linewidth=1.3)
for bar, val in zip(bars, signal_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Signal Quality KPIs (Overall)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
error_kpis = ['DL_BLER', 'UL_BLER']
error_means = [overall_means[k] for k in error_kpis]
ax = axes[0, 1]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(error_kpis)))
bars = ax.bar(error_kpis, error_means, color=colors, edgecolor='black', linewidth=1.3)
for bar, val in zip(bars, error_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.6f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
ax.set_title('Error Rates (Overall)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
mod_kpis = ['DL_MCS', 'UL_MCS', 'UL_NPRB', 'PRBs_DL_Current', 'PRBs_UL_Current']
mod_means = [overall_means[k] for k in mod_kpis]
ax = axes[1, 0]
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(mod_kpis)))
bars = ax.bar(mod_kpis, mod_means, color=colors, edgecolor='black', linewidth=1.3)
for bar, val in zip(bars, mod_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Modulation & Resources (Overall)', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
traffic_kpis = ['TX_Bytes', 'RX_Bytes', 'Estimated_UL_Buffer', 'PRB_Utilization_DL', 'PRB_Utilization_UL', 'UL_NumberOfPackets', 'DL_NumberOfPackets']
traffic_means = [overall_means[k] for k in traffic_kpis]
ax = axes[1, 1]
colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(traffic_kpis)))
bars = ax.bar(traffic_kpis, traffic_means, color=colors, edgecolor='black', linewidth=1.3)
for bar, val in zip(bars, traffic_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Traffic & Utilization (Overall)', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('KPI Means - Grouped by Category (Entire Dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
save_plot('graph5_kpi_means_overall.png')
plt.close()
fig, ax = plt.subplots(figsize=(14, 8))
sorted_indices = np.argsort([overall_vars[k] for k in kpi_names])[::-1]
sorted_names = [kpi_names[i] for i in sorted_indices]
sorted_vars = [overall_vars[kpi_names[i]] for i in sorted_indices]
colors_var = []
for v in sorted_vars:
    if v > 100:
        colors_var.append('#d62728')
    elif v < 1:
        colors_var.append('#2ca02c')
    else:
        colors_var.append('#1f77b4')
bars = ax.barh(sorted_names, sorted_vars, color=colors_var, edgecolor='black', linewidth=1.2)
for bar, value in zip(bars, sorted_vars):
    ax.text(value, bar.get_y() + bar.get_height() / 2, f' {value:.1f}', va='center', fontsize=8, fontweight='bold')
ax.set_xlabel('Variance Value (LOG SCALE)', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_title('KPI Variability - Overall Dataset\n(Log scale: TX_Bytes variance is 8.8M, BLER variance is near 0)', fontsize=12, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='x', which='both')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d62728', label='High (>100)'), Patch(facecolor='#1f77b4', label='Medium'), Patch(facecolor='#2ca02c', label='Low (<1)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
save_plot('graph6_variance_overall.png')
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
zone_anomalies = {}
for zone in zone_counts.keys():
    indices = [i for i, z in enumerate(zones) if z == zone]
    anomaly_count = sum((1 for i in indices if anomaly_exists[i]))
    zone_anomalies[zone] = (anomaly_count, len(indices))
zones_list = list(zone_anomalies.keys())
anomaly_in_zone = [zone_anomalies[z][0] for z in zones_list]
total_in_zone = [zone_anomalies[z][1] for z in zones_list]
ax = axes[0]
x = np.arange(len(zones_list))
width = 0.35
bars1 = ax.bar(x - width / 2, anomaly_in_zone, width, label='With Anomaly', color='#d62728', edgecolor='black')
bars2 = ax.bar(x + width / 2, total_in_zone, width, label='Total', color='#1f77b4', edgecolor='black')
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Anomalies by Zone', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(zones_list)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontweight='bold')
app_anomalies = {}
for app in app_counts.keys():
    indices = [i for i, a in enumerate(applications) if a == app]
    anomaly_count = sum((1 for i in indices if anomaly_exists[i]))
    app_anomalies[app] = (anomaly_count, len(indices))
apps_list = list(app_anomalies.keys())
anomaly_in_app = [app_anomalies[a][0] for a in apps_list]
total_in_app = [app_anomalies[a][1] for a in apps_list]
ax = axes[1]
x = np.arange(len(apps_list))
bars1 = ax.bar(x - width / 2, anomaly_in_app, width, label='With Anomaly', color='#d62728', edgecolor='black')
bars2 = ax.bar(x + width / 2, total_in_app, width, label='Total', color='#1f77b4', edgecolor='black')
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Anomalies by Application', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(apps_list, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
fig.suptitle('Anomaly Distribution by Zone & Application', fontsize=14, fontweight='bold')
plt.tight_layout()
save_plot('graph8_anomaly_zone_app.png')
plt.close()
print('\n' + '=' * 70)
print('DATASET SUMMARY STATISTICS')
print('=' * 70)
print(f'\nTotal Windows Analyzed: {len(df):,}')
print(f'Windows with Anomalies: {anomaly_counts[True]:,} ({anomaly_counts[True] / len(df) * 100:.1f}%)')
print(f'Normal Windows: {anomaly_counts[False]:,} ({anomaly_counts[False] / len(df) * 100:.1f}%)')
print(f'\nANOMALY TYPES:')
for atype, count in sorted(anomaly_type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {atype:20s}: {count:5d} ({count / len(df) * 100:5.1f}%)')
print(f'\nZONES:')
for zone, count in sorted(zone_counts.items()):
    anom_pct = zone_anomalies[zone][0] / zone_anomalies[zone][1] * 100
    print(f'  Zone {zone}: {count:5d} windows ({anom_pct:5.1f}% anomalies)')
print(f'\nAPPLICATIONS:')
for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True):
    anom_pct = app_anomalies[app][0] / app_anomalies[app][1] * 100
    print(f'  {app:20s}: {count:5d} windows ({anom_pct:5.1f}% anomalies)')
