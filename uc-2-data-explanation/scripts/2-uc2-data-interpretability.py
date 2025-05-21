import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import zscore

# Load datasets
frame_df = pd.read_csv('C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/frames-cleaned.csv')
heart_rate_df = pd.read_csv('C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/heart_rate.csv')

# Convert timestamps to datetime for alignment
frame_df['frame_timestamp'] = pd.to_datetime(frame_df['frame_timestamp'])
heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['timestamp'])

# Merge datasets on closest timestamp
merged_df = pd.merge_asof(frame_df.sort_values('frame_timestamp'),
                           heart_rate_df.sort_values('timestamp'),
                           left_on='frame_timestamp',
                           right_on='timestamp',
                           direction='nearest')

# Correlation Analysis
correlation_results = merged_df[['heart_rate', 'eyes_closed', 'yawning', 'alert']].corr()
print("Correlation Analysis:\n", correlation_results)

# Anomaly Detection
merged_df['Heart Rate Z-Score'] = zscore(merged_df['heart_rate'])
anomalies = merged_df[(merged_df['Heart Rate Z-Score'].abs() > 2) |
                       ((merged_df['yawning'] | merged_df['eyes_closed']) & ~merged_df['alert'])]
print("\nAnomalies Detected:\n", anomalies[['frame_timestamp', 'heart_rate', 'eyes_closed', 'yawning', 'alert']])

# Time Series Visualization
plt.figure(figsize=(12,6))
sns.lineplot(data=merged_df, x='frame_timestamp', y='heart_rate', label='Heart Rate')
sns.scatterplot(data=merged_df[merged_df['alert']], x='frame_timestamp', y='heart_rate', color='red', label='Alert')
plt.xticks(rotation=45)
plt.title('Heart Rate over Time with Alerts')
plt.legend()
plt.show()

# Clustering Analysis
features = merged_df[['heart_rate', 'eyes_closed', 'yawning', 'alert']]
kmeans = KMeans(n_clusters=4, random_state=42).fit(features)
merged_df['Cluster'] = kmeans.labels_

plt.figure(figsize=(8,6))
sns.scatterplot(data=merged_df, x='heart_rate', y='alert', hue='Cluster', palette='viridis')
plt.title('Driver State Clustering')
plt.show()

# Save processed data
merged_df.to_csv('C:/Users/USER/PycharmProjects/alfie/uc-2-data-explanation/csv-json/processed_driver_data.csv', index=False)
print("Processed data saved to processed_driver_data.csv")


"""
This script processes driver fatigue data by analyzing relationships between frame data (facial expressions, alerts) and heart rate. It performs four key functionalities:

1. Correlation Analysis

Computes statistical correlations between heart rate, eyes closed, yawning, and alerts to identify relationships between fatigue signs and physiological responses.

2. Anomaly Detection

Identifies unusual heart rate spikes or fatigue signs that occur without triggering an alert, flagging potential safety risks.
Time Series Visualization

3. Plots heart rate trends over time, highlighting moments when alerts were triggered, providing insights into how heart rate responds to different driving conditions.
Clustering Analysis

4. Uses K-Means clustering to group driving states based on heart rate, facial expressions, and alert conditions, categorizing different driver conditions (e.g., normal, fatigued, stressed).
"""