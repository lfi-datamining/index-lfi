import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('GlobalWeatherRepository.csv')
a = df[df['location_name'] == 'Kabul']


'''
data = a[['pressure_mb', 'wind_kph']]
data = data.sort_values('pressure_mb')

plt.plot(data['pressure_mb'], data['wind_kph']) 
plt.xlabel('Pressure, mb')
plt.ylabel('Wind speed, kph')
plt.show()

'''
#It is easy to notice thanks to the graph that as pressure increases in a given area of ​​​​space, the wind speed decreases. This is due to the fact that air moves to those points in space where the pressure is less and accelerates towards this area




'''
sns.set_theme(style="ticks")

# Initialize the figure with a linear x axis
f, ax = plt.subplots(figsize=(17, 6))
ax.set_xscale("linear")

plt.legend()
ax.get_xaxis().set_visible(False)
plt.title('Dependence of wind speed (in kilometers per hour) on atmospheric pressure (in millibars) for Kabul, Afghanistan')

# Plot the average_scores with horizontal boxes
sns.boxplot(
    a, x="condition_text", y="wind_kph", hue="condition_text",
    whis=[0, 100], width=.4, palette="vlag", legend='full'
)

# Add in points to show each student
sns.stripplot(a, x="condition_text", y="wind_kph", size=4, color=".3")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

#plt.title('Correlation between parental academic background and average scores')
plt.show()
'''

#Thanks to the graph, we notice that the wind speed can change over a wide range in sunny weather. The range of wind speed values ​​in cloudy weather has the smallest scatter



correlation_matrix = df[['latitude', 'longitude', 'temperature_celsius', 'wind_degree','pressure_mb','precip_mm','humidity','feels_like_celsius','visibility_km','uv_index','gust_mph']].corr()
plt.figure(figsize=(12, 12))

sns.heatmap(correlation_matrix, annot=True, cmap='bwr', vmin=0, vmax=1, linewidths=0.5)

plt.title('Correlation Matrix of Student Performance Metrics')
plt.show()

#We see that the actual temperature and the subjectively felt temperature correlate with the UV index



