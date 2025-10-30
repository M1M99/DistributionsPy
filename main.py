import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# houses_room_count = np.random.normal(4,1,1500)
# plt.hist(houses_room_count,bins=100,color='aqua',edgecolor='black')
# plt.xlabel('Room Count')
# plt.ylabel('House Count')
# plt.show()

# house_room_count = np.random.uniform(low=1, high=100, size=1000)
# plt.hist(house_room_count,bins=10,color='red',edgecolor='yellow')
# plt.xlabel('House Room Count')
# plt.ylabel('House count')
# plt.show()

# house_room_count = np.random.poisson(3,2000)
#
# plt.hist(house_room_count, bins=4,color='red',edgecolor='black')
# plt.xlabel('Room count')
# plt.ylabel('House Count')
# plt.show()


crypto = pd.read_csv('crypto_data.csv')
customer = crypto['Customers']
website_visits = crypto['Website_Visits']
crypto_low = crypto['Crypto_Low']
height_cm = crypto['Height_cm']
coin_age_years = crypto['Coin_Age_Years']
plt.hist(customer,bins=5) #poisson
plt.show()
test_scores = crypto['Test_Score']
plt.hist(test_scores, bins=8, color='skyblue', edgecolor='black')
plt.title("Normal Distribution of Test Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

plt.hist(website_visits, bins=8, color='skyblue', edgecolor='black')
plt.title("Website Visits Distribution")
plt.xlabel("Visits")
plt.ylabel("Frequency")
plt.show()

plt.hist(crypto_low, bins=8, color='skyblue', edgecolor='black')
plt.show()

plt.hist(coin_age_years, bins=4, color='skyblue', edgecolor='black')
plt.show()

plt.hist(height_cm, bins=4, color='skyblue', edgecolor='black')
plt.show()