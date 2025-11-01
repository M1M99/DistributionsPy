import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, poisson, chisquare,skew
crypto = pd.read_csv('crypto_data.csv')

#region Variables
customer = crypto['Customers']
website_visits = crypto['Website_Visits']
crypto_low = crypto['Crypto_Low']
coin_age_years = crypto['Coin_Age_Years']
crypto_open = crypto['Crypto_Open']
active_users = crypto['Active_Users']
test_scores = crypto['Crypto_Transactions']
#endregion

plt.hist(customer,bins=5) #poisson
plt.show()
plt.hist(test_scores, bins=8, color='skyblue', edgecolor='black')
plt.title("Normal Distribution of Test Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
values, counts = np.unique(customer, return_counts=True)
lambda_ = np.mean(customer)

expected = len(customer) * poisson.pmf(values, lambda_)
expected = expected * (np.sum(counts) / np.sum(expected))

chi_stat, p_value = chisquare(f_obs=counts, f_exp=expected)
print("Chi-square Statistic:", chi_stat)
print("p-value:", p_value)
if p_value > 0.05:
    print("Poisson Distribution")
else:
    print("Not Poisson Distribution")


plt.hist(website_visits, bins=8, color='skyblue', edgecolor='black')
plt.title("Website Visits Distribution")
plt.xlabel("Visits")
plt.ylabel("Frequency")
plt.show()

plt.hist(crypto_low, bins=8, color='skyblue', edgecolor='black')
plt.title("Crypto Low Distribution")
plt.show()

plt.hist(coin_age_years, bins=4, color='skyblue', edgecolor='black')
plt.title("Coin Age Distribution")
plt.xlabel("Ages")
plt.ylabel("Frequency")
plt.show()

plt.hist(active_users, bins=4, color='skyblue', edgecolor='black')
plt.title("Active Users Distribution")
plt.xlabel("Users")
plt.ylabel("Frequency")
plt.show()
print("Active Users Skewness:", skew(active_users))



plt.hist(crypto_open, bins=4, color='skyblue', edgecolor='black')
plt.title("Crypto Open Distribution")
plt.show()

#region ActiveUsersDistribution
plt.hist(active_users, bins=10, color='skyblue', edgecolor='black')
plt.title("Active Users Distribution")
plt.xlabel("Users")
plt.ylabel("Frequency")
plt.show()
stat, p = normaltest(active_users)
print("Test statistic:", stat)
print("p-value:", p)
if p < 0.05:
    print("Fail Not Normal Distribution")
else:
    print("Pass Normal Distribution")
#endregion

#region With_Loop
columns = {
    'Website_Visits': website_visits,
    'Crypto_Low': crypto_low,
    'Crypto_Open': crypto_open,
    'Crypto_Transactions': test_scores,
    'Coin_Age_Years': coin_age_years,
    'Customers': customer,
    'Active_Users': active_users,
}

for name,data in columns.items():
    plt.hist(data, bins=8, color='skyblue', edgecolor='black')
    plt.title(f"{name} Distribution")
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.show()
#endregion
plt.figure(figsize=(8,6))
sns.heatmap(crypto.corr(), annot=True) #corr
plt.title("Correlation Heatmap of Crypto Dataset")
plt.show()