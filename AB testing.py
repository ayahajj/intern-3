#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


# In[13]:


# Load the dataset
df = pd.read_csv('C:/Users/USER/Downloads/ab_data.csv')

# Display first few rows of the dataset
print(df.head())

# Display summary of the dataset
print(df.describe())


# In[14]:


# Filter data for control and treatment groups
df_ab = df[df['group'].isin(['control', 'treatment'])]

# Display the filtered data
print(df_ab.head())


# In[15]:


# Calculate conversion rates for each group
conversion_rates = df_ab.groupby('group')['converted'].agg(['sum', 'count']).reset_index()
conversion_rates.columns = ['group', 'Conversions', 'Total_Users']
conversion_rates['Conversion_Rate'] = conversion_rates['Conversions'] / conversion_rates['Total_Users']

# Display conversion rates
print(conversion_rates)


# In[16]:


# Plot conversion rates
sns.barplot(x='group', y='Conversion_Rate', data=conversion_rates)
plt.title('Conversion Rates by Group')
plt.ylabel('Conversion Rate')
plt.show()


# In[17]:


# Perform z-test
counts = conversion_rates['Conversions']
nobs = conversion_rates['Total_Users']

z_stat, p_value = proportions_ztest(count=counts, nobs=nobs)
print(f"Z-statistic: {z_stat:.2f}")
print(f"P-value: {p_value:.4f}")


# In[18]:


alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis. There is a statistically significant difference between the two groups.")
    if conversion_rates.loc[conversion_rates['group'] == 'treatment', 'Conversion_Rate'].values[0] > conversion_rates.loc[conversion_rates['group'] == 'control', 'Conversion_Rate'].values[0]:
        print("The treatment group performs better.")
    else:
        print("The control group performs better.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference between the two groups.")

