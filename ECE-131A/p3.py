#%%
'''
Author: Simon Lee (simonlee711@g.ucla.edu)

P3 of ECE 131A Project
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Example of adding logging to your code (add this at various points in your code as needed)
logging.info('User data loaded successfully')


# Load the user data
data_path = './data/user_data.csv'
data = pd.read_csv(data_path)
logging.info('User data loaded from {}'.format(data_path))


# Display the first few rows of the dataframe to understand its structure
data.head()

### PART A ####
def plot_pmf(data, column, title, xlabel, ax):
    pmf = data[column].value_counts().sort_index() / len(data)
    ax.bar(pmf.index, pmf.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability')
    ax.set_xticks(pmf.index)
    ax.grid(axis='y')

# Plot PMFs for each variable as subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 8))
plot_pmf(data, 'Bought', 'PMF of Bought', 'Bought Status (0 = Did Not Buy, 1 = Did Buy)', axs[0, 0])
plot_pmf(data, 'Spender Type', 'PMF of Type of Spender', 'Spender Type (1 = Large, 2 = Medium, 3 = Small)', axs[0, 1])
plot_pmf(data, 'Sex', 'PMF of Sex', 'Sex (0 = Female, 1 = Male)', axs[1, 0])
plot_pmf(data, 'Age', 'PMF of Age', 'Age', axs[1, 1])
plt.tight_layout()
plt.show()
logging.info('PMF plots generated successfully')


# %%
##### PART B
def plot_conditional_pmf_subplot(data, condition_column, condition_value, target_columns, titles, xlabels, figsize=(18, 5)):
    fig, axes = plt.subplots(1, len(target_columns), figsize=figsize)
    for i, (target_column, title, xlabel) in enumerate(zip(target_columns, titles, xlabels)):
        subset = data[data[condition_column] == condition_value]
        pmf = subset[target_column].value_counts(normalize=True).sort_index()
        axes[i].bar(pmf.index, pmf.values)
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Conditional Probability')
        axes[i].set_xticks(pmf.index)
        axes[i].grid(axis='y')
    plt.tight_layout()
    plt.show()


# Plotting for Bought = 1
plot_conditional_pmf_subplot(
    data, 'Bought', 1, 
    ['Spender Type', 'Sex', 'Age'], 
    ['Conditional PMF of Spender Type | Bought = 1', 'Conditional PMF of Sex | Bought = 1', 'Conditional PMF of Age | Bought = 1'], 
    ['Spender Type', 'Sex', 'Age'])

# Plotting for Bought = 0
plot_conditional_pmf_subplot(
    data, 'Bought', 0, 
    ['Spender Type', 'Sex', 'Age'], 
    ['Conditional PMF of Spender Type | Bought = 0', 'Conditional PMF of Sex | Bought = 0', 'Conditional PMF of Age | Bought = 0'], 
    ['Spender Type', 'Sex', 'Age'])
logging.info('PMF plots generated successfully')


#%%
#### PART C
# Calculating the prior probabilities P(B=0) and P(B=1) with the user data
prior_b0 = data['Bought'].value_counts(normalize=True)[0]
prior_b1 = data['Bought'].value_counts(normalize=True)[1]

# Estimating the conditional probabilities for B=0 and B=1 with the user data
# For T=1
p_t1_b0 = data[data['Bought'] == 0]['Spender Type'].value_counts(normalize=True)[1]
p_t1_b1 = data[data['Bought'] == 1]['Spender Type'].value_counts(normalize=True)[1]

# For S=0
p_s0_b0 = data[data['Bought'] == 0]['Sex'].value_counts(normalize=True)[0]
p_s0_b1 = data[data['Bought'] == 1]['Sex'].value_counts(normalize=True)[0]

# For A<=67
p_a67_b0 = data[(data['Bought'] == 0) & (data['Age'] <= 67)]['Age'].count() / data[data['Bought'] == 0]['Age'].count()
p_a67_b1 = data[(data['Bought'] == 1) & (data['Age'] <= 67)]['Age'].count() / data[data['Bought'] == 1]['Age'].count()

# Applying the Naive Bayes formula with the user data
p_b0_given_t1_s0_a_le_67 = p_t1_b0 * p_s0_b0 * p_a67_b0 * prior_b0
p_b1_given_t1_s0_a_le_67 = p_t1_b1 * p_s0_b1 * p_a67_b1 * prior_b1

# Normalizing these probabilities to ensure they sum to 1
total = p_b0_given_t1_s0_a_le_67 + p_b1_given_t1_s0_a_le_67
normalized_p_b0 = p_b0_given_t1_s0_a_le_67 / total
normalized_p_b1 = p_b1_given_t1_s0_a_le_67 / total

normalized_p_b0, normalized_p_b1
logging.info('Calculated user normalized probabilities: P(B=0)={}, P(B=1)={}'.format(normalized_p_b0, normalized_p_b1))

