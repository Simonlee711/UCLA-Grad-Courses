#%%
'''
Author: Simon Lee (simonlee711@g.ucla.edu)

P4 of ECE 131A Project
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#### PART A
t = 10**4  # Number of samples
n_values = [1, 2, 3, 10, 30, 100]  # Different values of n
a, b = 10, 16  # Parameters of the uniform distribution
logging.info('Parameters set for Part A: t={}, n_values={}, a={}, b={}'.format(t, n_values, a, b))

# Plot setup
plt.figure(figsize=(15, 10))
logging.info('Starting plot setup for Part A.')

for n in n_values:
    # Generate t samples of X_i for each i, then calculate Z_n for each sample set
    samples = np.random.uniform(a, b, (t, n))
    Z_n = samples.mean(axis=1)
    logging.info(f'Generated samples and calculated Z_n for n={n}.')
    
    # Plot the histogram of Z_n
    plt.hist(Z_n, bins='auto', density=True, alpha=0.5, label=f'n = {n}')

# Add labels and legend
plt.title('PDF of $Z_n$ for Different Values of $n$')
plt.xlabel('$Z_n$')
plt.ylabel('Density')
plt.legend()
logging.info('Completed plotting for Part A.')
plt.show()


#%%
#### PART C
logging.info('Starting Part C.')

plt.figure(figsize=(15, 10))

for n in n_values:
    # Generate t samples of X_i for each i, then calculate Z_n for each sample set
    samples = np.random.uniform(a, b, (t, n))
    Z_n = samples.mean(axis=1)
    logging.info(f'Generated samples and calculated Z_n for Part C, n={n}.')
    
    # Calculate the mean and standard deviation for the Gaussian
    mu = 13  # The mean is the same for all n
    sigma = np.sqrt(3 / n)  # Standard deviation for Z_n
    logging.info(f'Calculated Gaussian parameters for Part C, n={n}: mu={mu}, sigma={sigma:.4f}.')
    
    # Generate a range of x values for plotting the Gaussian PDF
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    
    # Plot the histogram of Z_n
    plt.hist(Z_n, bins='auto', density=True, alpha=0.5, label=f'Uniform Mean, n = {n}')
    
    # Plot the Gaussian PDF
    plt.plot(x, norm.pdf(x, mu, sigma), label=f'Gaussian Fit, n = {n}')
plt.title('PDF of $Z_n$ and Gaussian Fit for Different Values of $n$')
plt.xlabel('$Z_n$')
plt.ylabel('Density')
plt.legend()
logging.info('Completed plotting for Part C.')
plt.show()


#%%
#### Part D
logging.info('Starting Part D.')

prime_numbers = [2, 3, 5, 7, 11]
non_prime_numbers = [1, 4, 6, 8, 9, 10, 12]
weights = [2 if i in prime_numbers else 1 for i in range(1, 13)]
total_weight = sum(weights)
probabilities = [weight / total_weight for weight in weights]
logging.info('Calculated probabilities for modified 12-sided die.')

# Calculate mean and variance
values = np.arange(1, 13)
mu_die = np.sum(values * probabilities)
var_die = np.sum(((values - mu_die)**2) * probabilities)
sigma_die = np.sqrt(var_die)
logging.info(f'Calculated mean and variance for modified 12-sided die: mu={mu_die}, var={var_die}.')

# Plot setup
plt.figure(figsize=(18, 12))

for n in n_values:
    # Generate samples of Z_n for each n
    Z_n_samples = np.random.choice(values, size=(t, n), p=probabilities).mean(axis=1)
    logging.info(f'Generated Z_n samples for Part D, n={n}.')
    
    # Bin width adjustment as per instruction
    bin_width = 1 / (n + 1)
    
    # Calculate the standard deviation for Z_n
    sigma_Z_n = sigma_die / np.sqrt(n)
    
    # Generate a range of x values for plotting the Gaussian PDF
    x = np.linspace(mu_die - 3*sigma_Z_n, mu_die + 3*sigma_Z_n, 1000)
    
    # Plot the histogram of Z_n
    plt.hist(Z_n_samples, bins=np.arange(min(Z_n_samples), max(Z_n_samples) + bin_width, bin_width), 
             density=True, alpha=0.5, label=f'Discrete Mean, n = {n}')
    
    # Plot the Gaussian PDF
    plt.plot(x, norm.pdf(x, mu_die, sigma_Z_n), label=f'Gaussian Fit, n = {n}')
    logging.info(f'Completed plotting for Part D, n={n}.')
plt.title('PDF of $Z_n$ and Gaussian Fit for a 12-sided Die with Modified Probabilities')
plt.xlabel('$Z_n$')
plt.ylabel('Density')
plt.legend()
plt.show()


#%%
#### PART E
logging.info('Starting Part E: Gaussian Approximation for Exact PDF of Z_n.')
plt.figure(figsize=(14, 10))

# Mean and variance of X_i
mu_Xi = 13
var_Xi = 3
logging.info(f'Set mean and variance for X_i: mu={mu_Xi}, var={var_Xi}.')

# Loop over the values of n and plot the Gaussian PDF as an approximation for large n
for n in n_values:
    # Gaussian PDF parameters for Z_n
    mu_Zn = mu_Xi
    sigma_Zn = np.sqrt(var_Xi / n)
    logging.info(f'Calculated Gaussian PDF parameters for Z_n with n={n}: mu={mu_Zn}, sigma={sigma_Zn:.4f}.')

    # Generate x values for the Gaussian PDF
    x = np.linspace(mu_Zn - 3*sigma_Zn, mu_Zn + 3*sigma_Zn, 1000)
    
    # Plot the Gaussian PDF as the approximation for Z_n's exact PDF
    plt.plot(x, norm.pdf(x, mu_Zn, sigma_Zn), label=f'Gaussian Approx., n={n}')
    logging.info(f'Plotted Gaussian Approximation for n={n}.')

plt.title('Gaussian Approximation for Exact PDF of $Z_n$ for Different Values of n')
plt.xlabel('$Z_n$')
plt.ylabel('Probability Density Function (PDF)')
plt.legend()
logging.info('Completed plotting for Part E.')
plt.show()
logging.info('Script fully completed.')

