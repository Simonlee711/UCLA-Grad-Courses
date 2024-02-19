'''
Author: Simon Lee (simonlee711@g.ucla.edu)

Problem 2 for ECE 131A Project
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data from the uploaded file
file_path = './data/data.txt'
logging.info(f"Loading data from {file_path}")

# Read the data into a numpy array
try:
    data = np.loadtxt(file_path)
    logging.info("Data loaded successfully")
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    raise

# Calculate μ_MLE and σ_MLE based on the provided data
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # ddof=0 for population standard deviation as per MLE
logging.info(f"Calculated μ_MLE: {mu_mle}, σ_MLE: {sigma_mle}")

plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Data histogram')
logging.info("Histogram of the data plotted")

# Plot the PDF of the Gaussian distribution with μ_MLE and σ_MLE
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu_mle, sigma_mle)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian PDF')
title = "Histogram of Data with Gaussian PDF Overlay"
plt.title(title)
plt.xlabel("Data Value")
plt.ylabel("Density")
plt.legend()
logging.info("Displaying plot")
plt.show()

