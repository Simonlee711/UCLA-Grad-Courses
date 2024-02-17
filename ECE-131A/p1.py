'''
Author: Simon Lee (simonlee711@g.ucla.edu)

P1 of ECE 131A Project
'''

import numpy as np
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simulate_fair_die_tosses(toss_counts):
    logging.info("Starting simulation of fair die tosses")
    results = {}
    for tosses in toss_counts:
        logging.info(f"Simulating {tosses} fair die tosses")
        outcomes = np.random.randint(1, 13, tosses)  # Generate random tosses
        odd_count = np.sum(outcomes % 2 != 0)       # Count how many are odd
        probability_odd = odd_count / tosses        # Estimate probability
        results[tosses] = probability_odd
        logging.info(f"Completed simulation for {tosses} tosses with probability {probability_odd}")
    return results

def simulate_modified_die_tosses(toss_counts):
    logging.info("Starting simulation of modified die tosses")
    prime_numbers = [2, 3, 5, 7, 11]
    weights = [2 if i in prime_numbers else 1 for i in range(1, 13)]
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    
    results = {}
    for tosses in toss_counts:
        logging.info(f"Simulating {tosses} modified die tosses")
        outcomes = np.random.choice(range(1, 13), size=tosses, p=probabilities)
        odd_count = np.sum(outcomes % 2 != 0)
        probability_odd = odd_count / tosses
        results[tosses] = probability_odd
        logging.info(f"Completed simulation for {tosses} tosses with probability {probability_odd}")
    return results

# Toss counts to simulate
toss_counts = [50, 100, 1000, 2000, 3000, 10000, 100000]

# Simulate fair and modified die tosses
fair_die_results = simulate_fair_die_tosses(toss_counts)
modified_die_results = simulate_modified_die_tosses(toss_counts)

logging.info(fair_die_results, modified_die_results)

##### SECOND PART #######
logging.info("Calculating probability of rolling an odd number with modified die")

# Calculate the theoretical probability for the modified die scenario
non_prime_count = 7  # number of non primes

# Prime count (with double weight)
prime_count = 5  # Prime numbers on the die: 2, 3, 5, 7, 11
double_prime_weight = 2 * prime_count  # Double weight for primes

# Solve for p in the equation: 7p + 10p = 1
total_p = non_prime_count + double_prime_weight  # Total weight
p = 1 / total_p

# Calculate probabilities for odd numbers
odd_non_primes = [1, 9]  # Non-prime odd numbers
odd_primes = [3, 5, 7, 11]  # Prime odd numbers

# Probability of rolling an odd number
probability_odd = len(odd_non_primes) * p + len(odd_primes) * 2 * p

logging.info(probability_odd)

