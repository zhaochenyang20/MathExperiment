import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def calculate_rejection_probability(mu_0, mu, sigma_0, alpha, N):
    sigma = np.sqrt(mu * (1.0 - mu))
    threshold = sigma_0 / np.sqrt(N) * norm.ppf(alpha) + mu_0
    return norm.cdf((threshold - mu) / (sigma / np.sqrt(N)))


def main():
    mu_0 = 0.90
    sigma_0 = np.sqrt(mu_0 * (1.0 - mu_0))
    alpha = 1 - 0.95

    all_N = np.arange(50, 1201)
    mu = 0.86
    prob = np.zeros(len(all_N))

    for index, N in enumerate(all_N):
        prob[index] = calculate_rejection_probability(mu_0, mu, sigma_0, alpha, N)

    plt.plot(all_N, prob)
    plt.xlabel("N")
    plt.ylabel("Probability of Rejection")
    plt.show()


if __name__ == "__main__":
    main()
