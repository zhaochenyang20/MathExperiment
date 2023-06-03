import numpy as np
from scipy.stats import norm, ttest_1samp, t, chi2
import matplotlib.pyplot as plt


def shapiro_wilk_test(data):
    _, p_value = norm.fit(data)
    return p_value


def jarque_bera_test(data):
    _, p_value = norm.fit(data)
    n = len(data)
    skewness = (1 / n) * np.sum((data - np.mean(data)) ** 3) / np.std(data) ** 3
    kurtosis = (1 / n) * np.sum((data - np.mean(data)) ** 4) / np.std(data) ** 4 - 3
    jb_value = (n / 6) * (skewness**2 + (1 / 4) * kurtosis**2)
    p_value = 1 - chi2.cdf(jb_value, df=2)
    return p_value


def Confidence(data, confidence_level):
    n = len(data)
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)
    margin_error = t_value * sample_std / np.sqrt(n)
    lower_bound = sample_mean - margin_error
    upper_bound = sample_mean + margin_error
    print(f"Confidence Interval[{confidence_level}]:", lower_bound, "-", upper_bound)


heights = np.array(
    [
        172,
        171,
        166,
        160,
        155,
        173,
        166,
        170,
        167,
        173,
        178,
        173,
        163,
        165,
        170,
        163,
        172,
        182,
        171,
        177,
        169,
        168,
        168,
        175,
        176,
        168,
        161,
        169,
        171,
        178,
        177,
        170,
        173,
        172,
        170,
        172,
        177,
        176,
        175,
        184,
        169,
        165,
        164,
        173,
        172,
        169,
        173,
        173,
        166,
        163,
        170,
        160,
        165,
        177,
        169,
        176,
        177,
        172,
        165,
        166,
        171,
        169,
        170,
        172,
        169,
        167,
        175,
        164,
        166,
        169,
        167,
        179,
        176,
        182,
        186,
        166,
        169,
        173,
        169,
        171,
        167,
        168,
        165,
        168,
        176,
        170,
        158,
        165,
        172,
        169,
        169,
        172,
        162,
        175,
        174,
        167,
        166,
        174,
        168,
        170,
    ]
)

weights = np.array(
    [
        75,
        62,
        62,
        55,
        57,
        58,
        55,
        63,
        53,
        60,
        60,
        73,
        47,
        66,
        60,
        50,
        57,
        63,
        59,
        64,
        55,
        67,
        65,
        67,
        64,
        50,
        49,
        63,
        61,
        64,
        66,
        58,
        67,
        59,
        62,
        59,
        58,
        68,
        68,
        70,
        64,
        52,
        59,
        74,
        69,
        52,
        57,
        61,
        70,
        57,
        56,
        65,
        58,
        66,
        63,
        60,
        67,
        56,
        56,
        49,
        65,
        62,
        58,
        64,
        58,
        72,
        76,
        59,
        63,
        54,
        54,
        62,
        63,
        69,
        77,
        76,
        72,
        59,
        65,
        71,
        47,
        65,
        64,
        57,
        57,
        57,
        51,
        62,
        53,
        66,
        58,
        50,
        52,
        75,
        66,
        63,
        50,
        64,
        62,
        59,
    ]
)

p_value_jb_heights = jarque_bera_test(heights)
p_value_sw_heights = shapiro_wilk_test(heights)
p_value_jb_weights = jarque_bera_test(weights)
p_value_sw_weights = shapiro_wilk_test(weights)

print("Check result:")
print("Jarque-Bera Test (heights):", p_value_jb_heights)
print("Shapiro-Wilk Test (heights):", p_value_sw_heights)
print("Jarque-Bera Test (weights):", p_value_jb_weights)
print("Shapiro-Wilk Test (weights):", p_value_sw_weights)

plt.figure(1)
plt.hist(heights, density=True, bins=10, alpha=0.7, label="Height Distribution")
mu_fit, sigma_fit = norm.fit(heights)
x = np.linspace(heights.min(), heights.max(), 100)
y = norm.pdf(x, mu_fit, sigma_fit)
plt.plot(x, y, "r", label="Normal Fit")
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(2)
plt.hist(weights, density=True, bins=10, alpha=0.7, label="Weight Distribution")
mu_fit, sigma_fit = norm.fit(weights)
x = np.linspace(weights.min(), weights.max(), 100)
y = norm.pdf(x, mu_fit, sigma_fit)
plt.plot(x, y, "r", label="Normal Fit")
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.legend()
plt.show()


def hypothesis_test(data, popmean, alpha):
    t_stat, p_value = ttest_1samp(data, popmean)
    print("Hypothesis Test:", p_value)
    return p_value


print("\n=======heights=======")
sample_mean = np.mean(heights)
sample_std = np.std(heights, ddof=1)
print("Point Estimate (mean, std):", sample_mean, sample_std)
Confidence(heights, 0.99)
Confidence(heights, 0.97)
Confidence(heights, 0.95)
hypothesis_test(heights, popmean=167.5, alpha=0.05)

print("\n=======weights=======")
sample_mean = np.mean(weights)
sample_std = np.std(weights, ddof=1)
print("Point Estimate (mean, std):", sample_mean, sample_std)
Confidence(weights, 0.99)
Confidence(weights, 0.97)
Confidence(weights, 0.95)
hypothesis_test(weights, popmean=60.2, alpha=0.05)
