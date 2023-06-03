import numpy as np
from scipy.stats import norm, ttest_ind, levene, chi2
import matplotlib.pyplot as plt

patient = np.array(
    [
        0.2,
        10.4,
        0.3,
        0.4,
        10.9,
        11.3,
        1.1,
        2.0,
        12.4,
        16.2,
        2.1,
        17.6,
        18.9,
        3.3,
        3.8,
        20.7,
        4.5,
        4.8,
        24.0,
        25.4,
        4.9,
        40.0,
        5.0,
        42.2,
        5.3,
        50.0,
        60.0,
        7.5,
        9.8,
        45.0,
    ]
)

patient_del = np.array(
    [
        0.2,
        10.4,
        0.3,
        0.4,
        10.9,
        11.3,
        1.1,
        2.0,
        12.4,
        16.2,
        2.1,
        17.6,
        18.9,
        3.3,
        3.8,
        20.7,
        4.5,
        4.8,
        24.0,
        25.4,
        4.9,
        40.0,
        5.0,
        42.2,
        5.3,
    ]
)

normal = np.array(
    [
        0.2,
        5.4,
        0.3,
        5.7,
        0.4,
        5.8,
        0.7,
        7.5,
        1.2,
        8.7,
        1.5,
        8.8,
        1.5,
        9.1,
        1.9,
        10.3,
        2.0,
        15.6,
        2.4,
        16.1,
        2.5,
        16.5,
        2.8,
        16.7,
        3.6,
        20.0,
        4.8,
        20.7,
        4.8,
        33.0,
    ]
)


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


def plot_and_hist(data):
    plt.figure(1)
    plt.hist(data, density=True, bins=10, alpha=0.7, label="data")
    mu_fit, sigma_fit = norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    y = norm.pdf(x, mu_fit, sigma_fit)
    plt.plot(x, y, "r", label="Normal Fit")
    plt.xlabel("Data")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


p_patient = jarque_bera_test(patient)
p_patient_del = jarque_bera_test(patient_del)
p_normal = jarque_bera_test(normal)
print("Check result (before log transformation):", p_patient, p_patient_del, p_normal)

patient = np.log(patient)
patient_del = np.log(patient_del)
normal = np.log(normal)
p_patient = jarque_bera_test(patient)
p_patient_del = jarque_bera_test(patient_del)
p_normal = jarque_bera_test(normal)
print("Check result (after log transformation):", p_patient, p_patient_del, p_normal)

_, p_value_1 = levene(patient, normal)
_, p_value_2 = levene(patient_del, normal)
print("Var Check:", p_value_1, p_value_2)

plot_and_hist(patient)
plot_and_hist(patient_del)
plot_and_hist(normal)

t_1, p_1 = ttest_ind(patient, normal)
t_2, p_2 = ttest_ind(patient_del, normal)
print("Accept patient (with deletion):", p_1)
print("Accept patient (without deletion):", p_2)
