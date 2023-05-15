MAX_VALUE = 5
MIN_VALUE = 1750
MAX_WEIGHT = 1850

for x1 in range(MAX_VALUE + 1):
    for x2 in range(MAX_VALUE + 1):
        for x3 in range(MAX_VALUE + 1):
            for x4 in range(MAX_VALUE + 1):
                total_weight = 290 * x1 + 315 * x2 + 350 * x3 + 455 * x4
                if x1 + x2 + x3 + x4 <= MAX_VALUE and total_weight <= MAX_WEIGHT and total_weight >= MIN_VALUE:
                    print(x1, x2, x3, x4)