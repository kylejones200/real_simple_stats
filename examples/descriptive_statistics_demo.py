from real_simple_stats.descriptives import (
    five_number_summary,
    interquartile_range,
    sample_variance,
    sample_std_dev,
    coefficient_of_variation,
    draw_frequency_table,
    draw_cumulative_frequency_table,
    detect_fake_statistics
)

x = [1, 2, 5, 6, 7, 9, 12, 15, 18, 19, 27]
print("Five-number summary:", five_number_summary(x))
print("IQR:", interquartile_range(x))
print("Sample variance:", sample_variance(x))
print("Sample standard deviation:", sample_std_dev(x))
print("Coefficient of variation:", coefficient_of_variation(x))

blood_types = ['A', 'O', 'A', 'B', 'B', 'AB', 'B', 'B', 'O', 'A', 'O', 'O', 'O', 'AB', 'B', 'AB', 'AB', 'A', 'O', 'A']
print("Frequency table:", draw_frequency_table(blood_types))

values = [1, 1, 2, 2, 3, 3, 3, 4]
print("Cumulative frequency:", draw_cumulative_frequency_table(values))

print("Bias warnings:", detect_fake_statistics("diet pill company", True, True))
