import logging

from real_simple_stats.descriptive_statistics import (
    coefficient_of_variation,
    detect_fake_statistics,
    draw_cumulative_frequency_table,
    draw_frequency_table,
    five_number_summary,
    interquartile_range,
    sample_std_dev,
    sample_variance,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

x = [1, 2, 5, 6, 7, 9, 12, 15, 18, 19, 27]
logger.info("Five-number summary: %s", five_number_summary(x))
logger.info("IQR: %s", interquartile_range(x))
logger.info("Sample variance: %s", sample_variance(x))
logger.info("Sample standard deviation: %s", sample_std_dev(x))
logger.info("Coefficient of variation: %s", coefficient_of_variation(x))

blood_types = [
    "A",
    "O",
    "A",
    "B",
    "B",
    "AB",
    "B",
    "B",
    "O",
    "A",
    "O",
    "O",
    "O",
    "AB",
    "B",
    "AB",
    "AB",
    "A",
    "O",
    "A",
]
logger.info("Frequency table: %s", draw_frequency_table(blood_types))

values = [1, 1, 2, 2, 3, 3, 3, 4]
logger.info("Cumulative frequency: %s", draw_cumulative_frequency_table(values))

logger.info("Bias warnings: %s", detect_fake_statistics("diet pill company", True, True))
