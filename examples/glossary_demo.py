"""Demo: Statistical glossary - lookup statistical terms and symbols."""

import logging

from real_simple_stats.glossary import GLOSSARY, lookup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Look up specific terms
logger.info("μ (population mean): %s", lookup("μ"))
logger.info("σ (population std dev): %s", lookup("σ"))
logger.info("H0 (null hypothesis): %s", lookup("H0"))
logger.info("r (correlation): %s", lookup("r"))
logger.info("α (significance level): %s", lookup("α"))

# Browse all terms
logger.info("\nGlossary has %s terms. Examples:", len(GLOSSARY))
for term in ["μ", "σ", "α", "r", "r²", "H0", "H1", "E(X)", "Q1"]:
    logger.info("  %s: %s...", term, lookup(term)[:55])
