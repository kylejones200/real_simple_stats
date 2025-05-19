# --- STATISTICS GLOSSARY ---

GLOSSARY = {
    "!": "Factorial — product of all positive integers up to a given number n. Example: 4! = 4×3×2×1 = 24.",
    "f": "Frequency — number of times an event or value occurs in a dataset.",
    "n": "Sample size — the number of observations in a dataset.",
    "Q1": "First Quartile — the value below which 25% of the data fall.",
    "Q3": "Third Quartile — the value below which 75% of the data fall.",
    "P(X)": "Probability of event X occurring.",
    "E": "Margin of error — the expected amount of random sampling error in a survey's results.",
    "E(X)": "Expected value — the mean of a probability distribution.",
    "p": "Probability of success in a binomial distribution.",
    "q": "Probability of failure in a binomial distribution (q = 1 - p).",
    "μ": "Population mean — average of a population.",
    "σ": "Population standard deviation — measures spread of population data.",
    "s": "Sample standard deviation — measures spread of sample data.",
    "x̄": "Sample mean — average of a sample.",
    "σ²": "Variance — average of the squared deviations from the mean.",
    "α": "Alpha — significance level in hypothesis testing (e.g., 0.05).",
    "H0": "Null hypothesis — a statement of no effect or no difference.",
    "H1": "Alternate hypothesis — a statement indicating the presence of an effect or difference.",
    "r": "Correlation coefficient — measures linear relationship between two variables (range: -1 to 1).",
    "r²": "Coefficient of determination — proportion of variance explained by a regression model.",
    "Σ": "Summation — sum of a sequence of numbers.",
    "a": "Y-intercept in a linear regression equation.",
    "b": "Slope in a linear regression equation.",
    "ν": "Degrees of freedom — number of independent values that can vary in an analysis."
}

def lookup(term: str) -> str:
    """
    Look up a statistical term in the glossary.
    
    Args:
        term: A term from the glossary (case-sensitive for symbols).
        
    Returns:
        The definition if found, else a default message.
    """
    return GLOSSARY.get(term, "Definition not found. Try a symbol (e.g., 'μ') or abbreviation (e.g., 'H0').")

# Example usage
if __name__ == "__main__":
    terms = ["μ", "σ", "p", "r²", "H0", "Σ"]
    for t in terms:
        print(f"{t}: {lookup(t)}")
