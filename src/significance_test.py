import numpy as np
from scipy import stats

# Given means and standard deviations for each metric
data = {
    "Dummy Random": {
        "accuracy": (0.5017, 0.0129),
        "f1": (0.4750, 0.0135),
        "specificity": (0.5276, 0.0123),
        "sensitivity": (0.4732, 0.0135)
    },
    "Decision Tree": {
        "accuracy": (0.6769, 0.0058),
        "f1": (0.6159, 0.0068),
        "specificity": (0.7980, 0.0079),
        "sensitivity": (0.5438, 0.0068)
    },
    "Random Forest": {
        "accuracy": (0.7102, 0.0070),
        "f1": (0.6461, 0.0087),
        "specificity": (0.8510, 0.0082),
        "sensitivity": (0.5554, 0.0088)
    },
    "MLP": {
        "accuracy": (0.7555, 0.0055),
        "f1": (0.7377, 0.0051),
        "specificity": (0.7859, 0.0240),
        "sensitivity": (0.7221, 0.0192)
    }
}

# Generate synthetic data based on means and standard deviations
n = 100

synthetic_data = {}

for model, metrics in data.items():
    synthetic_data[model] = {}
    for metric, (mean, std) in metrics.items():
        synthetic_data[model][metric] = np.random.normal(mean, std, n)

comparisons = [
    ("Dummy Random", "Decision Tree"),
    ("Dummy Random", "Random Forest"),
    ("Dummy Random", "MLP"),
    ("Decision Tree", "Random Forest"),
    ("Random Forest", "MLP")
]

metrics = ["accuracy", "f1", "specificity", "sensitivity"]

results = [
    ["Comparison", "Metric", "t-statistic", "p-value"]
]

for model1, model2 in comparisons:
    for metric in metrics:
        data1 = synthetic_data[model1][metric]
        data2 = synthetic_data[model2][metric]
    
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        results.append([f"{model1} vs {model2}", metric, f"{t_stat:.4f}", f"{p_value:.4e}"])

# Print results as a markdown table
print("| Comparison                              | Metric      | t-statistic | p-value |")
print("|-----------------------------------------|-------------|-------------|---------|")
for row in results[1:]:
    print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
