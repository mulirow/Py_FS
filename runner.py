from Py_FS.wrapper.nature_inspired import GA, PSO, WOA, RDA, RDA_Mod_1, RDA_Mod_2, RDA_Mod_3
from Py_FS.datasets import get_dataset
import numpy as np
import pandas as pd

def main():
    num_runs = 30
    features = ['num_features', 'num_agents', 'max_iter', 'execution_time', 'convergence_curve', 'best_agent', 'best_fitness', 'best_accuracy', 'final_population', 'final_fitness', 'final_accuracy']

    # Load a dataset
    data = get_dataset("Hill-valley")

    # Results vector
    results = [[] for i in range(num_runs)]

    # Run the optimizer
    for i in range(num_runs):
        res = RDA(20, 100, data.data, data.target, save_conv_graph=False)

        # Handle results
        res.convergence_curve = list(res.convergence_curve.values())[0]
        props = [getattr(res, features[j]).tolist() if isinstance(getattr(res, features[j]), np.ndarray) else getattr(res, features[j]) for j in range(len(features))]
        results[i] = props

    # Convert the results list to a DataFrame
    df = pd.DataFrame(results, columns=features)

    # Save the DataFrame to a CSV file, with each parameter in a separate column and each run in a separate row
    df.to_csv("run/rda_hill.csv", index=False)

if __name__ == "__main__":
    main()