import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This file contains all the different functions to create the graphs.
results_1 should be the results from the food dataset, results_2 is from the papers
"""

#root_dir = "./results"
query_pattern = r"query_results_(\d+)_(\w+)\.csv"
overall_pattern = r"overall_summary_(\d+)_(\w+)\.csv"

overall_keys = ["MAP", "Precision", "Recall", "F1_Score", "Correct"]
query_keys = ["Query", "Precision", "Recall", "F1_Score"]
ks = [1, 2, 3, 4, 5, 10, 15, 20, 30]


# Build a graph for the average over all the queries based on the key
def overall_graphs_key(root_dir, keys):
    for key in keys:
        distances_values = []
        map_values = []
        recall_values = []
        f1_score_values = []
        precision_values = []
        for k in ks:
            for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                for file in files:
                    if key in file and file.startswith("o") and (f'{k}_') in file:
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        #distances_values.append(df['Distance'])
                        map_values.append(df['MAP'])
                        recall_values.append(df['Recall'])
                        f1_score_values.append(df['F1_Score'])
                        precision_values.append(df['Precision'])
        # Create a separate plot for each key
        if key == "['tags', 'name', 'steps']":
            print(f'Precision: {precision_values}')
            print(f'Recall: {recall_values}')
            print(f'F1_Score: {f1_score_values}')
            print(f'MAP: {map_values}')

        plt.figure(figsize=(10, 6))
        plt.plot(ks, map_values, label='MAP')
        plt.plot(ks, precision_values, label='Precision')
        plt.plot(ks, recall_values, label='Recall')
        plt.plot(ks, f1_score_values, label='F1_Score')
        plt.xlabel('Number of neighbours (k)')
        plt.ylabel('Percentage (between 0 and 1)')
        plt.title(f'Different metrics using key: {key}')
        plt.ylim(0, 1)
        plt.xticks(ks)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{key}.pdf')
        print(f"Saved plot for key: {key}")
        plt.close()

# Build a graph for the average over all the queries 
def overall_graphs_k(root_dir, keys):
    for metric in overall_keys:
        metric_values = {str(k): [] for k in ks}

        for key in keys:
            for k in ks:
                for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                    for file in files:
                        if key in file and file.startswith("o") and (f'{k}_') in file:
                            filepath = os.path.join(root, file)
                            df = pd.read_csv(filepath)
                            #if metric == 'Distance':
                             #   metric_values[str(k)].append(df['Distance'].mean())
                            if metric == 'MAP':
                                metric_values[str(k)].append(df['MAP'].mean())
                            elif metric == 'Recall':
                                metric_values[str(k)].append(df['Recall'].mean())
                            elif metric == 'Precision':
                                metric_values[str(k)].append(df['Precision'].mean())                            
                            elif metric == 'F1_Score':
                                metric_values[str(k)].append(df['F1_Score'].mean())
                            elif metric == 'Correct':
                                metric_values[str(k)].append(df['Correct'].mean())

        plt.figure(figsize=(14, 10))
        bar_width = 0.35
        opacity = 0.8
        index = np.arange(len(keys))
        if metric == "Precision":
            print(f'Precision: {metric_values}')
        if metric == "F1_Score":
            print(f'F1_Score: {metric_values}')
        if metric == "Recall":
            print(f'Recall: {metric_values}')
        if metric == "MAP":
            print(f'MAP: {metric_values}')

        for i, k in enumerate(ks):
            plt.bar(index + i * bar_width / len(ks), metric_values[str(k)], bar_width / len(ks), alpha=opacity, label=str(k))

        plt.xlabel('Keys used to generate the documents.')
        plt.ylabel(f'percentage (between 0 and 1)')
        plt.title(f'Metric {metric} for all keys and all k values')
        plt.xticks(index + bar_width / 2, keys, rotation=15)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{metric}_hist.pdf')
        print(f"Saved plot for metric: {metric}")
        plt.close()

def overall_graphs_ngram_key(root_dir, keys):
    root_dir = "./results/ngram"
    for key in keys:
        distances_values = []
        precision_values = []
        recall_values = []
        f1_score_values = []
        map_values = []
        for k in ks:
            for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                for file in files:
                    if key in file and file.startswith("o") and (f'{k}_') in file:
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        precision_values.append(df['Precision'])
                        map_values.append(df['MAP'])
                        recall_values.append(df['Recall'])
                        f1_score_values.append(df['F1_Score'])
        plt.figure(figsize=(10, 6))
        plt.plot(ks, map_values, label='MAP')
        plt.plot(ks, precision_values, label='Precision')
        plt.plot(ks, recall_values, label='Recall')
        plt.plot(ks, f1_score_values, label='F1_Score')
        plt.xlabel('Number of neighbours (k)')
        plt.ylabel('Percentage (between 0 and 1)')
        plt.title(f'Different metrics using key: {key}')
        plt.ylim(0, 1)
        plt.xticks(ks)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{key}.pdf')
        print(f"Saved plot for key: {key}")
        plt.close()

def overall_graphs_ngram_k(root_dir, keys):
    #root_dir = "./results/ngram"
    for metric in overall_keys:
        metric_values = {str(k): [] for k in ks}

        for key in keys:
            for k in ks:
                for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                    for file in files:
                        if key in file and file.startswith("o") and (f'{k}_') in file:
                            filepath = os.path.join(root, file)
                            df = pd.read_csv(filepath)
                            #if metric == 'Distance':
                             #   metric_values[str(k)].append(df['Distance'].mean())
                            if metric == 'MAP':
                                metric_values[str(k)].append(df['MAP'].mean())
                            elif metric == 'Recall':
                                metric_values[str(k)].append(df['Recall'].mean())
                            elif metric == 'Precision':
                                metric_values[str(k)].append(df['Precision'].mean())                            
                            elif metric == 'F1_Score':
                                metric_values[str(k)].append(df['F1_Score'].mean())
                            elif metric == 'Correct':
                                metric_values[str(k)].append(df['Correct'].mean())

        plt.figure(figsize=(14, 10))
        bar_width = 0.35
        opacity = 0.8
        index = np.arange(len(keys))

        for i, k in enumerate(ks):
            plt.bar(index + i * bar_width / len(ks), metric_values[str(k)], bar_width / len(ks), alpha=opacity, label=str(k))

        plt.xlabel('Keys used to generate the documents.')
        plt.ylabel(f'percentage (between 0 and 1)')
        plt.title(f'Metric {metric} for all keys and all k values using 2-grams')
        plt.xticks(index + bar_width / 2, keys, rotation=15)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{metric}_hist.pdf')
        print(f"Saved plot for metric: {metric}")
        plt.close()

def svd_graphs(root_dir):
    keys = ["['tags', 'name', 'steps']"]
    svd_values = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
    overall_keys = ['MAP', 'Recall', 'Precision', 'F1_Score', 'Correct']

    for k in ks:
        plt.figure(figsize=(14, 10))
        for metric in overall_keys:
            metric_values = {key: [] for key in keys}

            for i in svd_values:
                root_dir = f"{root_dir}/svd_{i}"
                for key in keys:
                    filepath = os.path.join(root_dir, str(k), key, f"overall_summary_{k}_{keys[0]}.csv")
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        metric_values[key].append(df[metric].mean())

            for key in keys:
                plt.plot(svd_values, metric_values[key], label=f'{metric} {key}')

        plt.xlabel('svd components')
        plt.ylabel('Metrics')
        plt.title(f'Different metrics for k={k} using svd ')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.xticks(svd_values, rotation=15)
        plt.savefig(f'{root_dir}/k_{k}_overall_all_metrics.pdf')
        print(f"Saved plot for all metrics for k={k}")
        plt.close()

def plot_speed_average(root_dir, keys):
    for metric in overall_keys:
        metric_values = {str(k): [] for k in ks}

        for key in keys:
            for k in ks:
                for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                    for file in files:
                        if key in file and file.startswith("o") and (f'{k}_') in file:
                            filepath = os.path.join(root, file)
                            df = pd.read_csv(filepath)
                            if metric == "Speed":
                                metric_values[str(k)].append(df['Speed'].mean())


    plt.figure(figsize=(14, 10))
    bar_width = 0.35
    opacity = 0.8
    index = np.arange(len(keys))

    for i, k in enumerate(ks):
        plt.bar(index + i * bar_width / len(ks), metric_values[str(k)], bar_width / len(ks), alpha=opacity, label=str(k))

    plt.ylabel(f'Average retrieval speed in seconds')
    plt.xlabel('Keys used to generate the documents.')
    plt.title(f'Metric: {metric}')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=15)
    plt.savefig(f'{root_dir}/overall_{metric}.pdf')
    print(f"Saved plot for metric: {metric}")
    plt.close()

# Build a graph for the average over all the queries 
def overall_graphs_normal_k_sum(root_dir, keys):
    keys = ["['title', 'abstract']", "['title', 'abstract', 'full']"]
    for metric in overall_keys:
        metric_values = {str(k): [] for k in ks}

        for key in keys:
            for k in ks:
                for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                    for file in files:
                        if key in file and file.startswith("o") and (f'{k}_') in file:
                            filepath = os.path.join(root, file)
                            df = pd.read_csv(filepath)
                            if metric == 'Distance':
                                metric_values[str(k)].append(df['Distance'])
                            elif metric == 'MAP':
                                metric_values[str(k)].append(df['MAP'])
                            elif metric == 'Recall':
                                metric_values[str(k)].append(df['Recall'])
                            elif metric == 'Precision':
                                metric_values[str(k)].append(df['Precision'])
                            elif metric == 'F1_Score':
                                metric_values[str(k)].append(df['F1_Score'])
                            elif metric == 'Correct':
                                metric_values[str(k)].append(df['Correct'])
                            elif metric == 'Speed':
                                metric_values[str(k)].append(df['Speed'])

        plt.figure(figsize=(14, 10))
        for k in ks:
            plt.plot(keys, metric_values[str(k)], label=str(k))
        plt.xlabel('Keys')
        plt.ylabel(f'{metric}')
        plt.title(f'Metric: {metric}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=15)
        plt.savefig(f'{root_dir}/overall_{metric}.pdf')
        print(f"Saved plot for metric: {metric}")
        plt.close()

def overall_graphs_normal_key_sum(root_dir, keys):
    keys = ["['title', 'abstract']", "['title', 'abstract', 'full']"]
    for key in keys:
        distances_values = []
        map_values = []
        recall_values = []
        f1_score_values = []
        precision_values = []
        for k in ks:
            for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                for file in files:
                    if key in file and file.startswith("o") and (f'{k}_') in file:
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        distances_values.append(df['Distance'])
                        map_values.append(df['MAP'])
                        recall_values.append(df['Recall'])
                        f1_score_values.append(df['F1_Score'])
                        precision_values.append(df['Precision'])
        # Create a separate plot for each key
        plt.figure(figsize=(10, 6))
        plt.plot(ks, distances_values, label='Distances')
        plt.plot(ks, map_values, label='MAP')
        plt.plot(ks, precision_values, label='Precision')
        plt.plot(ks, recall_values, label='Recall')
        plt.plot(ks, f1_score_values, label='F1_Score')
        plt.xlabel('k')
        plt.ylabel('Metrics')
        plt.title(f'Key: {key}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{key}.pdf')
        print(f"Saved plot for key: {key}")
        plt.close()
'''
def overall_graphs_normal_key(root_dir, keys):
    for key in keys:
        distances_values = []
        map_values = []
        recall_values = []
        f1_score_values = []
        precision_values = []
        for k in ks:
            for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                for file in files:
                    if key in file and file.startswith("o") and (f'{k}_') in file:
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        distances_values.append(df['Distance'])
                        map_values.append(df['MAP'])
                        recall_values.append(df['Recall'])
                        f1_score_values.append(df['F1_Score'])
                        precision_values.append(df['Precision'])
        # Create a separate plot for each key
        plt.figure(figsize=(10, 6))
        plt.plot(ks, distances_values, label='Distances')
        plt.plot(ks, map_values, label='MAP')
        plt.plot(ks, precision_values, label='Precision')
        plt.plot(ks, recall_values, label='Recall')
        plt.plot(ks, f1_score_values, label='F1_Score')
        plt.xlabel('k')
        plt.ylabel('Metrics')
        plt.title(f'Key: {key}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{key}.pdf')
        print(f"Saved plot for key: {key}")
        plt.close()
'''
def overall_graphs_normal_k_word(root_dir, keys):
    overall_keys = ["Distance", "MAP", "Precision", "Recall", "F1_Score", "Correct"]
    for metric in overall_keys:
        metric_values = {str(k): [] for k in ks}

        for key in keys:
            for k in ks:
                for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                    for file in files:
                        if key in file and file.startswith("o") and (f'{k}_') in file:
                            filepath = os.path.join(root, file)
                            df = pd.read_csv(filepath)
                            if metric == 'Distance':
                                metric_values[str(k)].append(df['Distance'])
                            elif metric == 'MAP':
                                metric_values[str(k)].append(df['MAP'])
                            elif metric == 'Recall':
                                metric_values[str(k)].append(df['Recall'])
                            elif metric == 'Precision':
                                metric_values[str(k)].append(df['Precision'])
                            elif metric == 'F1_Score':
                                metric_values[str(k)].append(df['F1_Score'])
                            elif metric == 'Correct':
                                metric_values[str(k)].append(df['Correct'])

        plt.figure(figsize=(14, 10))
        for k in ks:
            plt.plot(keys, metric_values[str(k)], label=str(k))
        plt.xlabel('Keys')
        plt.ylabel(f'{metric}')
        plt.title(f'Metric: {metric}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=15)
        plt.savefig(f'{root_dir}/overall_{metric}.pdf')
        print(f"Saved plot for metric: {metric}")
        plt.close()

def overall_graphs_normal_key(root_dir, keys):
    ks = [1, 2, 3, 4, 5, 10, 15, 20]
    if 'sum' in root_dir:
        keys = ["['sum']"]
    for key in keys:
        distances_values = []
        map_values = []
        recall_values = []
        f1_score_values = []
        precision_values = []
        for k in ks:
            for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                for file in files:
                    if key in file and file.startswith("o") and (f'{k}_') in file:
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        distances_values.append(df['Distance'])
                        map_values.append(df['MAP'])
                        recall_values.append(df['Recall'])
                        f1_score_values.append(df['F1_Score'])
                        precision_values.append(df['Precision'])
        # Create a separate plot for each key
        plt.figure(figsize=(10, 6))
        plt.plot(ks, distances_values, label='Distances')
        plt.plot(ks, map_values, label='MAP')
        plt.plot(ks, precision_values, label='Precision')
        plt.plot(ks, recall_values, label='Recall')
        plt.plot(ks, f1_score_values, label='F1_Score')
        plt.xlabel('Number of neighbours (k)')
        plt.ylabel('Percentage (between 0 and 1)')
        if 'sum' in root_dir:
            plt.title(f'Different metrics using compressed documents.')
        else:
            plt.title(f'Different metrics using key: {key}')
        plt.legend()
        plt.ylim(0, 1)
        plt.xticks(ks)
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{key}.pdf')
        print(f"Saved plot for key: {key}")
        plt.close()

def overall_graphs_normal_k(root_dir, keys):
    if 'sum' in root_dir:
        keys = ["['sum']"]
    for metric in overall_keys:
        metric_values = {str(k): [] for k in ks}

        for key in keys:
            for k in ks:
                for root, dirs, files in os.walk(f'{root_dir}/{k}/{key}'):
                    for file in files:
                        if key in file and file.startswith("o") and (f'{k}_') in file:
                            filepath = os.path.join(root, file)
                            df = pd.read_csv(filepath)
                            if metric == 'Distance':
                                metric_values[str(k)].append(df['Distance'].mean())
                            elif metric == 'MAP':
                                metric_values[str(k)].append(df['MAP'].mean())
                            elif metric == 'Recall':
                                metric_values[str(k)].append(df['Recall'].mean())
                            elif metric == 'Precision':
                                metric_values[str(k)].append(df['Precision'].mean())
                            elif metric == 'F1_Score':
                                metric_values[str(k)].append(df['F1_Score'].mean())
                            elif metric == 'Correct':
                                metric_values[str(k)].append(df['Correct'].mean())
                            elif metric == 'Speed':
                                metric_values[str(k)].append(df['Speed'].mean())

        plt.figure(figsize=(14, 10))
        bar_width = 0.35
        opacity = 0.8
        index = np.arange(len(keys))

        for i, k in enumerate(ks):
            plt.bar(index + i * bar_width / len(ks), metric_values[str(k)], bar_width / len(ks), alpha=opacity, label=str(k))

        plt.xlabel('Keys used to generate the documents.')
        if metric == "Speed":
            plt.ylabel(f'retrieval time (seconds)')
        else:
            plt.ylabel(f'percentage (between 0 and 1)')
        title = f'Metric {metric} for all keys and all k values'
        if "word" in root_dir:
            title += " with Word2Vec"
        if "fast" in root_dir:
            title += "with accelerated NN"
        if 'sum' in root_dir:
            title += " with compressed documents"
        plt.title(title)
        plt.xticks(index + bar_width / 2, keys, rotation=15)
        if metric != "Speed":
            plt.ylim(0, 1)
        else:
            plt.ylim(0, 0.004)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{root_dir}/overall_{metric}_hist.pdf')
        print(f"Saved plot for metric: {metric}")
        plt.close()


def build_graphs():
    # Word-level
    keys = ["['tags', 'name', 'steps', 'ingredients', 'description']",
        "['tags', 'name', 'steps', 'ingredients']",
        "['tags', 'name', 'steps']",
        "['name', 'ingredients']",
        "['tags', 'name']",
        "['name']"]
    overall_graphs_k("./results_1", keys)
    overall_graphs_key("./results_1", keys)
    overall_graphs_ngram_k("./results_1/ngram", keys)
    overall_graphs_ngram_key("./results_1/ngram", keys)
    svd_graphs("./results_1/svd")

    # Neural document embeddings
    keys = ["['title', 'author']", "['title', 'abstract', 'author']", "['title', 'full', 'author']"]
    ks = [1, 2, 3, 4, 5, 10, 15, 20]

    
    overall_graphs_normal_k("./results_2/", ["['title', 'abstract', 'author']"])
    overall_graphs_normal_key("./results_2/", ["['title', 'abstract', 'author']"])

    overall_graphs_normal_k("./results_2/word/1", ["['title', 'abstract', 'author']"])
    overall_graphs_normal_key("./results_2/word/1", ["['title', 'abstract', 'author']"])
    overall_graphs_normal_k_sum("./results_2/sum", [])
    overall_graphs_normal_key_sum("./results_2/sum", [])
    plot_speed_average("./results_2", keys)
    plot_speed_average("./results_2/fast", keys)
    overall_graphs_normal_key("./results_2/fast", keys)
    overall_graphs_normal_k("./results_2/fast", keys)
    overall_graphs_normal_k("./results_2/sum", keys)
    overall_graphs_normal_key("./results_2/sum", keys)
#build_graphs()


keys = ["['tags', 'name', 'steps', 'ingredients', 'description']",
    "['tags', 'name', 'steps', 'ingredients']",
    "['tags', 'name', 'steps']",
    "['name', 'ingredients']",
    "['tags', 'name']",
    "['name']"]
overall_graphs_k("./results_1/micro", keys)
overall_graphs_key("./results_1/micro", keys)