import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

class Metrics:
    def __init__(self, input_json_path):
        self.input_json_path = input_json_path
        self.output_dir = self.create_output_dir(input_json_path)
        self.data = None

    def create_output_dir(self, path):
        base_name = os.path.basename(path)
        dir_name = base_name.replace('.json', '')
        output_dir = os.path.join(os.getcwd(), dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def load_data(self):
        with open(self.input_json_path, 'r') as file:
            self.data = json.load(file)

    def calculate_metrics(self):
        if not self.data:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
        
        for entry in self.data:
            docs = entry.get('docs', [])
            if docs:
                average_relevance_score = sum(doc['score'] for doc in docs) / len(docs)
                entry['average_relevance_score'] = average_relevance_score

            recall = entry.get('recall')
            precision = entry.get('precision')
            if recall is not None and precision is not None:
                if recall + precision != 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0
                entry['f1_score'] = f1_score

    def save_metrics(self, output_file='metrics.json'):
        if not self.data:
            raise ValueError("Data not loaded or metrics not calculated. Please load data and calculate metrics first.")
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as file:
            json.dump(self.data, file)

    def plot_metrics(self):
        if not self.data:
            raise ValueError("Data not loaded or metrics not calculated. Please load data and calculate metrics first.")

        df = pd.DataFrame(self.data)
        df_non_zero_f1 = df[df['f1_score'] > 0]  # Filter out entries with F1 score of 0

        # Define colors for the plots
        color_palette = sns.color_palette("husl", 5)

        # Function to plot regression line and calculate R^2
        def plot_regression(x, y, data, ax):
            model = LinearRegression().fit(data[[x]], data[y])
            predictions = model.predict(data[[x]])
            ax.plot(data[x], predictions, color='red', label=f'R² = {r2_score(data[y], predictions):.2f}')
            ax.legend()

        # Rouge Score vs Correctness Score
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='rouge_score', y='correctness', data=df_non_zero_f1, ax=ax, palette=color_palette)
        plot_regression('rouge_score', 'correctness', df_non_zero_f1, ax)
        ax.set_title('Rouge Score vs Correctness Score')
        ax.set_xlabel('Rouge Score')
        ax.set_ylabel('Correctness Confidence')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        plt.savefig(os.path.join(self.output_dir, 'rouge_correct.png'))
        plt.show()

        # Average Relevance Score vs Relevance
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='average_relevance_score', y='relevance', data=df_non_zero_f1, ax=ax, palette=color_palette)
        plot_regression('average_relevance_score', 'relevance', df_non_zero_f1, ax)
        ax.set_title('Average Relevance Score vs Relevance Confidence')
        ax.set_xlabel('Average Given Relevance Score')
        ax.set_ylabel('Relevance Confidence')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        plt.savefig(os.path.join(self.output_dir, 'relevance.png'))
        plt.show()

        # Accuracy vs F1 Score
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='f1_score', y='accuracy', data=df_non_zero_f1, ax=ax, palette=color_palette)
        plot_regression('f1_score', 'accuracy', df_non_zero_f1, ax)
        ax.set_title('Accuracy vs F1 Score')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('F1 Score')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        plt.savefig(os.path.join(self.output_dir, 'f1_accuracy.png'))
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process some JSON files.")
    parser.add_argument('input_json_path', type=str, help='Path to the input JSON file')
    args = parser.parse_args()

    metrics_calculator = Metrics(args.input_json_path)
    metrics_calculator.load_data()
    metrics_calculator.calculate_metrics()
    metrics_calculator.save_metrics()
    metrics_calculator.plot_metrics()

if __name__ == "__main__":
    main()
