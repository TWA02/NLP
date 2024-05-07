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

        # Prepare the dataframe, filtering out unnecessary data
        df = pd.DataFrame(self.data).dropna(subset=['f1_score', 'rouge_score', 'correctness', 'average_relevance_score', 'relevance', 'accuracy'])
        df_non_zero_f1 = df[df['f1_score'] > 0]  # Filter out entries with F1 score of 0

        # Check if there's any data left after filtering
        if df_non_zero_f1.empty:
            print("No valid data available for plotting.")
            return

        # Generate and save each plot
        self.create_plot('rouge_score', 'correctness', df_non_zero_f1, 'Rouge Score vs Correctness Score', 'Rouge Score', 'Correctness Confidence', 'rouge_correct.png')
        self.create_plot('average_relevance_score', 'relevance', df_non_zero_f1, 'Average Relevance Score vs Relevance', 'Average Given Relevance Score', 'Relevance Confidence', 'relevance.png')
        self.create_plot('f1_score', 'accuracy', df_non_zero_f1, 'Accuracy vs F1 Score', 'Accuracy', 'F1 Score', 'f1_accuracy.png')


    # Define a function to plot regression line and calculate R^2
    def plot_regression(self, x, y, data, ax):
        if data.empty or len(data[x].dropna()) == 0 or len(data[y].dropna()) == 0:
            print("Data for regression is empty or NaN.")
            return  # Exit if there is no data to plot

        # Prepare data: remove any NaN values that may interfere with regression calculation
        data = data.dropna(subset=[x, y])
        
        model = LinearRegression()
        X = data[[x]]  # Feature matrix
        y_true = data[y]  # Target vector
        model.fit(X, y_true)
        predictions = model.predict(X)
        
        # Plotting the regression line
        ax.plot(data[x], predictions, color='red', label=f'R² = {r2_score(y_true, predictions):.2f}')
        ax.legend()

    def create_plot(self, x, y, data, title, xlabel, ylabel, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x=x, y=y, data=data, ax=ax)
        self.plot_regression(x, y, data, ax)  # This calls the regression line plotting
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_xlim(left=0)  # Comment out or remove this line
        # ax.set_ylim(bottom=0)  # Comment out or remove this line
        plt.savefig(os.path.join(self.output_dir, filename))
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
