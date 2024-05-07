import openai
import logging
import os
import json
import argparse
import numpy as np
import pandas as pd
from argparse import Namespace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureModelConfidenceTester:
    def __init__(self, args):
        self.args = args
        if args.openai_api:
            import openai 
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

            if args.azure:
                openai.api_key = OPENAI_API_KEY
                openai.api_base = OPENAI_API_BASE
                openai.api_type = 'azure'
                openai.api_version = '2023-05-15' 
            else: 
                openai.api_key = OPENAI_API_KEY
                openai.organization = OPENAI_ORG_ID


    def query_confidence(self, text, metric):
        definitions = {
            'relevance': "how relevant the citations are to the answers provided.",
            'accuracy': "how accurate the citations are in representing their source materials.",
            'correctness': "how correct the answers are in addressing the questions."
        }
        prompt = f"Given the text: \"{text}\" and considering {definitions[metric]}, on a scale of 0 to 100, provide a numerical score for {metric}."
        logger.info(f"Sending prompt for {metric}: {prompt}")

        try:
            response = openai.ChatCompletion.create(
                engine="nlp",
                model="gpt-35-turbo",
                messages=[{'role': 'system', 'content': "You are a numerical assistant and should only respond with a number."},
                          {'role': 'user', 'content': prompt}],
                temperature=0.5,
                max_tokens=10,
                top_p=1.0,
                stop=None
            )
            confidence_response = response['choices'][0]['message']['content'].strip()
            score = float(confidence_response) if confidence_response.isdigit() else None
            logger.info(f"Received numerical response for {metric}: {score}")
            return score
        except Exception as e:
            logger.error(f"Failed to query the model for {metric}: {e}")
            return None


def process_data(data, output_prefix):
    results = []

    for entry in data:
        question = entry["question"]
        answers = ''.join([ans for sublist in entry["answer"] for ans in sublist])
        docs = "; ".join([f"Document {doc['id']} Title: {doc['title']} Text: {doc['text']}" for doc in entry["docs"]])
        combined_text = f"Question: {question} Answers: {answers} {docs}"

        args = Namespace(openai_api=True, azure=True, model='gpt-3.5-turbo-0301', text=combined_text)
        tester = AzureModelConfidenceTester(args)

        metrics = {'relevance': None, 'accuracy': None, 'correctness': None}
        for metric in metrics.keys():
            metrics[metric] = tester.query_confidence(combined_text, metric)

        entry.update(metrics)
        results.append(entry)

    with open(output_prefix + '.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON data and query model confidence on Azure.')
    parser.add_argument('--file', type=str, required=True, help='File path of the JSON data file.')
    parser.add_argument('--output', type=str, required=True, help='Output file path to save the JSON data with confidence and F1 scores.')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error("File does not exist")
    else:
        with open(args.file, 'r') as file:
            data = json.load(file)
            process_data(data, args.output)