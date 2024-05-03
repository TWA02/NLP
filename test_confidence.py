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

    def query_confidence(self, text, max_tokens=50):
        prompt = f"Given the answers and citations in this text: \"{text}\". On a scale of 0 to 100, give a confidence score on the accuracy and relevance of these citations? Provide only a number and nothing else."
        logger.info(f"Sending prompt to model: {prompt}")

        try:
            response = openai.ChatCompletion.create(
                    engine="nlp",
                    model="gpt-35-turbo",
                    messages=[{'role': 'system', 'content': "You are a helpful assistant."},
                            {'role': 'user', 'content': prompt}],
                    temperature=0.5,
                    max_tokens=max_tokens,
                    top_p=1.0,
                    stop=None
            )
            confidence_response = response['choices'][0]['message']['content']
            logger.info(f"Received response: {confidence_response}")
            return confidence_response
        except Exception as e:
            logger.error(f"Failed to query the model: {e}")
            return None

def process_data(data, output_file):
    results = []

    for entry in data:
        question = entry["question"]
        answers = ', '.join([ans for sublist in entry["answer"] for ans in sublist])
        docs = "; ".join([f"Document {doc['id']} Title: {doc['title']} Text: {doc['text']}" for doc in entry["docs"]])
        combined_text = f"Question: {question} Answers: {answers} {docs}"

        args = Namespace(openai_api=True, azure=True, model='gpt-3.5-turbo-0301', text=combined_text)
        tester = AzureModelConfidenceTester(args)
        confidence = tester.query_confidence(combined_text)

        entry.update({
            "confidence_score": float(confidence) if confidence and confidence.isdigit() else None,
        })
        results.append(entry)

    # Save the updated data with confidence and F1 scores to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    # Prepare data for correlation analysis
    df = pd.DataFrame.from_records(results, columns=['confidence_score'])
    

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
            if "data" in data:
                process_data(data['data'], args.output)
            else:
                logger.error("JSON does not contain 'data' key")
