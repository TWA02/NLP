import openai
import logging
import os
import json
import argparse
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
                    engine="nlp",  # Ensure this is the correct Azure engine ID
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

def process_data(data):
    
    confidences = []

    for entry in data:
        question = entry["question"]
        answers = ', '.join([ans for sublist in entry["answers"] for ans in sublist])
        docs = "; ".join([f"Document {doc['id']} Title: {doc['title']} Text: {doc['text']}" for doc in entry["docs"]])
        combined_text = f"Question: {question} Answers: {answers} {docs}"

        args = Namespace(openai_api=True, azure=True, model='gpt-3.5-turbo-0301', text=combined_text)
        tester = AzureModelConfidenceTester(args)
        confidence = tester.query_confidence(combined_text)
        if confidence:
            try:
                # Attempt to convert the confidence response to a float
                conf_value = float(confidence)
                print(f"Model Confidence for entry ID {entry['id']}: {confidence}")
                confidences.append(conf_value)
                entry["confidence_score"] = conf_value
            except ValueError:
                # Handle the case where conversion fails
                print(f"Received non-numeric confidence response for entry ID {entry['id']}: {confidence}")
                confidences.append(None)
                entry["confidence_score"] = None
        else:
            print(f"Failed to obtain confidence for entry ID {entry['id']}.")
            confidences.append(None)
        
    # Save the updated data with confidence scores to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    # Calculate average confidence, ignoring None values
    valid_confidences = [conf for conf in confidences if conf is not None]
    if valid_confidences:
        average_confidence = sum(valid_confidences) / len(valid_confidences)
        print(f"Average Confidence: {average_confidence}")
    else:
        print("No valid confidence scores available to calculate an average.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process JSON data and query model confidence on Azure.')
    parser.add_argument('--file', type=str, required=True, help='File path of the JSON data file.')
    parser.add_argument('--output', type=str, required=True, help='Output file path to save the JSON data with confidence scores.')

    args = parser.parse_args()

    # Load the data from the JSON file
    if not os.path.exists(args.file):
        logger.error("File does not exist")
    else:
        with open(args.file, 'r') as file:
            data = json.load(file)
            if "data" in data:
                process_data(data['data'])
            else:
                logger.error("JSON does not contain 'data' key")
    
