import os
from google.cloud import storage
import re
import nltk
from nltk.corpus import stopwords
import sys
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_openai import OpenAIEmbeddings
from Backend.key import openAI,googledeet
# Add the project directory to the Python path

from Backend.key import hugembed, embedding_url
import requests
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from nltk.tokenize import sent_tokenize
class Rulebooks:
# Set the path to your service account key file
    googledeet = "path/to/your/service/account/key/file.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = googledeet

    @classmethod
    def _load_rules(cls, sport):
        # Initialize a client
        client = storage.Client()

        # Get the bucket
        bucket = client.get_bucket('sportrules')

        if sport == 'NFL':
            blob_name = 'NFLRuleBook2023.txt'
        elif sport == 'NBA':
            blob_name = 'NBARuleBook2023.txt'
        else:
            return "Sport not supported."

        # Get the blob (file) you want to pull
        blob = bucket.get_blob(blob_name)

        # Download the content of the file
        content = blob.download_as_string()

        return content.decode('utf-8')

    def NFLRule(sport: str) -> str:
        return sport._load_rules('NFL')
        
    def NBARule(sport : str) -> str:
        
        return sport._load_rules('NBA')
            


def generate_embedding(text:str) -> list[float]:
    response =requests.post(
        embedding_url,
        headers ={"Authorization": f"Bearer {hugembed}"},
        json ={"inputs": text} )

    if response.status_code != 200:
        raise ValueError(f"Request failed due to status code {response.status_code}: {response.text}")
    return response.json()

embeddings = OpenAIEmbeddings(
    api_key=openAI
)

def parse_rules(content):
    # Assuming content is bytes-like
    content = content.decode('utf-8')  # Decode to a string

    rules = []
    current_rule = None
    current_content = []

    for line in content.split('\n'):
        line = line.strip()

        if line.startswith('RULE'):
            # Start of a new rule
            if current_rule is not None:
                # Make sure there is some content before adding the rule
                if current_content:
                    current_rule['page_content'] = ' '.join(current_content)
                    # Generate embeddings for the content of the current rule
                    current_rule['Embeddings'] = embeddings.embed_query(current_rule['page_content'])
                    rules.append(current_rule)
                current_content = []

            rule_name = line.split(' ', 1)[1]
            current_rule = {'Index': len(rules) + 1, 'Rule': rule_name, 'page_content': ''}

        else:
            # Accumulate words until the next rule
            current_content.append(line)

    # Add the content of the last rule
    if current_rule and current_content:  # Check if there is a rule and it has content
        current_rule['page_content'] = ' '.join(current_content)
        # Generate embeddings for the content of the last rule
        current_rule['Embeddings'] = embeddings.embed_query(current_rule['page_content'])
        rules.append(current_rule)

    return rules



class CleanText():

    def remove_special_characters(text):
        text = text.decode("utf-8")  # Decode bytes-like object to string
        return re.sub(r'[^\w\s.]', '', text)
    def to_lowercase(text):
        return text.lower()

    def remove_stopwords(text):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)


    def tokenize_with_nltk(text):
        return sent_tokenize(text)
    
    @classmethod
    def CleanText(txt):
        lowertxt=CleanText.to_lowercase(txt)
        rms=CleanText.remove_special_characters(lowertxt)
        stopwwordtxt = CleanText.remove_stopwords(rms)
        token= CleanText.tokenize_with_nltk(stopwwordtxt)
        return token
                        

