import requests
import os
import sys

FRAMEWORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, FRAMEWORK_DIR)

from config_loader import load_config

config = load_config()

# install doc2hpo api from https://github.com/stormliucong/Doc2Hpo2.0
def call_api_requests(method, text, api_key=None):
    '''Method:["actree", "scispacy", "gpt"]'''
    base_url = config.get("URLS", {}).get("DOC2HPO", "http://localhost:5010").rstrip("/")
    url = f"{base_url}/api/search/{method}"
    
    if method == 'gpt':
        data = {
            "text": text,
            "openaiKey": api_key,
        }
    else:
        data = {
            "text": text,
        }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        hpo_list = list(set([i[3].get('id') for i in result]))
        return hpo_list
    else:
        print("error:", response.status_code, response.text)
