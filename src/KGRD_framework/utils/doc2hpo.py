import requests
# install doc2hpo api from https://github.com/stormliucong/Doc2Hpo2.0
def call_api_requests(method, text, api_key=None):
    '''Method:["actree", "scispacy", "gpt"]'''
    url = f"http://localhost:5010/api/search/{method}"
    
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