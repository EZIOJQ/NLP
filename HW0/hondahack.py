import requests
import json
url = "https://ipacdev.hondaresearch.com:8443/hackathon/hondadsrc/evtwarn?device=2004&trip=12"

headers = {
    'key' : "AC85FK223FNP90AK72",
    'cache-control' : "no-cache"
}

response = requests.request("GET", url, headers=headers)
with open("response.json",'w') as r:
	json_obj = json.loads(response.text)
	r.write(json.dumps(json_obj, indent=4))