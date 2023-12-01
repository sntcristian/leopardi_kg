import requests
import json
import time
from tqdm import tqdm

with open("data/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

url = 'https://query.wikidata.org/sparql'
names = set()
entities = {}

pbar = tqdm(total=len(data))
for item in data:
    persons = item["persons"]
    places = item["places"]
    for person in persons:
        viaf_id = '"'+person["key"]+'"'
        name = person["persName"]
        if name in names:
            continue
        else:
            query = '''
            SELECT ?item WHERE {
                ?item wdt:P214 '''+viaf_id+'''
            }
            LIMIT 10
            '''
            r = requests.get(url, headers={'User-Agent': 'LeopardiBot'}, params={'format': 'json', 'query': query})
            result = r.json()["results"]["bindings"]
            if len(result)>0:
                q_id = result[0]["item"]["value"]
                entities[q_id]=name
                names.add(name)

    for place in places:
        geonames_id = '"'+place["key"]+'"'
        name = place["placeName"]
        if name in names:
            continue
        else:
            query = '''
            SELECT ?item WHERE {
                ?item wdt:P1566 '''+geonames_id+'''
            }
            LIMIT 10
            '''
            r = requests.get(url, headers={'User-Agent': 'LeopardiBot'}, params={'format': 'json', 'query': query})
            result = r.json()["results"]["bindings"]
            if len(result)>0:
                q_id = result[0]["item"]["value"]
                entities[q_id]=name
                names.add(name)
    pbar.update(1)

pbar.close()

with open("entities.json", "w", encoding="utf-8") as f:
    json.dump(entities, f, indent=4, ensure_ascii=False)



