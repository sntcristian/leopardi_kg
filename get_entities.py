import requests
import json
import time
from tqdm import tqdm

with open("data/data_gpt4_refined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

url = 'https://query.wikidata.org/sparql'
viaf_to_wiki = dict()
geonames_to_wiki = dict()
entities = {}

pbar = tqdm(total=len(data))
for item in data:
    persons = item["persons"]
    places = item["places"]
    for person in persons:
        viaf_id = '"'+person["key"]+'"'
        name = person["persName"]
        q_id = None
        if viaf_id in viaf_to_wiki.keys():
            q_id = viaf_to_wiki[viaf_id]
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
                viaf_to_wiki[viaf_id]=q_id
        if q_id!=None:
            if q_id not in entities:
                entities[q_id]=[name]
            else:
                entities[q_id].append(name)

    for place in places:
        geonames_id = '"'+place["key"]+'"'
        name = place["placeName"]
        if geonames_id in geonames_to_wiki.keys():
            q_id = geonames_to_wiki[geonames_id]
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
                geonames_to_wiki[geonames_id]=q_id
        if q_id!=None:
            if q_id not in entities:
                entities[q_id]=[name]
            else:
                entities[q_id].append(name)

    pbar.update(1)

pbar.close()

with open("entities_refined.json", "w", encoding="utf-8") as f:
    json.dump(entities, f, indent=4, ensure_ascii=False)



