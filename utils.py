import json
import re

def get_relations_gpt(data):
    relations = set()
    for row in data:
        triples = row["chat-gpt"]
        for triple in triples:
            relations.add(triple[1])
    return len(relations)

def get_entities_gpt(data):
    count = 0
    for row in data:
        entities = set()
        triples = row["chat-gpt"]
        for triple in triples:
            entities.add(triple[0])
            entities.add(triple[2])
        count += len(entities)
    return count

def get_relations_rebel(data):
    relations = set()
    for row in data:
        triples = row["gpt_answer"]
        for triple in triples:
            item_lst = re.findall(r'<([^>]*)>', triple[0])
            relations.add(item_lst[1])
    return len(relations)


def get_entities_rebel(data):
    count = 0
    for row in data:
        entities = set()
        triples = row["gpt_answer"]
        for triple in triples:
            item_lst = re.findall(r'<([^>]*)>', triple[0])
            entities.add(item_lst[0])
            entities.add(item_lst[2])
        count += len(entities)
    return count
def get_relations_rebel_filtered(data):
    relations = set()
    for row in data:
        triples = row["gpt_answer"]
        for triple in triples:
            if triple[1]==1:
                item_lst = re.findall(r'<([^>]*)>', triple[0])
                relations.add(item_lst[1])
    return len(relations)

def get_entities_rebel_filtered(data):
    count = 0
    for row in data:
        entities = set()
        triples = row["gpt_answer"]
        for triple in triples:
            if triple[1]==1:
                item_lst = re.findall(r'<([^>]*)>', triple[0])
                entities.add(item_lst[0])
                entities.add(item_lst[2])
        count += len(entities)
    return count

gpt_data = json.load(open("data/data_gpt4.json", "r", encoding="utf-8"))
rebel_data = json.load(open("results/results_final.json", "r", encoding="utf-8"))

print(get_relations_gpt(gpt_data))
print(get_relations_rebel(rebel_data))
print(get_relations_rebel_filtered(rebel_data))

print(get_entities_gpt(gpt_data))
print(get_entities_rebel(rebel_data))
print(get_entities_rebel_filtered(rebel_data))



