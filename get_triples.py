from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import requests

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# questo codice usa il modello mREBEL di Babelscape
# per documentazione vedi: https://huggingface.co/Babelscape/mrebel-large
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


# Load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1,
}

model.to(device)

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

properties = set()
with open("data/data_gpt4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pbar = tqdm(total=len(data))

results = []

for row in data:
    result_entry = dict()
    result_entry["id_doc"]=row["id_doc"]
    gpt_triples = row["chat-gpt"]
    output_triples_set = set()
    for triple in gpt_triples:
        prop = triple[1]
        new_prop = ""
        for char in prop:
            if char==":":
                continue
            elif char.islower() == True:
                new_prop+=char
            else:
                new_prop+=" "+char.lower()
        triple_string1 = " ".join([triple[0], new_prop, triple[2]])
        model_inputs = tokenizer(triple_string1, max_length=256, padding=True, truncation=True, return_tensors='pt')
        generated_tokens = model.generate(
            model_inputs["input_ids"].to(model.device),
            attention_mask=model_inputs["attention_mask"].to(model.device),
            **gen_kwargs,
        )

        # Extract text
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        # Extract triplets
        for sentence in decoded_preds:
            triples = extract_triplets(sentence)
            for pred_triple in triples:
                if "date of birth" in pred_triple["type"] or "date of death" in pred_triple["type"]:
                    continue
                elif (pred_triple["head"]==triple[0] or pred_triple["head"]==triple[2]) \
                        and pred_triple["head"]!=pred_triple["tail"]:
                    triple_string2 = "<" + pred_triple["head"] + "> <" + pred_triple["type"] + "> <" + pred_triple[
                        "tail"] + ">"
                    embedding1 = sentence_transformer.encode(triple_string1, convert_to_tensor=True)
                    embedding2 = sentence_transformer.encode(triple_string2, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(embedding1, embedding2)
                    if similarity.item() >= 0.9:
                        properties.add(pred_triple["type"])
                        output_triples_set.add((triple_string2, 1))
                    else:
                        output_triples_set.add((triple_string2, 0))
    triples_lst = list(output_triples_set)
    result_entry["gpt_answer"] = triples_lst
    results.append(result_entry)
    pbar.update(1)



with open("results/results_final.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

url = 'https://query.wikidata.org/sparql'

property_to_id = {}
for property in properties:
    query = '''
    SELECT ?prop
    WHERE
    {
      ?prop wikibase:directClaim ?a .
      ?prop rdfs:label ?propLabel.
      filter(lang(?propLabel) = "en")
      filter(regex(?propLabel, "^'''+property+'''$", "i")).
    }
    '''
    r = requests.get(url, headers={'User-Agent': 'LeopardiBot'}, params={'format': 'json', 'query': query})
    result = r.json()["results"]["bindings"]
    if len(result) > 0:
        q_id = result[0]["prop"]["value"]
        print(q_id)
        property_to_id[property]=q_id

with open("data/properties_final.json", "w", encoding="utf-8") as f:
    json.dump(property_to_id, f, indent=4, ensure_ascii=False)


