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
    "num_return_sequences": 3,
}

# for mREBEL
# tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", src_lang="en_XX", tgt_lang="tp_XX")
# to set Italian as source
# tokenizer._src_lang = "it_XX"
# tokenizer.cur_lang_code_id = tokenizer.convert_tokens_to_ids("it_XX")
# tokenizer.set_src_lang_special_tokens("it_XX")
# model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
# model.to(device)
model.to(device)
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
    "forced_bos_token_id": None,
}


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
                triple_string2 = "<" + pred_triple["head"] + "> <" + pred_triple["type"] + "> <" + pred_triple[
                    "tail"] + ">"
                embedding1 = sentence_transformer.encode(triple_string1, convert_to_tensor=True)
                embedding2 = sentence_transformer.encode(triple_string2, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embedding1, embedding2)
                if similarity.item() >= 0.9 and (pred_triple["head"]==triple[0] or pred_triple["head"]==triple[2]):

                    properties.add(pred_triple["type"])
                    output_triples_set.add(triple_string2)
    triples_lst = list(output_triples_set)
    result_entry["gpt_answer"] = triples_lst
    results.append(result_entry)
    pbar.update(1)
    
    # To get triples from raw text
    # result_entry["text"]=row["text"]
    # raw_text = row["text"]
    # text = raw_text
    # model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
    # generated_tokens = model.generate(
    #     model_inputs["input_ids"].to(device),
    #     attention_mask=model_inputs["attention_mask"].to(device),
    #     decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
    #     **gen_kwargs,
    # )
    # # Extract text
    # decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    # # Extract triplets
    # output_triples_set = set()
    # for sentence in decoded_preds:
    #     triples = extract_triplets_typed(sentence)
    #     for pred_triple in triples:
    #         if pred_triple["head"].startswith("Lettera di Leopardi") or pred_triple["tail"].startswith("Lettera di "
    #                                                                                                 "Leopardi"):
    #             continue
    #         else:
    #             triple_string = "<" + pred_triple["head"] + "> <" + pred_triple["type"] + "> <" + pred_triple["tail"] + ">"
    #             output_triples_set.add(triple_string)
    # triples_lst = list(output_triples_set)
    # result_entry["raw_text"] = triples_lst
    # To generate triples from all triples
    # gpt_triples = [" ".join(triple) for triple in gpt_triples]
    # text = ". ".join(gpt_triples)
    # model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')
    # generated_tokens = model.generate(
    #     model_inputs["input_ids"].to(model.device),
    #     attention_mask=model_inputs["attention_mask"].to(model.device),
    #     decoder_start_token_id = tokenizer.convert_tokens_to_ids("tp_XX"),
    #     **gen_kwargs,
    # )
    # # Extract text
    # decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    # # Extract triplets
    # output_triples_set = set()
    # for sentence in decoded_preds:
    #     triples = extract_triplets_typed(sentence)
    #     for pred_triple in triples:
    #         triple_string = "<" + pred_triple["head"] + "> <" + pred_triple["type"] + "> <" + pred_triple[
    #             "tail"] + ">"
    #         properties.add(pred_triple["type"])
    #         output_triples_set.add(triple_string)
    # triples_lst = list(output_triples_set)
    # result_entry["gpt_answer"]=triples_lst




with open("results/results_gpt4.json", "w", encoding="utf-8") as f:
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

with open("data/properties_gpt4.json", "w", encoding="utf-8") as f:
    json.dump(property_to_id, f, indent=4, ensure_ascii=False)


