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
def extract_triplets_typed(text):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets


# for mREBEL
tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", src_lang="en_XX", tgt_lang="tp_XX")
# to set Italian as source
tokenizer._src_lang = "it_XX"
tokenizer.cur_lang_code_id = tokenizer.convert_tokens_to_ids("it_XX")
tokenizer.set_src_lang_special_tokens("it_XX")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
model.to(device)

gen_kwargs = {
    "max_length": 256,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 3,
    "forced_bos_token_id": None,
}


properties = set()
with open("data/data_gpt4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pbar = tqdm(total=len(data))

results = []

for row in data:
    result_entry = dict()
    result_entry["id_doc"]=row["id_doc"]
    output_triples_set = set()
    # To get triples from raw text
    result_entry["text"]=row["text"]
    raw_text = row["text"]
    text = raw_text
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(device),
        attention_mask=model_inputs["attention_mask"].to(device),
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
        **gen_kwargs,
    )
    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    # Extract triplets
    output_triples_set = set()
    for sentence in decoded_preds:
        triples = extract_triplets_typed(sentence)
        for pred_triple in triples:
            if pred_triple["head"].startswith("Lettera di Leopardi") or pred_triple["tail"].startswith("Lettera di "
                                                                                                    "Leopardi"):
                continue
            else:
                triple_string = "<" + pred_triple["head"] + "> <" + pred_triple["type"] + "> <" + pred_triple["tail"] + ">"
                output_triples_set.add(triple_string)
    triples_lst = list(output_triples_set)
    result_entry["raw_text"] = triples_lst
    results.append(result_entry)
    pbar.update(1)

with open("results/results_baseline_test.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)


