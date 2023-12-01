from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import re
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS, OWL
import urllib.parse


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


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


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", tgt_lang="tp_XX")
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


with open("data/data.json", "r", encoding="utf-8") as f1:
    data = json.load(f1)
with open("data/entities.json", "r", encoding="utf-8") as f2:
    entities = json.load(f2)
with open("data/properties.json", "r", encoding="utf-8") as f3:
    properties = json.load(f3)

triple_id = 0
G = Graph()
ns1 = Namespace('http://www.cidoc-crm.org/cidoc-crm/')

pbar = tqdm(total=41)
for row in data:
    triples_added = set()
    id_doc = row["id_doc"]
    title = row["title"]
    G.add((URIRef("http://example.org/"+id_doc), RDF.type, ns1.E73))
    G.add((URIRef("http://example.org/"+id_doc), RDFS.label, Literal(title, lang="en")))
    G.add((URIRef("http://example.org/" + id_doc), ns1.P212, URIRef("https://cudl.lib.cam.ac.uk/view/" + id_doc+"/1")))
    gpt_triples = row["chat-gpt"]
    gpt_triples = [" ".join(triple) for triple in gpt_triples]
    text = ". ".join(gpt_triples)
    model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
    generated_tokens = model.generate(
        model_inputs["input_ids"].to(model.device),
        attention_mask=model_inputs["attention_mask"].to(model.device),
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
        **gen_kwargs,
    )
    # Extract text
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    for sentence in decoded_preds:
        triples = extract_triplets_typed(sentence)
        for pred_triple in triples:
            id_property = properties.get(pred_triple["type"], None)
            if id_property:
                triple_id += 1
                triple_uri = URIRef("http://example.org/statement_" + str(triple_id))
                uri_property = URIRef("http://example.org/"+re.sub("\W","_", pred_triple["type"]))
                head_uri = URIRef("http://example.org/"+re.sub("\W","_", pred_triple["head"]))
                tail_uri = URIRef("http://example.org/"+re.sub("\W","_", pred_triple["tail"]))
                if (head_uri, uri_property, tail_uri) in triples_added:
                    continue
                else:
                    G.add((triple_uri, RDF.type, RDF.Statement))
                    G.add((triple_uri, RDF.subject, head_uri))
                    G.add((triple_uri, RDF.predicate, uri_property))
                    G.add((triple_uri, RDF.object, tail_uri))
                    G.add((URIRef("http://example.org/" + id_doc), ns1.P148, triple_uri))
                    G.add((triple_uri, DCTERMS.source, URIRef("http://example.org/"+id_doc)))
                    G.add((head_uri, RDF.type, OWL.Thing))
                    G.add((head_uri, RDFS.label, Literal(pred_triple["head"], lang="en")))
                    G.add((uri_property, RDF.type, OWL.ObjectProperty))
                    G.add((uri_property, RDFS.label, Literal(pred_triple["type"], lang="en")))
                    G.add((uri_property, RDFS.seeAlso, URIRef(id_property)))
                    G.add((tail_uri, RDF.type, OWL.Thing))
                    G.add((tail_uri, RDFS.label, Literal(pred_triple["tail"], lang="en")))
                    head_id = [key for key, value in entities.items() if value == pred_triple["head"]]
                    if 0 < len(head_id) < 2:
                        head_id=URIRef(head_id[0])
                        G.add((head_uri, RDFS.seeAlso, head_id))
                    tail_id = [key for key, value in entities.items() if value == pred_triple["tail"]]
                    if 0 < len(tail_id) < 2:
                        tail_id = URIRef(tail_id[0])
                        G.add((tail_uri, RDFS.seeAlso, tail_id))
                    triples_added.add((head_uri, uri_property, tail_uri))
    pbar.update(1)

G.bind('cidoc', ns1)
G.serialize(destination="results/leopardi_cudl.ttl")