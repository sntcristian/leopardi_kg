
import json
from tqdm import tqdm
import re
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS, OWL


with open("data/data_gpt4.json", "r", encoding="utf-8") as f1:
    data = json.load(f1)
with open("results/results_gpt4.json", "r", encoding="utf-8") as f2:
    results = json.load(f2)
with open("data/entities.json", "r", encoding="utf-8") as f3:
    entities = json.load(f3)
with open("data/properties_gpt4.json", "r", encoding="utf-8") as f4:
    properties = json.load(f4)


## Da fare:
## Aggiunti evento :LetterCorrespondence
## :LetterCorrespondence a crm:E5_Event
##     :P7_took_place_at :Orig_Place_1
##     :P4_has_time_span :E52_Time-Span
##     :P11_had_participant :E21_Person
##     :P94_has_created :E31_Document


triple_id = 0
G = Graph()
agrelon = Namespace("https://d-nb.info/standards/elementset/agrelon#")
crm = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
frbroo = Namespace("http://iflastandards.info/ns/fr/frbr/frbroo/")

G.bind("agrelon", agrelon)
G.bind("crm", crm)
G.bind("frbroo", frbroo)
G.bind("dcterms", DCTERMS)
G.bind("owl", OWL)


pbar = tqdm(total=41)
for row1, row2 in zip(data, results):
    triples_added = set()
    id_doc = row1["id_doc"]
    title = row1["title"]
    language = row1["lang"]
    orig_date = row1["orig_date"]
    extent = row1["extent"]
    G.add((URIRef("http://example.org/"+id_doc), RDF.type, crm.E31_Document))
    G.add((URIRef("http://example.org/"+id_doc), DCTERMS.title, Literal(title, lang="en")))
    G.add((URIRef("http://example.org/" + id_doc), DCTERMS.language, Literal(language, lang="en")))
    G.add((URIRef("http://example.org/" + id_doc), DCTERMS.created, Literal(orig_date, lang="it")))
    G.add((URIRef("http://example.org/" + id_doc), DCTERMS.extent, Literal(extent, lang="it")))
    G.add((URIRef("http://example.org/" + id_doc), crm.P212_has_display_uri,
           URIRef("https://cudl.lib.cam.ac.uk/view/" + id_doc+"/1")))
    gpt_triples = row2["gpt_answer"]
    for triple in gpt_triples:
        print(triple)
        item_lst = re.findall(r'<([^>]*)>', triple)
        print(item_lst)
        id_property = properties.get(item_lst[1], None)
        if id_property:
            triple_id += 1
            triple_uri = URIRef("http://example.org/statement_" + str(triple_id))
            uri_property = URIRef("http://example.org/" + re.sub("\W", "_", item_lst[1]))
            head_uri = URIRef("http://example.org/" + re.sub("\W", "_", item_lst[0]))
            tail_uri = URIRef("http://example.org/" + re.sub("\W", "_", item_lst[2]))
            if (head_uri, uri_property, tail_uri) in triples_added:
                continue
            elif head_uri == tail_uri:
                continue
            else:
                G.add((triple_uri, RDF.type, RDF.Statement))
                G.add((triple_uri, RDF.subject, head_uri))
                G.add((triple_uri, RDF.predicate, uri_property))
                G.add((triple_uri, RDF.object, tail_uri))
                G.add((triple_uri, DCTERMS.source, URIRef("http://example.org/" + id_doc)))
                G.add((head_uri, RDF.type, OWL.Thing))
                G.add((head_uri, RDFS.label, Literal(item_lst[0], lang="en")))
                G.add((uri_property, RDF.type, OWL.ObjectProperty))
                G.add((uri_property, RDFS.label, Literal(item_lst[1], lang="en")))
                G.add((uri_property, RDFS.seeAlso, URIRef(id_property)))
                G.add((tail_uri, RDF.type, OWL.Thing))
                G.add((tail_uri, RDFS.label, Literal(item_lst[2], lang="en")))
                head_id = [key for key, value in entities.items() if value == item_lst[0]]
                if 0 < len(head_id) < 2:
                    head_id = URIRef(head_id[0])
                    G.add((head_uri, RDFS.seeAlso, head_id))
                tail_id = [key for key, value in entities.items() if value == item_lst[2]]
                if 0 < len(tail_id) < 2:
                    tail_id = URIRef(tail_id[0])
                    G.add((tail_uri, RDFS.seeAlso, tail_id))
                triples_added.add((head_uri, uri_property, tail_uri))
    pbar.update(1)




#     text = ". ".join(gpt_triples)
#     model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
#     generated_tokens = model.generate(
#         model_inputs["input_ids"].to(model.device),
#         attention_mask=model_inputs["attention_mask"].to(model.device),
#         decoder_start_token_id=tokenizer.convert_tokens_to_ids("tp_XX"),
#         **gen_kwargs,
#     )
#     # Extract text
#     decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
#     for sentence in decoded_preds:
#         triples = extract_triplets_typed(sentence)
#         for pred_triple in triples:
#             id_property = properties.get(pred_triple["type"], None)
#             if id_property:
#                 triple_id += 1
#                 triple_uri = URIRef("http://example.org/statement_" + str(triple_id))
#                 uri_property = URIRef("http://example.org/"+re.sub("\W","_", pred_triple["type"]))
#                 head_uri = URIRef("http://example.org/"+re.sub("\W","_", pred_triple["head"]))
#                 tail_uri = URIRef("http://example.org/"+re.sub("\W","_", pred_triple["tail"]))
#                 if (head_uri, uri_property, tail_uri) in triples_added:
#                     continue
#                 elif head_uri==tail_uri:
#                     continue
#                 else:
#                     G.add((triple_uri, RDF.type, RDF.Statement))
#                     G.add((triple_uri, RDF.subject, head_uri))
#                     G.add((triple_uri, RDF.predicate, uri_property))
#                     G.add((triple_uri, RDF.object, tail_uri))
#                     G.add((URIRef("http://example.org/" + id_doc), ns1.P148, triple_uri))
#                     G.add((triple_uri, DCTERMS.source, URIRef("http://example.org/"+id_doc)))
#                     G.add((head_uri, RDF.type, OWL.Thing))
#                     G.add((head_uri, RDFS.label, Literal(pred_triple["head"], lang="en")))
#                     G.add((uri_property, RDF.type, OWL.ObjectProperty))
#                     G.add((uri_property, RDFS.label, Literal(pred_triple["type"], lang="en")))
#                     G.add((uri_property, RDFS.seeAlso, URIRef(id_property)))
#                     G.add((tail_uri, RDF.type, OWL.Thing))
#                     G.add((tail_uri, RDFS.label, Literal(pred_triple["tail"], lang="en")))
#                     head_id = [key for key, value in entities.items() if value == pred_triple["head"]]
#                     if 0 < len(head_id) < 2:
#                         head_id=URIRef(head_id[0])
#                         G.add((head_uri, RDFS.seeAlso, head_id))
#                     tail_id = [key for key, value in entities.items() if value == pred_triple["tail"]]
#                     if 0 < len(tail_id) < 2:
#                         tail_id = URIRef(tail_id[0])
#                         G.add((tail_uri, RDFS.seeAlso, tail_id))
#                     triples_added.add((head_uri, uri_property, tail_uri))
#     pbar.update(1)
#
# G.bind('crm', crm)
G.serialize(destination="results/leopardi_cudl_gpt4.ttl")