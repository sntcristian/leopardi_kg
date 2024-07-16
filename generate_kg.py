
import json
from tqdm import tqdm
import re
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS, OWL


with open("data/data_gpt4.json", "r", encoding="utf-8") as f1:
    data = json.load(f1)
with open("results/results_final.json", "r", encoding="utf-8") as f2:
    results = json.load(f2)
with open("data/entities.json", "r", encoding="utf-8") as f3:
    entities = json.load(f3)
with open("data/properties_final.json", "r", encoding="utf-8") as f4:
    properties = json.load(f4)




G = Graph()
agrelon = Namespace("https://d-nb.info/standards/elementset/agrelon#")
crm = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
leopardi_kg = Namespace("https://sntcristian.github.io/leopardi_kg/")

G.bind("crm", crm)
G.bind("dcterms", DCTERMS)
G.bind("owl", OWL)



pbar = tqdm(total=41)
entities_map = dict()
entities_count = 1
triple_count = 0

for row1, row2 in zip(data, results):
    triples_added = set()
    id_doc = row1["id_doc"]
    title = row1["title"]
    language = row1["lang"]
    orig_date = row1["orig_date"]
    extent = row1["extent"]
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_"+id_doc), RDF.type, crm.E31_Document))
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_"+id_doc), DCTERMS.title, Literal(title,
                                                                                                        lang="en")))
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_"+id_doc), DCTERMS.language, Literal(language,
                                                                                                           lang="en")))
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_"+id_doc), DCTERMS.created, Literal(orig_date,
                                                                                                          lang="it")))
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_"+id_doc), DCTERMS.extent, Literal(extent,
                                                                                                         lang="en")))
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_" + id_doc), crm.P52_has_current_owner,
           URIRef("https://www.lib.cam.ac.uk/")))
    G.add((URIRef("https://sntcristian.github.io/leopardi_kg/document_"+id_doc), crm.P212_has_display_uri,
           URIRef("https://cudl.lib.cam.ac.uk/view/" + id_doc+"/1")))
    gpt_triples = row2["gpt_answer"]
    for triple in gpt_triples:
        triple_str = triple[0]
        triple_value = triple[1]
        if triple_value==1:
            item_lst = re.findall(r'<([^>]*)>', triple_str)
            id_property = properties.get(item_lst[1], None)
            if id_property:
                uri_property = "https://sntcristian.github.io/leopardi_kg/property_"+re.sub("\s", "_", item_lst[1])
                triple_count += 1
                head_id = [key for key, value in entities.items() if item_lst[0] in value]
                if 0 < len(head_id) < 2:
                    wikidata_id = head_id[0]
                else:
                    wikidata_id = None
                if wikidata_id:
                    if wikidata_id in entities_map.keys():
                        head_uri = entities_map[wikidata_id]
                    else:
                        entities_count += 1
                        head_uri = "https://sntcristian.github.io/leopardi_kg/entity_"+str(entities_count)
                        entities_map[wikidata_id]=head_uri
                    G.add((URIRef(head_uri), RDFS.seeAlso, URIRef(wikidata_id)))
                else:
                    entities_count += 1
                    head_uri = "https://sntcristian.github.io/leopardi_kg/entity_" + str(entities_count)
                G.add((URIRef(head_uri), RDF.type, OWL.Thing))
                G.add((URIRef(head_uri), RDFS.label, Literal(item_lst[0], lang="en")))
                tail_id = [key for key, value in entities.items() if item_lst[2] in value]
                if 0 < len(tail_id) < 2:
                    wikidata_id = tail_id[0]
                else:
                    wikidata_id = None
                if wikidata_id:
                    if wikidata_id in entities_map.keys():
                        tail_uri = entities_map[wikidata_id]
                    else:
                        entities_count += 1
                        tail_uri = "https://sntcristian.github.io/leopardi_kg/entity_" + str(entities_count)
                        entities_map[wikidata_id] = tail_uri
                    G.add((URIRef(tail_uri), RDFS.seeAlso, URIRef(wikidata_id)))
                else:
                    entities_count += 1
                    tail_uri = "https://sntcristian.github.io/leopardi_kg/entity_" + str(entities_count)
                G.add((URIRef(tail_uri), RDF.type, OWL.Thing))
                G.add((URIRef(tail_uri), RDFS.label, Literal(item_lst[2], lang="en")))
                triple_uri = "https://sntcristian.github.io/leopardi_kg/statement_"+str(triple_count)
                G.add((URIRef(triple_uri), RDF.type, RDF.Statement))
                G.add((URIRef(triple_uri), RDFS.label, Literal(triple_str, lang="en")))
                G.add((URIRef(triple_uri), RDF.subject, URIRef(head_uri)))
                G.add((URIRef(triple_uri), RDF.predicate, URIRef(uri_property)))
                G.add((URIRef(triple_uri), RDF.object, URIRef(tail_uri)))
                G.add((URIRef(triple_uri), DCTERMS.source, URIRef(
                    "https://sntcristian.github.io/leopardi_kg/document_"+id_doc)))
                G.add((URIRef(uri_property), RDF.type, OWL.ObjectProperty))
                G.add((URIRef(uri_property), RDFS.label, Literal(item_lst[1], lang="en")))
                G.add((URIRef(uri_property), RDFS.seeAlso, URIRef(id_property)))
    pbar.update(1)


G.serialize(destination="results/leopardi_kg_v1.ttl")