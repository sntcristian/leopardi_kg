from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

# Define namespaces
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")


with open("queries/get_entities_sorted.txt", "r", encoding="utf-8") as f1:
    get_entities_q = f1.read()

with open("queries/get_relations_sorted.txt", "r", encoding="utf-8") as f2:
    get_relations_q = f2.read()
# Initialize the graph
g = Graph()

# Parse the RDF data (replace 'your_data_file.rdf' with your actual data file)
g.parse("results/leopardi_kg_v1.ttl", format="ttl")

query1 = prepareQuery(get_entities_q)

query2 = prepareQuery(get_relations_q)




results1 = g.query(query1)
# Print results of the first query
print(f"{'Entity':<70} {'Label':<70} {'Q_ID':<20} {'Count':<10}")
print("="*170)
q_ids = set()
for row in results1:
    entity = row.entity
    label = row.label if row.label else "No label"
    q_id = row.Q_ID if row.Q_ID else "No Q_ID"
    count = row.statementCount
    if q_id not in q_ids or q_id=="No Q_ID":
        print(f"{str(entity):<70} {label:<70} {q_id:<20} {count:<10}")
        q_ids.add(q_id)

print("\n\n")

# Execute the second query
results2 = g.query(query2)
# Print results of the second query
print(f"{'Property':<70} {'Label':<70} {'Q_ID':<20} {'Count':<10}")
print("="*170)
for row in results2:
    property_ = row.property
    label = row.label if row.label else "No label"
    q_id = row.Q_ID if row.Q_ID else "No Q_ID"
    count = row.statementCount
    print(f"{str(property_):<70} {label:<70} {q_id:<20} {count:<10}")
