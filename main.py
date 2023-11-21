from lxml import etree
import glob

def parse_tei(xml_file_path):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.parse(xml_file_path)
    root = tree.getroot()

    # Use the namespace in the XPath query
    ms_item_1 = root.find('.//tei:msItem/tei:title', namespaces=ns)
    
    # Check if the title element exists before accessing its text
    if ms_item_1 is not None:
        ms_item_title = ms_item_1.text

    ms_item_2 = root.find('.//tei:msItem/tei:textLang', namespaces=ns)
    if ms_item_2 is not None:
        ms_item_lang = ms_item_2.text

    support_desc_1 = root.find('.//tei:supportDesc/tei:support', namespaces=ns)
    if support_desc_1 is not None:
        support_desc_support = support_desc_1.text
    
    support_desc_2 = root.find('.//tei:supportDesc/tei:extent', namespaces=ns)
    if support_desc_2 is not None:
        support_desc_extent = support_desc_2.text
    
    orig_date = root.find('.//tei:origin/tei:origDate', namespaces=ns).text
    orig_place = root.find('.//tei:origin/tei:origPlace', namespaces=ns)
    orig_place_key = orig_place.get("key")
    orig_place_text = orig_place.text
    persons = root.findall('.//tei:listPerson/tei:person', namespaces=ns)
    pers_list = []
    for person in persons:
        ref = person.get("{http://www.w3.org/XML/1998/namespace}id")
        key = person.find("tei:persName", namespaces=ns).get("key")
        forename = person.find("tei:persName/tei:forename", namespaces=ns).text
        surname = person.find("tei:persName/tei:surname", namespaces=ns).text
        pers_list.append({"key":key, "ref":ref, "forename":forename, "surname":surname, "persName":forename+" "+surname})
    places = root.findall('.//tei:listPlace/tei:place', namespaces=ns)
    place_list = []
    for place in places:
        ref = place.get("{http://www.w3.org/XML/1998/namespace}id")
        key = place.find("tei:placeName", namespaces=ns).get("key")
        name = person.find("tei:placeName", namespaces=ns).text
        place_list.append({"key":key, "ref":ref, "placeName":name})





for tei_doc in glob.glob("xml_tei/*.txt"):
    xml_file_path = tei_doc
    parse_tei(xml_file_path)

    