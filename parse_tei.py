from lxml import etree
import glob

ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

for tei_doc in glob.glob("xml_tei/*.txt"):
    xml_file_path = tei_doc
    tree = etree.parse(xml_file_path)
    root = tree.getroot()

    # Use the namespace in the XPath query
    ms_item = root.find('.//tei:msItem/tei:title', namespaces=ns)
    
    # Check if the title element exists before accessing its text
    if ms_item is not None:
        ms_item_title = ms_item.text
        print(ms_item_title)
    else:
        print("No title found in this msItem")