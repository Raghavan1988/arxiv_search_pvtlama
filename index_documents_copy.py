from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack import Document
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import TextConverter 

import json
# Initialize an in-memory document store
document_store = InMemoryDocumentStore(use_bm25=True)
data = []
# Read JSON file
counter = 0
with open("/home/raghavan/arxiv-metadata-oai-snapshot.json", "r") as f:
    for line in f:
        data.append(json.loads(line))
        counter += 1
        ### index only 1000000 documents
        if (counter > 200000):
            break
        
counter = 0
# Convert JSON data to Haystack Document objects
documents = []
for item in data:
    # Assuming each item in the JSON has a 'text' and 'meta' field
    #print(item)
    #input()
    try:
        text = item.get("abstract", "")
        title = item.get("title", "")
        id = item.get("id", "")
        meta = {"name": title, "id": id}
        document_dict = {}
        document_dict["meta"] = meta
        document_dict["content"] = title + " " + text
        document_dict["id"] = id
        document_dict["score"] = 0.8
        doc = Document.from_dict(document_dict)
        documents.append(doc)
        
    except Exception as e:
        print (e)
        print("error:" + str(counter))


# Index documents into the document store
document_store.write_documents(documents)

# Initialize a retriever (BM25)
print ("number of documents" + str(len(documents)))


retriever = BM25Retriever(document_store=document_store)

from haystack.nodes.prompt import PromptNode
pn = PromptNode("gpt-3.5-turbo", api_key="sk1",model_kwargs={"stream":False})
pipeline = DocumentSearchPipeline(retriever)

while True:
    ## input user query
    query = "gravity"
    query = input("Enter query:")
    raw_query = query
    if query == "exit":
        #print(results["documents"])

        text += "RESULT ID :" + result.meta["id"] + "\n"
        w.write("RESULT ID :" + result.meta["id"] + "\n")

        text += result.content + "\n"
        w.write(result.content + "\n")
        w.write("---------------------------\n")
   
    text += "Based on all the results above, generate a PARAGRAPH OF SUMMARY summarizing of the results for the QUERY: " + raw_query
    w.write("Based on all the results above, generate a PARAGRAPH OF SUMMARY summarizing of the results for the QUERY: " + raw_query)
    w.close()
    #text += " Rank all the RESULT ID based on the relevance for the QUERY and output the SORTED ORDER OF the RESULTS in a SINGLE JSONARRAY of IDS with the KEY RESULTS\n"
    #w.write(" Rank all the RESULT ID based on the relevance for the QUERY and output the SORTED ORDER OF the RESULTS in a SINGLE JSONARRAY of IDS with the KEY RESULTS\n")
    print("wrote the prompt to the file: " + raw_query)

    


    try:
        output= pn.prompt(text)
        print(output[0])
    except Exception as e:
        print(e)
        print("error")
