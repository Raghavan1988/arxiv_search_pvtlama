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
        if (counter > 200000):
            break
        
print(counter)

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
#print(documents[:-5])
#print(doc)
# Initialize a retriever (BM25)
print ("number of documents" + str(len(documents)))


retriever = BM25Retriever(document_store=document_store)

from haystack.nodes.prompt import PromptNode
pn = PromptNode("gpt-3.5-turbo", api_key="123",model_kwargs={"stream":False})


pipeline = DocumentSearchPipeline(retriever)

while True:

    query = "gravity"
    query = input("Enter query:")
    raw_query = query
    if query == "exit":
        break

    prompt = 'rewrite the QUERY: "' + query + '" in EXACTLY 2 different ways to optimize for the recall. Output a SINGLE JSONARRAY of STRINGs with the KEY QUERY'


    response = pn.prompt(prompt)
    #print(response)
    response_json = json.loads(response[0])
    Q1 = response_json['QUERY'][0]
    Q2 = response_json['QUERY'][1]

    KP = []
    if (len(query.split()) >= 6):
        key_phrases = 'EXTRACT the 2 key phrases of the QUERY: "' + query + '" Output a SINGLE STRING JSONARRAY of the keyphrases with the KEY KP'
        response = pn.prompt(key_phrases)
        response_json = json.loads(response[0])
        #print(response)
        try:
            KP = response_json['KP']
        except:
            KP = []

   

    queries = [raw_query,Q1,Q2]
    if (len(KP) > 0):
        queries.append(KP[0])
        queries.append(KP[1])

    print ("Number of backend queries:" + str(len(queries)))
    print(queries)

    documents_to_evaluate = []
    titles = set()
    for query in queries:
        results = pipeline.run(query, params={"Retriever": {"top_k": 3}})
        print("backend query:" + query)
        #print(results["documents"])
        
        for document in results["documents"]:
            if document.meta["id"] not in titles:
                documents_to_evaluate.append(document)
                titles.add(document.meta["id"])
            else:
                print("duplicate") 
                #print(document.meta["id"])
                #print(document.meta["name"])
   
    w = open("prompt_text","w")
    text = ""
    counter = 0


    #####
    for result in documents_to_evaluate:
        text += "RESULT ID :" + result.meta["id"] + "\n"
        w.write("RESULT ID :" + result.meta["id"] + "\n")

        text += result.content + "\n"
        w.write(result.content + "\n")
        w.write("---------------------------\n")
   
    text += "Based on all the results above, generate a summary of the results using the QUERY: " + raw_query
    w.write("Based on all the results above, generate a summary of the results using the QUERY: " + raw_query)
    #text += " Rank all the RESULT ID based on the relevance for the QUERY and output the SORTED ORDER OF the RESULTS in a SINGLE JSONARRAY of IDS with the KEY RESULTS\n"
    #w.write(" Rank all the RESULT ID based on the relevance for the QUERY and output the SORTED ORDER OF the RESULTS in a SINGLE JSONARRAY of IDS with the KEY RESULTS\n")
    w.close()
    print("wrote to file" + raw_query)

    try:
        pn = PromptNode("gpt-3.5-turbo", api_key="123",model_kwargs={"stream":False})

        output= pn.prompt(text)
        print(output)
    except Exception as e:
        print(e)
        print("error")
