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
        if (counter > 500000):
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
pn = PromptNode("gpt-3.5-turbo", api_key="sk-123",model_kwargs={"stream":False})
pipeline = DocumentSearchPipeline(retriever)

while True:
    ## input user query
    query = "gravity"
    query = input("Enter query:")
    raw_query = query.strip()
    query = query.strip()
    if query == "exit":
        break

    ## ASK llama to generate 2 different queries to optimize for recall
    prompt = 'rewrite the QUERY: "' + query + '" in EXACTLY 3 different ways to optimize for the recall WITHOUT STOP WORDS. Output a SINGLE JSONARRAY of STRINGs with the KEY QUERY'


    response = pn.prompt(prompt)
    response_json = json.loads(response[0])
    Q1 = response_json['QUERY'][0]
    Q2 = response_json['QUERY'][1]
    Q3 = response_json['QUERY'][2]

    KP = []
    ## Ask llama to generate 2 key phrases for the query
    if (len(query.split()) >= 6):
        key_phrases = 'EXTRACT the 2 key phrases of the QUERY: "' + query + '" Output a SINGLE STRING JSONARRAY of the keyphrases with the KEY KP'
        response = pn.prompt(key_phrases)
        response_json = json.loads(response[0])
        #print(response)
        try:
            KP = response_json['KP']
        except:
            KP = []

   

    queries = [raw_query,Q1,Q2,Q3]
    if (len(KP) > 0):
        queries.append(KP[0])
        queries.append(KP[1])

    print ("Number of backend queries:" + str(len(queries)))
    print(queries)
    documents_to_evaluate = []
    titles = set()
    round  = 0
    for query in queries:
        results = pipeline.run(query, params={"Retriever": {"top_k":5}})
        print ("Round : " + str(round))
        round += 1
        print("Backend query:" + query)
        
        for document in results["documents"]:
            if document.meta["id"] not in titles:
                documents_to_evaluate.append(document)
                titles.add(document.meta["id"])
        

   
    w = open("prompt_text","w")
    text = " Take a deep breadth and do the following task. You are a Search Engine Ranker. You have QUERY and SEARCH RESULTS. Rank all the RESULT IDS in SEARCH RESULTS based on CONTENT's relevance to the QUERY. QUERY: " + raw_query.strip() + "\n"
    text += "SEARCH RESULTS: \n"
    counter = 0


    #####
    print ("Number of documents to evaluate:" + str(len(documents_to_evaluate)))
    for result in documents_to_evaluate:
        text += "RESULT ID :" + result.meta["id"] + "\n"
        w.write("RESULT ID :" + result.meta["id"] + "\n")

        text += "CONTENT:" + result.content.strip() + "\n"
        w.write("CONTENT:" + result.content + "\n")
        w.write("---------------------------\n")
   
    text += "Based on all the results above, generate a DETAILED SUMMARY providing multiple POINTS OF VIEW summarizing of the results and provide a ANSWER and TAKEAWAY for the QUERY: " + raw_query.strip()
    #w.write("Based on all the results above, generate a PARAGRAPH OF SUMMARY summarizing of the results for the QUERY: " + raw_query)


    #text += " Rank all the RESULT IDs based on the relevance for the QUERY " + raw_query + " AND CONTENT. Output the SORTED ORDER OF the RESULTS in a SINGLE JSONARRAY of RANKED RESULT IDS based on relevance to the QUERY with the KEY ORDERED"
    #w.write(" Rank all the RESULT ID based on the relevance for the QUERY and output the SORTED ORDER OF the RESULTS in a SINGLE JSONARRAY of IDS with the KEY RESULTS")
    w.close()
    
    print("wrote the prompt to the file: " + raw_query)

    try:
        pn = PromptNode("gpt-3.5-turbo", api_key="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",model_kwargs={"stream":False})

        output= pn.prompt(text)
        print("Model Response:", output)  # Print the model's response

        print(output[0])
    except Exception as e:
        print(e)
        print("error")
