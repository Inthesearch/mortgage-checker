import math
from src.embed import embed_text, load_index
import numpy as np

def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += (a[i] * b[i])
    return result

def length(a):
    result=0
    for i in range(len(a)):
        result+=(a[i]**2)
    result=math.sqrt(result)
    return result

def search_rules(query, k):
    query_embedding=embed_text(query)
    q=np.array(query_embedding)
    json_array=load_index("data/rules_index.json")
    embedding_arr=[json_arr["embedding"] for json_arr in json_array]
    m=np.array(embedding_arr)
    scores=(m @ q)/ (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    topk=np.argsort(scores)[::-1][:k]
    return best_chunks_info(topk, json_array, scores)

def best_chunks_info(topk, json_array, scores):
    chunks=[]
    for a in topk:
        d={}
        d["source"]=json_array[a]["source"]
        d["section_id"]=json_array[a]["section_id"]
        d["text"]=json_array[a]["text"]
        d["score"]=float(scores[a])
        chunks.append(d)
    return chunks


def cosine_similarity(a, b):
    return dot_product(a, b)/(length(a)* length(b))


if "__main__" == __name__:
    print(search_rules("How much down payment is required if the house price is 435,000 Canadian dollars?", 3))
