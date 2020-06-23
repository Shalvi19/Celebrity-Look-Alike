import face_recognition as freg
import pickle
import numpy as np

def main():
    file = open("celeb_embedding.pickle","rb")
    celeb_data = pickle.load(file)
    file2 = open("Query_embedding.pickle","rb")
    query_data = pickle.load(file2)
    celeb_embeddings = celeb_data['embeddings']
    celeb_names = celeb_data['names']
    celeb_paths = celeb_data['paths']
    query_embeddings = query_data['embeddings']
    query_names = query_data['names']
    query_paths = query_data['paths']
    print("len of embedding",len(celeb_embeddings))
    result = []
    for i in range(len(query_embeddings)):
        result.append(freg.face_distance(celeb_embeddings, query_embeddings[i]))

    for i in range(len(query_embeddings)):
        print("For ",query_names[i]," Best Matche(s):")
        minval = result[i].min()
        for j in range(len(result[i])):
            if minval == result[i][j]:
                print(celeb_names[j],minval)


        # print(celeb_names[j],result[i][j])


main()