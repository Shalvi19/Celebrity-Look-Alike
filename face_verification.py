import face_recognition as freg
import pickle
import numpy as np
import sklearn.svm as svm
from sklearn import neighbors

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
    # print(query_embeddings)
    result = dict()
    result1=[]

    #Using SVM classifier
    for i in query_names:
        result[i] = []
    clf = svm.SVC(gamma='scale')
    clf.fit(celeb_embeddings, celeb_names)

    for i in range(len(query_embeddings)):
        result[query_names[i]].append(clf.predict([query_embeddings[i]])[0])

    print(result)

    #Using KNN classifier
    knn= neighbors.KNeighborsClassifier()
    knn.fit(celeb_embeddings,celeb_paths)
    closest_distances = knn.kneighbors(query_embeddings, n_neighbors=1)

    are_matches = [closest_distances[0][i][0] <= 0.6 for i in range(len(query_embeddings))]
    print("KNN",are_matches)

    #Using Euclidean Distance
    for i in range(len(query_embeddings)):
        result1.append(freg.face_distance(celeb_embeddings, query_embeddings[i]))

    for i in range(len(query_embeddings)):
        print("For ",query_names[i]," Best Matche(s):")
        minval = result1[i].min()
        for j in range(len(result1[i])):
            if minval == result1[i][j]:
                print(celeb_names[j],minval)

main()


