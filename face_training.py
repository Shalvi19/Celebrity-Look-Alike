import cv2
import dlib
import face_recognition as freg
import os
import numpy as np
import pickle

def main():

    folder_path = 'Query_dataset'    #100 Celeb dataset. 15 images per celeb

    names = os.listdir(folder_path)
    total = 0

    knownEmbeddings = []
    knownNames = []
    knownPath = []

    detector = dlib.get_frontal_face_detector()  #Dlib Face detector

    for name in names:    #iterating over the folders in dataset - folders named with celeb names
        print(name)
        img_list = os.listdir(folder_path+'/'+name)

        print((img_list))

        cnt = 0
        for img_path in img_list:

            pixels = cv2.imread(folder_path+'/'+name+'/'+img_path)
            if pixels is None:
               continue
            if pixels.shape[0]<640 and pixels.shape[1]<480:    #discard face if its size is less than a particular value
                continue

            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            boxes = freg.face_locations(rgb)

            if len(boxes)==1:
                cnt += 1
                encodings = freg.face_encodings(rgb, boxes,num_jitters=10,model='large')

                knownEmbeddings.append(encodings[0])
                knownNames.append(name)
                knownPath.append(folder_path+'/'+name+'/'+img_path)
            if cnt == 15:
                break
        print(name,cnt)
    data = {"embeddings": knownEmbeddings, "names": knownNames,"paths":knownPath}

    #write the dictionary in the pickle file
    f = open('Query_embedding.pickle', "wb")
    f.write(pickle.dumps(data))
    f.close()


main()