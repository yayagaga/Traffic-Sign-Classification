import csv
import time
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

def readTrafficSigns(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    prefix = rootpath +'/' #subdirectory for test
    gtFile = open(prefix + 'test' + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        img=Image.open(prefix + row[0])  # the 1th column is the filename
        # preprocesing image, make sure the images are in the same size
        img=img.crop((int(row[3]),int(row[4]),int(row[5]),int(row[6])))
        img=img.resize((32,32), Image.BICUBIC)
        img=np.array(img)
        images.append(img)
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    return images, labels

# load the images
testImages, testLabels = readTrafficSigns('Test')
# print number of historical images
print('number of historical data=', len(testLabels))

#start = time.time()

X=[]
Y=[]
for i in range(0,len(testLabels)):
    # input X just the flattern image, you can design other features to represent a image
    X.append(testImages[i].flatten())  #from 3D to 1D - vector
    Y.append(int(testLabels[i]))
X=np.array(X)
Y=np.array(Y)

#load the classifiers
clf_rf = pickle.load(open('clf_rf.sav','rb'))
Ypred_rf = clf_rf.predict(X)
print('original class: ',Y)
print('predicted class: ', Ypred_rf)
test_score_rf = accuracy_score(Y, Ypred_rf)
print("Test set score of rf: %0.6f" % test_score_rf)
