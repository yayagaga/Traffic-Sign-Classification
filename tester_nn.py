import csv
import time
from PIL import Image
import numpy as np
from sklearn import decomposition
from sklearn.metrics import accuracy_score
import pickle

def readTrafficSigns(rootpath):
    '''Reads traffic sign data
    Arguments: path to the traffic sign data, for example './TrafficSignData/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over N classes, at most we have 42 classes
    N=15
    for c in range(0,N):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        #gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            img=Image.open(prefix + row[0])  # the 1th column is the filename
            img=img.crop((int(row[3]),int(row[4]),int(row[5]),int(row[6])))  # cut the image by the region of interest
            img=img.resize((32,32), Image.BICUBIC)  # preprocessing image, make sure the images are in the same size
            img=np.array(img)
            images.append(img)
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

# Read test data
def read_TestTrafficSigns(rootpath):
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


start = time.time()
# load the images
trainImages, trainLabels = readTrafficSigns('TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(trainLabels))
# load the test set images
testImages, testLabels = read_TestTrafficSigns('Test')
print('number of test data=', len(testLabels))

# To get the required dimension after PCA, we fuse the test into the historical data first
# design the input and output for model
X=[]
Y=[]
for i in range(0,len(trainLabels)):
    # input X just the flatten image, you can design other features to represent a image
    X.append(trainImages[i].flatten())  #from 3D to 1D - vector
    Y.append(int(trainLabels[i]))
for j in range(0,len(testLabels)):
    X.append(testImages[j].flatten())  # from 3D to 1D - vector
    Y.append(int(testLabels[j]))
X=np.array(X)
Y=np.array(Y)

# data preprocessing
X = X/255
pca = decomposition.PCA(n_components= 0.98)
newX = pca.fit_transform(X)

clf_nn = pickle.load(open('clf_nn.sav','rb'))
Ypred_nn = clf_nn.predict(newX[len(trainLabels)+1:])
print('original class: ',Y[len(trainLabels)+1:])
print('predicted class: ',Ypred_nn)
test_score_nn = accuracy_score(Y[len(trainLabels)+1:], Ypred_nn)
print("Test set score of nn: %0.6f" % test_score_nn)

