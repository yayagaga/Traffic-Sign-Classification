import csv
import time
from PIL import Image
import numpy as np
from sklearn import decomposition
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle


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

# load the images
trainImages, trainLabels = readTrafficSigns('TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(trainLabels))
start = time.time()
# design the input and output for model
X=[]
Y=[]
for i in range(0,len(trainLabels)):
    # input X just the flattern image, you can design other features to represent a image
    X.append(trainImages[i].flatten())  #from 3D to 1D - vector
    Y.append(int(trainLabels[i]))
X=np.array(X)
Y=np.array(Y)
# data preprocessing
X,Y=shuffle(X,Y,random_state=66)
pca = decomposition.PCA(n_components= 0.98)
newX = pca.fit_transform(X)

# train a SVM
# set the parameters of svm model by GridSearchCV
param_grid = {'C':[0.1, 1, 10, 30, 50, 70, 100],
              'gamma':[0.0001, 0.001, 'scale'],
              'kernel': ['rbf'],
              'decision_function_shape':['ovr']}
model = svm.SVC()
clf_svm = GridSearchCV(model, param_grid)
clf_svm.fit(newX, Y)
best_parameters = clf_svm.best_estimator_.get_params()
best_score = clf_svm.best_score_
print(best_parameters)
print(best_score)
print("the best parameters among the grid is {},".format(best_parameters), end = ' ')
print("and the best score is %0.6f" % best_score)

# compute the running time
duracy = time.time() - start
print('Running time: {}s'.format(duracy))


