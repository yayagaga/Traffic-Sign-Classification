import csv
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
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

start = time.time()
# load the images
trainImages, trainLabels = readTrafficSigns('TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(trainLabels))

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
X = X/255
pca = decomposition.PCA(n_components= 0.98)
newX = pca.fit_transform(X)
# split the data into training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(newX, Y, test_size=0.3, random_state=66)

# train a Neural network model
print('Training Starts')
clf_nn = MLPClassifier(hidden_layer_sizes = (100, 50), solver = 'adam', max_iter = 30, learning_rate = 'adaptive')
clf_nn.fit(X_train, Y_train)
print("Training set score: %0.6f" % clf_nn.score(X_train, Y_train))
Y_test_pred = clf_nn.predict(X_test)
# test set score
test_score = accuracy_score(Y_test, Y_test_pred)
print("Test set score: %0.6f" % test_score)

# cross validation
x,y=shuffle(newX,Y,random_state=66)
scores = cross_val_score(clf_nn, x, y, cv=5)
print("Mean accuracy of cross validation: %0.6f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Classification report
cm = confusion_matrix(Y_test, Y_test_pred, labels=clf_nn.classes_)
# Confusion Matrix of the classification
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf_nn.classes_)
disp.plot()
plt.title('Confusion Matrix of nn')
plt.show()
print(classification_report(Y_test, Y_test_pred, labels=clf_nn.classes_))
p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
                                 Y_test, Y_test_pred, labels=clf_nn.classes_)
# the labels that are hard to be classified (i.e., with the lowest precision)
print('The label that is hard to classify: {}'.format(np.argmin(p_class)))
duracy = time.time() - start
print('Running time: {}s'.format(duracy))

# save the model
pickle.dump(clf_nn,open('clf_nn.sav','wb'))
