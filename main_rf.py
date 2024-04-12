import matplotlib.pyplot as plt
import csv
import time
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
import pickle

#get data
def readTrafficSigns(rootpath):
    images = [] 
    labels = [] 
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
            # preprocesing image, make sure the images are in the same size
            img=img.crop((int(row[3]),int(row[4]),int(row[5]),int(row[6])))
            img=img.resize((32,32), Image.BICUBIC)
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
    X.append(trainImages[i].flatten())
    Y.append(int(trainLabels[i]))
X=np.array(X)
Y=np.array(Y)

#split the data into test data and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=66)

#train the model
print('Training Starts')
clf_rf=RandomForestClassifier(n_estimators=180, n_jobs=-1, random_state=80)
clf_rf.fit(X_train,Y_train)
print("Training set score: %0.6f" % clf_rf.score(X_train, Y_train))
#predict test data
Y_test_pred=clf_rf.predict(X_test)

#check the accuracy
test_score=accuracy_score(Y_test,Y_test_pred)
print("Test set score: %0.6f" % test_score)

# cross validation
x,y=shuffle(X,Y,random_state=66)
scores = cross_val_score(clf_rf, x, y, cv=3)
print("Mean accuracy of cross validation: %0.6f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Classification report
cm = confusion_matrix(Y_test, Y_test_pred, labels=clf_rf.classes_)
# Confusion Matrix of the classification
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf_rf.classes_)
disp.plot()
plt.title('Confusion Matrix of rf')
plt.show()
print(classification_report(Y_test, Y_test_pred, labels=clf_rf.classes_))
p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
                                 Y_test, Y_test_pred, labels=clf_rf.classes_)

# the labels that are hard to be classified (i.e., with the lowest precision)
print('The label that is hard to classify: {}'.format(np.argmin(p_class)))
duration = time.time() - start
print('Running time: {}s'.format(duration))

# save the model
pickle.dump(clf_rf,open('clf_rf.sav','wb'))

