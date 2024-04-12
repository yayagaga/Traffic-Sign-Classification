import matplotlib.pyplot as plt
import csv
import time
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


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
trainImages, trainLabels = readTrafficSigns('TrafficSignData/TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(trainLabels))
# show one sample image
plt.imshow(trainImages[44])
plt.show()
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

#shuffle the X Y
X,Y=shuffle(X,Y,random_state=66)

#parameter test with train data
train_scores = np.zeros((10,10))
randomForest = RandomForestClassifier(n_estimators=1, n_jobs=-1, random_state=0)

for i in range(10):
    randomForest.set_params(n_estimators=((i+1)*20))
    for j in range (10):
        randomForest.set_params(random_state=((j+1)*10))
        rf=randomForest.fit(X, Y)
        train_scores[i][j]=np.mean(cross_val_score(rf, X, Y, cv=3))
        #test_scores.append(randomForest.score(x, y))
print(train_scores.max())
print(np.where(train_scores == np.max(train_scores)))
print("Finish")

# compute the running time
duracy = time.time() - start
print('Running time: {}s'.format(duracy))