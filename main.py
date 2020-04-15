from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from cv2 import CascadeClassifier, cvtColor, COLOR_BGR2GRAY
from sklearn.neighbors import KNeighborsClassifier
from cv2 import imread, CASCADE_SCALE_IMAGE
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import os.path as __path__
import logging
import numpy
import time
import cv2
import os


def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')


def faceDetection(imagePath):
    faceCascade = CascadeClassifier(
        __path__.join(__path__.curdir, 'haarcascade_frontalface_default.xml')
    )
    img = imread(imagePath, -1)
    gray = cvtColor(img, COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None
    elif len(faces) == 1:
        return cv2.equalizeHist(gray)


def readDataset(datasetName=None):
    logging.debug('Loading dataset...')
    images, labels, label, dataset = [], [], None, dict()
    root = __path__.join(__path__.curdir, datasetName)
    if datasetName == 'tarrlab':
        for race in os.listdir(root):
            if str(race) == 'africanamerican':
                label = 'non asian'
            elif str(race) == 'asian':
                label = 'asian'
            elif str(race) == 'caucasian':
                label = 'non asian'
            elif str(race) == 'hispanic':
                label = 'non asian'
            race = __path__.join(root, race)
            if __path__.isdir(race):
                for target in os.listdir(race):
                    filenameTemp = target.split('_')
                    if filenameTemp[1] == '1100':
                        image = faceDetection(__path__.join(race, target))
                        if image is not None:
                            image = image.reshape([1, 250 * 250])
                            images.append(image)
                            labels.append(label)
        images = numpy.reshape(images, (len(images), 250*250))
    elif datasetName == 'fairface':
        i = 0
        root = __path__.join(__path__.curdir, datasetName)
        for row in open(__path__.join(root, 'fairface.csv'), mode='r'):
            if i == 0:
                i += 1
                continue
            row = row.strip().split(',')
            if row[4] == 'True':
                path = row[0].split('/')
                path = __path__.join(path[0], path[1])
                path = __path__.join(root, path)
                image = faceDetection(path)
                if image is not None:
                    image = image.reshape([1, 224 * 224])
                    images.append(image)
                    if row[3] == 'Black':
                        labels.append('non asian')
                    elif row[3] == 'East Asian':
                        labels.append('asian')
                    elif row[3] == 'Indian':
                        labels.append('non asian')
                    elif row[3] == 'Latino_Hispanic':
                        labels.append('non asian')
                    elif row[3] == 'Middle Eastern':
                        labels.append('non asian')
                    elif row[3] == 'Southeast Asian':
                        labels.append('asian')
                    elif row[3] == 'White':
                        labels.append('non asian')
                    i += 1
        images = numpy.reshape(images, (len(images), 224*224))

    dataset['images'] = numpy.array(images, dtype=numpy.uint8)
    dataset['labels'] = numpy.array(labels)
    logging.info('Loading process finished.')

    logging.debug('Saving dataset...')
    numpy.save('dataset', dataset)
    logging.debug('Saving process finished.')

    return dataset


def loadDataset(limit):
    logging.debug('Loading dataset...')
    data = dict()
    dataset = numpy.load('dataset.npy', allow_pickle=True)
    data['images'] = dataset[()]['images']
    data['labels'] = dataset[()]['labels']
    logging.info(str(len(data['labels']))+' data processed.')
    logging.info('Data limitation set into '+str(limit))

    images = data['images']
    labels = data['labels']
    asianImages = []
    nonasianImages = []
    i = 0
    for target_list in labels[:limit]:
        if labels[i] == 'asian':
            asianImages.append(images[i])
        elif labels[i] == 'non asian':
            nonasianImages.append(images[i])
        i += 1

    logging.info('Balancing asian and non-asian data count...')
    minimumCount = min(len(asianImages), len(nonasianImages))
    logging.info('Using '+str(minimumCount)+' data for each label...')

    X = []
    y = []
    for i in range(len(asianImages)):
        X.append(asianImages[i])
        y.append(0)
        X.append(nonasianImages[i])
        y.append(1)

    logging.info('Balancing process finished.')
    targetNames = numpy.unique(data['labels'])
    logging.info(str(len(y))+' data loaded.')
    return (numpy.array(X), numpy.array(y), targetNames)


def fisherface(X_train, X_test, y_train, y_test):
    logging.debug('Training '+str(len(y_train))+' images...')
    stamp = time.time()
    logging.info('Fitting data into PCA..')
    pca = PCA(n_components=(len(X_train) - 2))
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    logging.info('Data fitting into PCA finished.')
    logging.info('Fitting data into LDA..')
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    X_train = lda.transform(X_train)
    X_test = lda.transform(X_test)
    logging.info('Data fitting into LDA finished.')
    logging.info('Training process finished in ' +
                 str(round((time.time() - stamp), 5))+' seconds.')

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    limit = 4000
    n_neighbors = 5
    dataSplit = 0.2
    logging.basicConfig(
        format='%(asctime)s -- %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler("test.txt"),
            logging.StreamHandler()
        ]
    )
    clear()

    # uncomment the function below to preprocess the dataset into one
    # database called 'dataset.npy'. You need to run this once then you
    # might want to recomment the line below:
    # readDataset(datasetName='fairface')

    logging.info('==========\t '+str(limit)+' data \t==========')
    kf = KFold(n_splits=5)
    X, y, targetNames = loadDataset(limit)
    accuracy = []
    splitter = round(len(y)*dataSplit)
    X_train, X_test = X[splitter:], X[:splitter]
    y_train, y_test = y[splitter:], y[:splitter]
    X_train, X_test, y_train, y_test = fisherface(
        X_train, X_test, y_train, y_test)
    logging.info('Predicting '+str(len(y_test)) +
                 ' data using KNN with n = '+str(n_neighbors))
    neigh = KNeighborsClassifier(n_neighbors)
    neigh.fit(X_train, y_train)
    prediction = neigh.predict(X_test)
    logging.info('Data prediction finished.')
    logging.info('Classification report\n'+classification_report(
        y_test, prediction, target_names=targetNames
    ))
    result = accuracy_score(
        y_test, prediction
    )
    accuracy.append(result)
    logging.info(
        'Accuracy: '+str(round(numpy.mean(accuracy) * 100, 5))+'%')
    logging.info('\n\n\t========================================\t\n\n')
