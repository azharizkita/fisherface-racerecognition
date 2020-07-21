import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import IncrementalPCA
import cv2
import gc 
import warnings
import pickle
from sklearn.metrics import plot_confusion_matrix
warnings.filterwarnings("ignore", category=RuntimeWarning) 
numpy.seterr(all='ignore')

prefix = ''
def principal_component_analysis(train, test, label, n_component, c_component):
    chunk_size = 200
    # pca = IncrementalPCA(n_components=chunk_size - c_component, batch_size=200)
    # n = len(train)
    # if n > chunk_size:
    #     for i in range(0, int(n/chunk_size)):
    #         print('Partial fit -', i)
    #         pca.partial_fit(train[i*chunk_size : (i+1)*chunk_size])
    # else:
    #     pca.fit(train)
    # pickle.dump(pca, open("pca.pkl","wb"))

    pca = pickle.load(open("pca.pkl",'rb'))
    # weight_pca_train = pca.transform(train)
    # pickle.dump(weight_pca_train, open("Wpca_train.pkl","wb"))
    weight_pca_train = pickle.load(open("Wpca_train.pkl",'rb'))
    weight_pca_test = pca.transform(test)

    cols = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'gray']
    plt_col = [cols[int(k)] for k in label]
    plt.figure()
    plt.scatter(weight_pca_train[:, 0], weight_pca_train[:, 1], color=plt_col)
    plt.show()
    plt.close()

    eigenfaces = pca.components_.reshape(
        ((chunk_size - c_component), 224, 224))
    i = 0
    eigen_plot = eigenfaces[:10]
    for img in eigen_plot:
        i += 1
        plt.subplot(2, 5, i)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    return weight_pca_train, weight_pca_test, pca


def linear_discriminant_analysis(weight_pca_train, weight_pca_test, pca, label_trains, class_component):
    lda = LinearDiscriminantAnalysis().fit(weight_pca_train, label_trains)
    lda.fit(weight_pca_train, label_trains)
    weight_fld_train = lda.transform(weight_pca_train)
    weight_fld_test = lda.transform(weight_pca_test)

    cols = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'gray']
    plt_col = [cols[int(k)] for k in label_trains]
    plt.figure()
    plt.scatter(weight_fld_train[:, 0], weight_fld_train[:, 0], color=plt_col)
    plt.show()
    plt.close()

    for i in range(class_component - 1):
        fisherface_feature = pca.inverse_transform(lda.scalings_[:, i])
        fisherface_feature.shape = [224, 224]
        plt.figure()
        plt.imshow(fisherface_feature)
        plt.axis('off')
        plt.show()
        plt.close()

    return weight_fld_train, weight_fld_test


def fisherface(train_image, test_image, train_labels):
    c = len(numpy.unique(train_labels))
    n = len(train_image)
    train_images = numpy.reshape(train_image, (len(train_image), 224 * 224))
    test_images = numpy.reshape(test_image, (len(test_image), 224 * 224))

    weight_pca_train, weight_pca_test, pca = principal_component_analysis(
        train_images, test_images, train_labels, n, c)
        
    weight_fld_train, weight_fld_test = linear_discriminant_analysis(
        weight_pca_train, weight_pca_test, pca, train_labels, c)
    return weight_fld_train, weight_fld_test


def classification(utrain, utest, label_utrain, label_utest, base, opposite, path):
    K = [60]
    labels = [base, opposite]
    for k in K:
        neigh = KNeighborsClassifier(k)
        neigh.fit(utrain, label_utrain)
        prediction = neigh.predict(utest)
        true = 0
        false = 0
        for i, path_name in enumerate(path):
            if label_utest[i] == prediction[i]:
                print('')
                if true < 5:
                    plt.subplot(2, 5, 1+true+false)
                    plt.axis('off')
                    plt.title('TRUE, as '+labels[prediction[i]])
                    plt.imshow(plt.imread('./fairface/val/'+str(path_name)+'.jpg'))
                    true+=1
            else:
                if false < 5:
                    plt.subplot(2, 5, 1+true+false)
                    plt.axis('off')
                    plt.title('FALSE, as '+labels[prediction[i]])
                    plt.imshow(plt.imread('./fairface/val/'+str(path_name)+'.jpg'))
                    false += 1
        plt.show()
        print(classification_report(
            label_utest, prediction, target_names=labels
        ))
        print(accuracy_score(label_utest, prediction))
        disp = plot_confusion_matrix(neigh, utest, label_utest,
                            display_labels=labels,
                            cmap=plt.cm.Blues)
        plt.show()
        plt.close()


if __name__ == "__main__":
    numpy.seterr(all='ignore')
    print("Garbage collection thresholds:", 
                    gc.get_threshold())

    train = numpy.load(prefix+ 'train-he.npy', allow_pickle=True)[()]
    test = numpy.load(prefix+ 'val-he.npy', allow_pickle=True)[()]

    X_train = train['image']
    y_train = train['label']
    X_test = test['image'] 
    y_test = test['label']
    y_path = test['path']

    Utrain, Utest = fisherface(X_train, X_test, y_train)

    classification(Utrain, Utest, y_train, y_test, 'Asian', 'Non Asian', y_path[:50])
