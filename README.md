## About this program
This program is meant to classify the race of facial images between asian and non-asian. Here is some methods that I implement:
### Face Detection
* OpenCV's [Haar casecade detection](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html).

### Image Enhancements
1. OpenCV's [Histogram Equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html).
2. (More enhancements suggestions are welcome)

### Feature extraction
* Fisherface. This is basically a combination of sklearn's [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and sklearn's [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

### Classifier
* Sklearn's [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


## Requirements
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Numpy](https://pypi.org/project/numpy/)
* [Sklearn](https://pypi.org/project/scikit-learn/)

## How to use this?
* You need to uncomment this once, feel free to recomment after you ran it once. This code supposed to preprocess the dataset then it puts the output into one database called '`dataset.npy`'.
```python
# readDataset(datasetName='fairface')
```
