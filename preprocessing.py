import os
import cv2
import numpy
import matplotlib.pyplot as plt


def face_detection(image_path, blur_threshold, histogram_equalization):
    face_cascade = cv2.CascadeClassifier(
        os.path.join(os.path.curdir, 'haarcascade_frontalface_default.xml')
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(os.path.curdir, 'haarcascade_eye.xml')
    )
    gray = cv2.imread(image_path, 0)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=20,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    elif len(faces) == 1:
        # print(image_path, 'Face detected.')
        for (faces_x, faces_y, faces_w, faces_h) in faces:
            roi_gray = gray[faces_y:faces_y +
                            faces_h, faces_x:faces_x+faces_w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 2:
                # print(image_path, 'Eyes detected.')
                if histogram_equalization == 1:
                    plt.figure()
                    plt.imshow(cv2.imread(image_path, -1))
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.imshow(gray)
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.imshow(cv2.equalizeHist(gray))
                    plt.show()
                    plt.close()
                    return cv2.equalizeHist(gray)
                else:
                    plt.figure()
                    plt.imshow(cv2.imread(image_path, -1))
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.imshow(gray)
                    plt.show()
                    plt.close()
                    plt.figure()
                    plt.imshow(cv2.equalizeHist(gray))
                    plt.show()
                    plt.close()
                    return gray
            else:
                return None


if __name__ == "__main__":
    BLUR_THRESHOLD = [0]
    HISTOGRAM_EQUALIZATION = [0]
    root = os.path.join(os.path.curdir, 'fairface')
    BASE = ['Southeast Asian', 'East Asian']
    OPPOSITE = ['Black', 'White', 'Indian', 'Latino_Hispanic', 'Middle Eastern']
    LABELS = BASE + OPPOSITE
    for histogram_equalization in HISTOGRAM_EQUALIZATION:
        for threshold_blur in BLUR_THRESHOLD:
            for race in LABELS:
                dataset = dict()
                images = []
                paths = []
                i = 0
                for row in open(os.path.join(root, 'validation.csv'), mode='r'):
                    if i == 0:
                        i += 1
                        continue
                    row = row.strip().split(',')
                    path = row[0].split('/')
                    path = os.path.join(path[0], path[1])
                    path = os.path.join(root, path)
                    image_id = row[0].split('/')[1].split('.')[0]
                    if row[3] == race:
                        image = face_detection(
                            path, threshold_blur, histogram_equalization)
                        # image = cv2.imread(path, 0)
                        if image is not None:
                            images.append(image)
                            paths.append(image_id)
                            print(row[3], image_id)
                    i += 1
                dataset['images'] = images
                dataset['paths'] = paths
                # numpy.save(str(race) + '_' + str(threshold_blur) +
                #         '_' + str(histogram_equalization), dataset)
                numpy.save(str(race) + '_val', dataset)
