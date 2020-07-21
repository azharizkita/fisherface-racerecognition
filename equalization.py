import cv2
import numpy
import matplotlib.pyplot as plt

def load_data(race, blur_threshold=0, histogram_equalization=0, face_detection=False, validation=False):
    if validation:
        if face_detection:
            dataset = numpy.load(str(str(race) + '_' + str(blur_threshold) +
                                    '_' + str(histogram_equalization) + '_val.npy'), allow_pickle=True)[()]
        else:
            dataset = numpy.load(str(race) + '_val.npy', allow_pickle=True)[()]

    else:
        if face_detection:
            dataset = numpy.load(str(str(race) + '_' + str(blur_threshold) +
                                    '_' + str(histogram_equalization) + '.npy'), allow_pickle=True)[()]
        else:
            dataset = numpy.load(str(race) + '.npy', allow_pickle=True)[()]

    images_data = []
    paths_data = []
    for index, img in enumerate(dataset['images'][:5714]):
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # plt.close()
        # plt.figure()
        # plt.imshow(exposure.equalize_hist(img))
        # plt.show()
        # plt.close()
        # images_data.append(numpy.reshape(img, (1, 224 * 224)))
        images_data.append(img)
        paths_data.append(dataset['paths'][index])
    print(len(images_data))
    del dataset
    return images_data, len(images_data), paths_data

def data_equalization(image_base, base_label, opposite_label, path_data, limit):
    equalized_images_base = []
    equalized_labels_base = []
    equalized_paths_base = []

    equalized_images_opposite = []
    equalized_labels_opposite = []
    equalized_paths_opposite = []

    equalized_images, equalized_labels, equalized_paths = [], [], []
    label_names = base_label + opposite_label
    base_equalizer_counter = 0
    opposite_equalizer_counter = 0
    for i in range(limit):
        for index, race in enumerate(label_names):
            if race in opposite_label:
                equalized_images_base.append(image_base[race][i])
                equalized_labels_base.append(1)
                equalized_paths_base.append(path_data[race][i])
            elif race in base_label:
                equalized_images_opposite.append(image_base[race][i])
                equalized_labels_opposite.append(0)
                equalized_paths_opposite.append(path_data[race][i])

    for i in range(limit * 2):
        # plt.figure()
        # plt.imshow(equalized_images_base[i])
        # plt.show()
        # plt.close()
        # plt.figure()
        # plt.imshow(cv2.equalizeHist(equalized_images_base[i]))
        # plt.show()
        # plt.close()
        
        equalized_images.append(numpy.reshape(cv2.equalizeHist(equalized_images_base[i]), (1, 224 * 224)))
        # equalized_images.append(numpy.reshape(equalized_images_base[i], (1, 224 * 224)))
        equalized_labels.append(equalized_labels_base[i])
        equalized_paths.append(equalized_paths_base[i])

        equalized_images.append(numpy.reshape(cv2.equalizeHist(equalized_images_opposite[i]), (1, 224 * 224)))
        # equalized_images.append(numpy.reshape(equalized_images_opposite[i], (1, 224 * 224)))
        equalized_labels.append(equalized_labels_opposite[i])
        equalized_paths.append(equalized_paths_opposite[i])
    # print(numpy.unique(equalized_labels_base, return_counts=True))
    # print(numpy.unique(equalized_labels_opposite, return_counts=True))
    print(numpy.unique(equalized_labels, return_counts=True))
    # if limit is None:
    #     limit = min(len(labels_base), len(labels_opposite))
    # for i in range(limit):

    #     equalized_images.append(image_base[i])
    #     equalized_images.append(image_opposite[i])
    #     label_names.append(global_label.index(labels_base[i]))
    #     label_names.append(global_label.index(labels_opposite[i]))
    #     equalized_labels.append(0)
    #     equalized_labels.append(1)
    # print(numpy.unique(labels_base, return_counts=True))
    # print(numpy.unique(labels_opposite, return_counts=True))
    return numpy.array(equalized_images), numpy.array(equalized_labels), numpy.array(equalized_paths)

if __name__ == "__main__":
    BASE = ['Southeast Asian', 'East Asian']
    OPPOSITE = ['Black', 'White', 'Indian', 'Latino_Hispanic', 'Middle Eastern']
    GLOBAL_LABEL = BASE + OPPOSITE

    trains = dict()
    validations = dict()
    data_count = []
    path = dict()

    for target in GLOBAL_LABEL:
        base_image, total_data, paths_data = load_data(
            target, 0, 0, face_detection=False, validation=True)
        trains[target] = base_image
        path[target] = paths_data
        data_count.append(len(base_image))

    print('Minimal data:', min(data_count))
    X, y, path_image = data_equalization(
        trains, BASE, OPPOSITE, path, min(data_count))
    
    dataset = dict()
    dataset['image'] = X
    dataset['label'] = y
    dataset['path'] = path_image
    numpy.save('val-he', dataset)
