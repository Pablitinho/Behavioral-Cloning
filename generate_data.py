import pickle

import cv2
import csv
import numpy as np
def get_image(line,id,img_path):
    source_path = line[id]
    file_name = source_path.split('/')[-1]
    #file_name = source_path.split('\\')[-1]
    current_path = img_path + file_name
    image = cv2.imread(current_path)
    return image

def augmentation_shift(image, desplacement):
    num_rows, num_cols = image.shape[:2]

    translation_matrix = np.float32([[1, 0, desplacement], [0, 1, 0]])
    img_translation = cv2.warpAffine(image, translation_matrix, (num_cols, num_rows))
    return img_translation

def create_pickle(log_data,img_path):
    samples = []
    with open(log_data) as csvfile:
        reader = csv.reader(csvfile)
        first_time = True
        for line in reader:

            if first_time == True:
                first_time = False
            else:
                samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # -------------------------------------------------------------------------------------------------------------------
    # Validation generation
    # -------------------------------------------------------------------------------------------------------------------
    images = []
    measurements = []
    cnt = 0
    num_sub_samples = 0
    for line in validation_samples:
        if num_sub_samples < 16:
           num_sub_samples += 1
           image_center = get_image(line, 0, img_path)

           steering_center = float(line[3])
           images.append(image_center)
           measurements.append(steering_center)
        else:
            images = np.array(images)
            measurements = np.array(measurements)
            print("Batch Validation: ", cnt)
            with open('./augmented_data/images_validation_' + str(cnt) + '.pickle', 'wb') as handle:
                 pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./augmented_data/stearing_validation_' + str(cnt) + '.pickle', 'wb') as handle:
                 pickle.dump(measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)
            num_sub_samples = 0
            images = []
            measurements = []
            cnt += 1
    # -------------------------------------------------------------------------------------------------------------------
    # Training generation
    # -------------------------------------------------------------------------------------------------------------------    images = []
    measurements = []
    images = []
    cnt = 0
    num_sub_samples = 0
    for line in train_samples:
        if num_sub_samples < 16:

            num_sub_samples += 1
            image_center = get_image(line, 0, img_path)
            image_left = get_image(line, 1, img_path)
            image_right = get_image(line, 2, img_path)

            steering_center = float(line[3])

            correction = 0.3  # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            #
            images.extend([image_center, image_left, image_right])
            measurements.extend([steering_center, steering_left, steering_right])
            # -------------------------------------------------------
            # Flip image only when steering
            # -------------------------------------------------------
            # if abs(steering_center) > 0.13:
            #     img_flipped = cv2.flip(image_center, 1)
            #     images.append(img_flipped)
            #     measurements.append(-steering_center)
            # -------------------------------------------------------
            # Shift to the left
            # -------------------------------------------------------
            for iter in range(1, 4):
                img_translation = augmentation_shift(image_center, -iter*2)
                images.append(img_translation)
                measurements.append(steering_center-0.05*iter)
            # -------------------------------------------------------
            # Shift to the right
            # -------------------------------------------------------
            for iter in range(1, 4):
                img_translation = augmentation_shift(image_center, iter*2)
                images.append(img_translation)
                measurements.append(steering_center + 0.05 * iter)
        else:
            print("Batch: ", cnt)
            images = np.array(images)
            measurements = np.array(measurements)
            with open('./augmented_data/images_train_' + str(cnt) + '.pickle', 'wb') as handle:
                pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('./augmented_data/stearing_train_' + str(cnt) + '.pickle', 'wb') as handle:
                pickle.dump(measurements, handle, protocol=pickle.HIGHEST_PROTOCOL)
            num_sub_samples = 0
            images = []
            measurements = []
            cnt += 1
    # -------------------------------------------------------------------------------------------------------------------


