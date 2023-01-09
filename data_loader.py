import cv2
import os
import glob
import numpy as np
import pandas as pd
import random
import util_function as utils

IMAGE_SIZE = 128
NUM_CHANNEL = 3
CROP_RATIO = 0.15


# histogram equalization + cropping + resize + normalize
def preprocessing(image, image_size, histeq=True):

    # hist equalization
    if histeq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    crop_size = round(image.shape[1]*CROP_RATIO)
    image = image[:, crop_size:image.shape[1] - crop_size]    # cropping
    image = np.array(utils.normalize_x(cv2.resize(image, (image_size, image_size))), np.float32)  # normalize(-1, 1)

    return image


def make_data_excel(path):

    data_list = {}

    # dataset folder structure : Type / patient_id / modal / slice_no
    for label_folder in ['AML_crop', 'CCRCC_crop']:
        for id_name in glob.glob1(path + '/' + label_folder + '/', '*'):

            data_list[id_name] = {}
            # data_list['image_id'].append(id_name)
            label = 0 if label_folder == 'AML_crop' or label_folder == 'AML_incomplete' else 1
            data_list[id_name]['label'] = label

            for modal_name in ['pc', 'ec', 'dc', 'tm', 'am']:
                num_slice = len(glob.glob1(path + '/' + label_folder + '/'+ id_name + '/' + modal_name + '/', '*.png'))

                data_list[id_name][modal_name] = num_slice
                print(id_name, label, modal_name, num_slice)

    #  # random both lists and save to excel
    l = list(data_list.items())
    random.shuffle(l)
    data_list_shuffle = dict(l)
    data_list = pd.DataFrame(data_list_shuffle).T
    data_list.to_excel('C:/Users/Job/Documents/DoctorProject/kidney_tumor/dataset/data_list_shuffle2.xlsx',
                       index=True, header=True)


def load_data_from_xlsx(path, modal_lists, cropped=True, image_type=1, image_size=128):
    data_list = pd.DataFrame(pd.read_excel(path + 'data_list_shuffle2.xlsx', engine='openpyxl')).values.tolist()

    images = []
    labels = []

    m_idx = []
    for m, modal in enumerate(modal_lists):
        if modal == 'pc':
            m_idx.append(2)
        if modal == 'ec':
            m_idx.append(3)
        if modal == 'dc':
            m_idx.append(4)
        if modal == 'tm':
            m_idx.append(5)
        if modal == 'am':
            m_idx.append(6)

    for data in data_list:
        id_name = data[0]
        if image_type == 1:  # image

            # find minimum slices in modalities
            num_slice = min([data[i] for i in m_idx])
            if cropped:
                img_type_folder = 'AML_crop/' if data[1] == 0 else 'CCRCC_crop/'
            else:
                img_type_folder = 'AML_old/' if data[1] == 0 else 'CCRCC_old/'

            for n in range(num_slice):
                image_all_modal = np.zeros((image_size, image_size, NUM_CHANNEL * len(modal_lists)), np.float32)
                for m, modal_name in enumerate(modal_lists):
                    try:
                        slice_name = glob.glob1(path + img_type_folder + id_name + '/' + modal_name, '*.png')[n]
                    except:
                        print(path + img_type_folder + id_name + '/' + modal_name)
                        print(slice_name)

                    # print(img_type_folder + id_name + '/' + modal_name + '/' + slice_name)
                    image = cv2.imread(path + img_type_folder + id_name + '/' + modal_name + '/' + slice_name, cv2.IMREAD_GRAYSCALE)
                    image = preprocessing(image, image_size, histeq=False)
                    if NUM_CHANNEL == 1:
                        image_all_modal[:, :, m*NUM_CHANNEL:m*NUM_CHANNEL+NUM_CHANNEL] = image[:, :, np.newaxis]
                    elif NUM_CHANNEL == 3:
                        image_all_modal[:, :, m*NUM_CHANNEL:m*NUM_CHANNEL+NUM_CHANNEL] = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

                images.append(image_all_modal)
                labels.append(data[1])

        elif image_type == 0:  # mask

            # find minimum slices in modalities
            num_slice = min([data[i] for i in m_idx])
            if cropped:
                img_type_folder = 'AML_crop/' if data[1] == 0 else 'CCRCC_crop/'
            else:
                img_type_folder = 'AML_old/' if data[1] == 0 else 'CCRCC_old/'

            for n in range(num_slice):
                image_all_modal = np.zeros((image_size, image_size, len(modal_lists)), np.float32)
                for m, modal_name in enumerate(modal_lists):
                    slice_name = glob.glob1(path + img_type_folder + id_name + '/' + modal_name, '*.png')[n]
                    image = cv2.imread(path + img_type_folder + id_name + '/' + modal_name + 'L/' + slice_name)
                    # print(img_type_folder + id_name + '/' + modal_name + 'L/' + slice_name)

                    # preprocessing
                    crop_size = round(image.shape[1] * CROP_RATIO)
                    image = image[:, crop_size:image.shape[1] - crop_size]  # cropping

                    # create mask (red label)
                    mask = np.zeros_like(image[:, :, 0])
                    mask[image[:, :, 2] == 255] = 255
                    mask[image[:, :, 0] > 0] = 0
                    image = np.array(utils.normalize_y(cv2.resize(mask, (image_size, image_size))), np.float32)

                    image_all_modal[:, :, m] = image

                images.append(image_all_modal)
                labels.append(data[1])

    images = np.stack(images, axis=0)
    labels = np.expand_dims(np.stack(labels, axis=0), axis=-1)
    return images, labels


def load_data_from_folder(path, modal_lists, image_type=1):
    # dataset folder structure : Type / patient_id / modal / slice_no

    # get total files
    total_files = 0
    for root, dirs, files in os.walk(path):
        if root[-2:] + '/' in modal_lists:
            total_files += len(files)

    print(path)
    print('total_files in folders = ' + str(total_files))
    channels = 3 if image_type == 1 else 1
    images = np.zeros((total_files, IMAGE_SIZE, IMAGE_SIZE, channels), np.float32)
    image_names = []
    idx = 0
    for id_name in glob.glob1(path, '*'):
        for modal_name in modal_lists:
            if image_type == 1:   # image
                for slice_name in glob.glob1(path + id_name + '/' + modal_name, '*.png'):
                    image_names.append(id_name + '_' + modal_name[:-1] + '_' + slice_name)
                    image = cv2.imread(path + id_name + '/' + modal_name + slice_name)
                    image = preprocessing(image, histeq=False)
                    images[idx] = image
                    idx += 1

            elif image_type == 0:   # gt
                for slice_name in glob.glob1(path + id_name + '/' + modal_name, '*.png'):
                    image_names.append(id_name + '_' + modal_name[:-1] + '_' + slice_name)
                    image = cv2.imread(path + id_name + '/' + modal_name[:-1] + 'L/' + slice_name)
                    crop_size = round(image.shape[1]*CROP_RATIO)
                    image = image[:, crop_size:image.shape[1]-crop_size]     # cropping

                    # create mask (red label)
                    mask = np.zeros_like(image[:, :, 0])
                    mask[image[:, :, 2] == 255] = 255
                    mask[image[:, :, 0] > 0] = 0

                    image = np.array(utils.normalize_y(cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))), np.float32)
                    images[idx] = image[:, :, np.newaxis]
                    idx += 1

    return images, image_names


# make_data_excel('C:/Users/Job/Documents/DoctorProject/kidney_tumor/dataset')
# modal_lists = ['ec', 'dc']
# images, labels = load_data_from_xlsx('C:/Users/Job/Documents/DoctorProject/kidney_tumor/dataset/', modal_lists, image_type=0)
# print(images.shape)
# print(labels.shape)