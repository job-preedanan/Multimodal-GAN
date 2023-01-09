import cv2
import tensorflow as tf
import random
import util_function as utils
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
# import config as cfg
import numpy as np
import os
from data_loader import load_data_from_xlsx
from matplotlib.colors import NoNorm
from itertools import combinations


PATH = 'C:/Users/Job/Documents/DoctorProject/kidney_tumor/'
GAN_PATH = PATH + 'GAN/'
DATA_FOLDER = 'dataset/'
EXPORT_FOLDER = 'exports/crop_images/cgan_L1+d+2content_augment/'
IMAGE_SIZE = 128
MODAL_CH = 5
BATCH_SIZE = 16
STEP = 15001
SAMPLE_INTERVAL = 100
CROP_RATIO = 0
CROSS_NUM = 0


# histogram equalization + cropping + resize + normalize
def preprocessing(image, histeq=True):
    # hist equalization
    if histeq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    crop_size = round(image.shape[1] * CROP_RATIO)
    image = image[:, crop_size:image.shape[1] - crop_size]  # cropping
    image = np.array(utils.normalize_x(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))), np.float32)  # normalize(-1, 1)

    return image


# list moving function
def list_index_move(list, split_num):
    split_idx = int(round(len(list) * split_num))  # split index
    new_list = list[split_idx:]
    new_list = np.concatenate([new_list, list[:split_idx]])
    return new_list


# data spliting function
def split_train_test(x, y, val_ratio=0.2, random_sample=True):
    # zip x and y
    samples = list(zip(x, y))
    if random_sample:
        random.shuffle(samples)

    split_idx = int(round(len(samples) * val_ratio))  # split index
    test = samples[:split_idx]
    train = samples[split_idx:]

    # unzip and convert to array
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


def batch_resize(batch_images, img_size):
    images = np.zeros((len(batch_images), img_size, img_size))
    for b in range(len(batch_images)):
        images[b] = cv2.resize(batch_images[b], (img_size, img_size))

    return images


class Pix2Pix():
    def __init__(self):
        # Input shape
        self.num_classes = 2
        self.img_rows = IMAGE_SIZE  # 128
        self.img_cols = IMAGE_SIZE  # 128
        self.channels = 5
        self.batch_size = 16
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.non_available_modals = np.ones((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, self.channels))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['mse'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # self.global_discriminator = self.build_discriminator()
        # self.local_discriminator = self.build_local_discriminator()
        # self.global_discriminator.compile(loss=['mse'],
        #                                   optimizer=optimizer,
        #                                   metrics=['accuracy'])
        # self.local_discriminator.compile(loss=['mse'],
        #                                  optimizer=optimizer,
        #                                  metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Use a pre-trained VGG19 model to extract image features
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Input images and their conditioning images
        img_all = Input(shape=self.img_shape)
        img_some = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        gen_img1 = self.generator(img_some)
        # [gen_img1, gen_img2] = self.generator(img_some)

        # For the combined model we will only train the generator
        # self.global_discriminator.trainable = False
        # self.local_discriminator.trainable = False
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([gen_img1, img_some])
        # g_valid = self.global_discriminator([gen_img2, img_some])
        # gen_non_available_modals = gen_img2 * self.non_available_modals
        # l_valid = self.local_discriminator(gen_non_available_modals)

        # output for vgg loss
        # images_up1 = UpSampling2D(size=2)(gen_img1[:, :, :, 0][:, :, :, np.newaxis])    # 256 * 256
        # # images_up1 = gen_img1[:, :, :, 0][:, :, :, np.newaxis]
        # gen_features1 = self.vgg(Concatenate()([images_up1, images_up1, images_up1]))
        # images_up2 = UpSampling2D(size=2)(gen_img1[:, :, :, 1][:, :, :, np.newaxis])    # 256 * 256
        # # images_up2 = gen_img1[:, :, :, 1][:, :, :, np.newaxis]
        # gen_features2 = self.vgg(Concatenate()([images_up2, images_up2, images_up2]))
        # images_up3 = UpSampling2D(size=2)(gen_img1[:, :, :, 2][:, :, :, np.newaxis])    # 256 * 256
        # # images_up3 = gen_img1[:, :, :, 2][:, :, :, np.newaxis]
        # gen_features3 = self.vgg(Concatenate()([images_up3, images_up3, images_up3]))
        # images_up4 = UpSampling2D(size=2)(gen_img1[:, :, :, 3][:, :, :, np.newaxis])    # 256 * 256
        # # images_up4 = gen_img1[:, :, :, 3][:, :, :, np.newaxis]
        # gen_features4 = self.vgg(Concatenate()([images_up4, images_up4, images_up4]))
        # images_up5 = UpSampling2D(size=2)(gen_img1[:, :, :, 4][:, :, :, np.newaxis])    # 256 * 256
        # # images_up5 = gen_img1[:, :, :, 4][:, :, :, np.newaxis]
        # gen_features5 = self.vgg(Concatenate()([images_up5, images_up5, images_up5]))

        images_ct = UpSampling2D(size=2)(gen_img1[:, :, :, :3])  # 1-3 ch , 256 * 256
        gen_features_ct = self.vgg(images_ct)

        images_mri = Concatenate()([gen_img1[:, :, :, 0][:, :, :, np.newaxis], gen_img1[:, :, :, 3:]])
        images_mri = UpSampling2D(size=2)(images_mri)  # 1 + 4-5 ch , 256 * 256
        gen_features_mri = self.vgg(images_mri)

        self.combined = Model(inputs=[img_all, img_some], outputs=[valid,
                                                                   gen_img1,
                                                                   gen_features_ct,
                                                                   gen_features_mri])

        self.combined.compile(loss=['mse',
                                    'mae',
                                    self.content_loss,
                                    self.content_loss,
                                    self.content_loss],
                              loss_weights=[1, 50, 1, 1],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)  # 128

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)  # 64
        d2 = conv2d(d1, self.gf * 2)  # 32
        d3 = conv2d(d2, self.gf * 4)  # 16
        d4 = conv2d(d3, self.gf * 8)  # 8
        d5 = conv2d(d4, self.gf * 8)  # 4

        # Upsampling
        u1 = deconv2d(d5, d4, self.gf * 8)
        u2 = deconv2d(u1, d3, self.gf * 4)
        u3 = deconv2d(u2, d2, self.gf * 2)
        u4 = deconv2d(u3, d1, self.gf)
        u5 = UpSampling2D(size=2)(u4)

        output_img1 = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)

        # # Downsampling 2
        # d1_2 = conv2d(output_img1, self.gf, bn=False)  # 128
        # d2_2 = conv2d(d1_2, self.gf*2)          # 64
        # d3_2 = conv2d(d2_2, self.gf*4)          # 32
        # d4_2 = conv2d(d3_2, self.gf*8)          # 16
        # d5_2 = conv2d(d4_2, self.gf*8)          # 8
        #
        # # Upsampling 2
        # u1_2 = deconv2d(d5_2, d4_2, self.gf*8)
        # u2_2 = deconv2d(u1_2, d3_2, self.gf*4)
        # u3_2 = deconv2d(u2_2, d2_2, self.gf*2)
        # u4_2 = deconv2d(u3_2, d1_2, self.gf)
        # u5_2 = UpSampling2D(size=2)(u4_2)
        #
        # output_img2 = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5_2)

        return Model(d0, output_img1)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_some = Input(shape=self.img_shape)
        img_all = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_all, img_some])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_all, img_some], validity)

    def build_local_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_all = Input(shape=self.img_shape)

        d1 = d_layer(img_all, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img_all, validity)

    def build_cc_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_some = Input(shape=self.img_shape)
        img_all = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_all, img_some])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        label = Flatten()(d4)
        label = Dense(self.num_classes+1, activation="softmax")(label)

        return Model([img_all, img_some], [validity, label])

    def build_vgg(self):

        vgg = VGG19(weights='imagenet', include_top=False)
        vgg.trainable = False

        img_feature1 = vgg.get_layer('block1_conv1').output
        img_feature2 = vgg.get_layer('block2_conv1').output
        img_feature3 = vgg.get_layer('block3_conv1').output
        img_feature4 = vgg.get_layer('block4_conv1').output

        return Model(inputs=vgg.input, outputs=[img_feature1, img_feature2, img_feature3, img_feature4])

    def content_loss(self, ref_features, gen_features):

        # feaures_w = [1, 1, 1, 1]
        # for i in range(ref_features):
        #     mae_feature = feaures_w[i] * tf.reduce_mean(tf.abs(ref_features[i] - gen_features[i]))

        mae_features = tf.reduce_mean(tf.abs(ref_features - gen_features))

        return tf.reduce_sum(mae_features)

    def available_mae(self, ref, gen):
        weights = self.available_modals
        return tf.reduce_mean(weights * tf.abs(ref - gen))

    def non_available_mae(self, ref, gen):
        weights = self.non_available_modals
        return tf.reduce_mean(weights * tf.abs(ref - gen))

    def augmentation(self, batch_images):
        import albumentations as A

        # define augmentation methods
        transform = A.Compose([
            A.RandomResizedCrop(p=0.5, height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.61, 0.82), ratio=(0.1, 4.01)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(p=0.5, limit=(-20, 20), rotate_method='largest_box', crop_border=False)
        ])

        augmented_batch_images = np.ones_like(batch_images)
        for b in range(len(batch_images)):
            # random fixed seed
            seed = random.randint(1, 200)

            # f, axarr = plt.subplots(2, batch_images.shape[3])
            for m in range(batch_images.shape[3]):
                random.seed(seed)
                augmented_batch_images[b, :, :, m] = transform(image=batch_images[b, :, :, m])['image']

            #     # display sample images
            #     axarr[0, m].grid(False)
            #     axarr[0, m].imshow(utils.denormalize_x(batch_images[b, :, :, m]), cmap='gray', vmin=0, vmax=255)
            #     axarr[1, m].imshow(utils.denormalize_x(augmented_batch_images[b, :, :, m]), cmap='gray', vmin=0, vmax=255)
            # plt.show()

        return augmented_batch_images

    def apply_modal_available_masks(self, images, max_mask_num=2, all_combinations=False):

        # return all possible combination
        if all_combinations:
            m_idx = list(range(0, images.shape[3]))  # create list of modal idx
            all_comp_idx = list(combinations(m_idx, max_mask_num))  # get all combinations
            comp_length = sum(1 for _ in all_comp_idx)  # get size of combinations
            masked_modals = np.zeros((comp_length * len(images), images.shape[1], images.shape[2], images.shape[3]))
            m_i = 0  # all comp idex for masked modals
            for b in range(len(images)):
                for comp_idx in all_comp_idx:
                    masked_modals[m_i] = images[b].copy()
                    for idx in list(comp_idx):
                        masked_modals[m_i, :, :, idx] = np.zeros_like(images[0, :, :, 0])
                    m_i += 1
            mask_flag = []

        # return random masked modalities
        else:
            masked_modals = images.copy()
            mask_flag = np.ones((len(images), images.shape[1], images.shape[2], 5))
            for b in range(len(masked_modals)):
                # mask_num = np.random.randint(1, max_mask_num+1)     # randomly select no. of mask
                mask_num = max_mask_num
                mask_idx = np.random.choice(images.shape[3], mask_num, replace=False)
                for idx in mask_idx:
                    mask_flag[b, :, :, idx] = np.zeros_like(images[0, :, :, 0])
                    masked_modals[b, :, :, idx] = np.zeros_like(images[0, :, :, 0])
                # print(mask_flag[b])

        return masked_modals, mask_flag

    def train(self, samples, epochs, batch_size=16, sample_interval=100, step_masks=False):

        # Load the dataset (all modalities)
        y_train = samples[0]
        y_label = samples[1]
        x_train = y_train.copy()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        g_train_interval = 1
        d_train_interval = 3
        save_interval = 1000

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)

            # if epoch < 1000:
            #     non_available_mask = step_masks[0]
            # elif 1000 <= epoch < 5000:
            #     non_available_mask = step_masks[1]
            # elif 5000 <= epoch < 10000:
            #     non_available_mask = step_masks[2]
            # elif epoch >= 10000:
            #     non_available_mask = step_masks[3]

            # augmentation
            augmented_x_train = self.augmentation(x_train[idx])

            imgs_some_modals, self.available_modals = self.apply_modal_available_masks(augmented_x_train, max_mask_num=2)
            self.non_available_modals = np.where(self.available_modals > 0.5, 0, 1)
            # mask_flag = tf.convert_to_tensor(mask_flag)
            # non_mask_flag = tf.convert_to_tensor(non_mask_flag)
            imgs_all_modals = augmented_x_train  # y_train[idx]


            # One-hot encoding of labels
            valid_label = to_categorical(labels[idx], num_classes=self.num_classes+1)
            fake_label = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # # display sample images
            # f, axarr = plt.subplots(2, imgs_all_modals.shape[3])
            # idx = random.randint(0, len(imgs_all_modals))
            # for m in range(imgs_all_modals.shape[3]):
            #     # axarr[0, m].set_title(modal)
            #     axarr[0, m].grid(False)
            #     axarr[0, m].imshow(utils.denormalize_y(self.available_modals[idx, :, :, m]), cmap='gray', vmin=0, vmax=255)
            #     axarr[1, m].imshow(utils.denormalize_y(self.non_available_modals[idx, :, :, m]), cmap='gray', vmin=0, vmax=255)

            # plt.show()

            # Generate a batch of new images
            # [gen_all_modals1, gen_all_modals2] = self.generator.predict(imgs_some_modals)
            gen_all_modals2 = self.generator.predict(imgs_some_modals)

            # update discriminator
            if (epoch == 0) or (epoch % d_train_interval == 0):
                # Train the conditional discriminator [concat(source, target) / concat(source, gen)]
                # d_global_loss_real = self.discriminator.train_on_batch([imgs_all_modals, imgs_some_modals], [valid, valid_label])
                # d_global_loss_fake = self.discriminator.train_on_batch([gen_all_modals2, imgs_some_modals], [fake, fake_label])
                d_global_loss_real = self.discriminator.train_on_batch([imgs_all_modals, imgs_some_modals], valid)
                d_global_loss_fake = self.discriminator.train_on_batch([gen_all_modals2, imgs_some_modals], fake)
                d_loss = 0.5 * np.add(d_global_loss_real, d_global_loss_fake)

                # imgs_non_available_modals = imgs_all_modals * self.non_available_modals
                # gen_non_available_modals = gen_all_modals2 * self.non_available_modals

                # d_local_loss_real = self.local_discriminator.train_on_batch(imgs_non_available_modals, valid)
                # d_local_loss_fake = self.local_discriminator.train_on_batch(gen_non_available_modals, fake)
                # d_local_loss = 0.5 * np.add(d_local_loss_real, d_local_loss_fake)

                # display sample images
                # f, axarr = plt.subplots(3, imgs_all_modals.shape[3])
                # idx = random.randint(0, len(imgs_all_modals))
                # for m in range(imgs_all_modals.shape[3]):
                #     # axarr[0, m].set_title(modal)
                #     axarr[0, m].grid(False)
                #     axarr[0, m].imshow(utils.denormalize_y(self.non_available_modals[idx, :, :, m]), cmap='gray',
                #                        vmin=0, vmax=255)
                #     axarr[1, m].imshow(utils.denormalize_x(imgs_non_available_modals[idx, :, :, m]), cmap='gray',
                #                        vmin=0, vmax=255)
                #     axarr[2, m].imshow(utils.denormalize_x(gen_non_available_modals[idx, :, :, m]), cmap='gray',
                #                        vmin=0, vmax=255)
                # plt.show()

            # ---------------------
            #  Train Generator
            # ---------------------

            if (epoch == 0) or (epoch % g_train_interval == 0):
                # resize to 256*256
                imgs_modal1 = batch_resize(imgs_all_modals[:, :, :, 0], 256)
                imgs_modal2 = batch_resize(imgs_all_modals[:, :, :, 1], 256)
                imgs_modal3 = batch_resize(imgs_all_modals[:, :, :, 2], 256)
                imgs_modal4 = batch_resize(imgs_all_modals[:, :, :, 3], 256)
                imgs_modal5 = batch_resize(imgs_all_modals[:, :, :, 4], 256)

                # Extract ground truth image features using pre-trained VGG19 model
                # imgs_modal1 = np.repeat(imgs_modal1[:, :, :, np.newaxis], 3, axis=3)
                # imgs_modal2 = np.repeat(imgs_modal2[:, :, :, np.newaxis], 3, axis=3)
                # imgs_modal3 = np.repeat(imgs_modal3[:, :, :, np.newaxis], 3, axis=3)
                # imgs_modal4 = np.repeat(imgs_modal4[:, :, :, np.newaxis], 3, axis=3)
                # imgs_modal5 = np.repeat(imgs_modal5[:, :, :, np.newaxis], 3, axis=3)

                # image_features1 = self.vgg.predict(imgs_modal1)
                # image_features2 = self.vgg.predict(imgs_modal2)
                # image_features3 = self.vgg.predict(imgs_modal3)
                # image_features4 = self.vgg.predict(imgs_modal4)
                # image_features5 = self.vgg.predict(imgs_modal5)

                image_features1 = self.vgg.predict(np.concatenate((imgs_modal1[:, :, :, np.newaxis],
                                                                   imgs_modal2[:, :, :, np.newaxis],
                                                                   imgs_modal3[:, :, :, np.newaxis]), axis=3))
                image_features2 = self.vgg.predict(np.concatenate((imgs_modal1[:, :, :, np.newaxis],
                                                                   imgs_modal4[:, :, :, np.newaxis],
                                                                   imgs_modal5[:, :, :, np.newaxis]), axis=3))

                # mag_imgs = self.image_gradient_magnitude(imgs_all_modals)(16, 5))
                g_loss = self.combined.train_on_batch([imgs_all_modals, imgs_some_modals],
                                                      [valid,
                                                       imgs_all_modals,
                                                       image_features1,
                                                       image_features2])

            # Plot the progress
            print("[Epoch %d/%d] [global D loss: %f, acc: %3d%%] [G loss: %f]" % (
                epoch, epochs, d_loss[0], 100 * d_loss[1], np.sum(g_loss[0])))

            # print("[Epoch %d/%d] [global D loss: %f, acc: %3d%%] [local D loss: %f, acc: %3d%%] [G loss: %f]" % (
            #     epoch, epochs, d_loss[0], 100 * d_loss[1], d_local_loss[0], 100 * d_local_loss[1], np.sum(g_loss[0])))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                idx = np.random.randint(0, x_train.shape[0], 2)
                augmented_x_test = self.augmentation(x_train[idx])
                test_images, _ = self.apply_modal_available_masks(augmented_x_test, max_mask_num=2)
                # test_gt = y_train[idx]
                # self.sample_images(epoch, test_images, test_gt)
                self.sample_images(epoch, test_images, augmented_x_test)

            if epoch % save_interval == 0:
                self.save_model(epoch)

    def sample_images(self, epoch, test_images, test_gt):

        # [gen_all_modals1, gen_all_modals2] = self.generator.predict(test_images)
        gen_all_modals2 = self.generator.predict(test_images)

        imgs_some_modals = 0.5 * test_images + 0.5
        imgs_all_modals = 0.5 * test_gt + 0.5
        # gen_all_modals1 = 0.5 * gen_all_modals1 + 0.5
        gen_all_modals2 = 0.5 * gen_all_modals2 + 0.5

        num_samples = 2
        fig, axarr = plt.subplots(int(4 * num_samples), imgs_all_modals.shape[3])
        for n in range(num_samples):
            for m, modal in enumerate(modal_lists):
                axarr[0, m].set_title(modal)
                axarr[n * 4, m].imshow(imgs_all_modals[n, :, :, m], cmap="gray", norm=NoNorm())
                axarr[n * 4, m].axis('off')
                axarr[n * 4 + 1, m].imshow(imgs_some_modals[n, :, :, m], cmap="gray", norm=NoNorm())
                axarr[n * 4 + 1, m].axis('off')
                axarr[n * 4 + 2, m].imshow(gen_all_modals2[n, :, :, m], cmap="gray", norm=NoNorm())
                axarr[n * 4 + 2, m].axis('off')
                axarr[n * 4 + 3, m].imshow(gen_all_modals2[n, :, :, m], cmap="gray", norm=NoNorm())
                axarr[n * 4 + 3, m].axis('off')

        # plt.show()
        fig.savefig(GAN_PATH + os.sep + EXPORT_FOLDER + '/samples/%d.png' % epoch)
        plt.close()

    def save_model(self, epoch_number):

        def save(model, model_name):
            model_path = GAN_PATH + os.sep + EXPORT_FOLDER + '/saved_model/%s.json' % model_name
            weights_path = GAN_PATH + os.sep + EXPORT_FOLDER + '/saved_model/%s_weights.hdf5' % model_name
            options = {'file_arch': model_path,
                       'file_weight': weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, 'generator_' + str(epoch_number))
        # save(self.global_discriminator, 'discriminator_' + str(epoch_number))
        # save(self.local_discriminator, 'l_discriminator_' + str(epoch_number))

    def predict(self, samples, saved_itr):
        import math
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

        # remove some modals
        test_images, _ = self.apply_modal_available_masks(samples, max_mask_num=2, all_combinations=True)

        self.generator.load_weights(GAN_PATH + os.sep + EXPORT_FOLDER +
                                    '/saved_model/generator_' + str(saved_itr) + '_weights.hdf5')
        # [gen_all_modals1, gen_all_modals] = self.generator.predict(test_images, batch_size=16)
        gen_all_modals = self.generator.predict(test_images, batch_size=16)

        total_modals = gen_all_modals.shape[-1]
        ssim_values = np.zeros(len(gen_all_modals) * total_modals)
        mse_values = np.zeros(len(gen_all_modals) * total_modals)
        psnr_values = np.zeros(len(gen_all_modals) * total_modals)

        modal_label = np.zeros(len(gen_all_modals) * total_modals)
        available_label = np.zeros(len(gen_all_modals) * total_modals)

        for i in range(len(gen_all_modals)):
            j = int(math.floor(i / 10))

            org_image = 0.5 * np.squeeze(samples[j]) + 0.5
            test_image = 0.5 * np.squeeze(test_images[i]) + 0.5
            # gen_image1 = 0.5 * np.squeeze(gen_all_modals1[i]) + 0.5
            gen_image = 0.5 * np.squeeze(gen_all_modals[i]) + 0.5

            fig, axs = plt.subplots(4, 5)
            for m, modal in enumerate(modal_lists):
                # axs[0, m].set_title(modal)
                # axs[0, m].imshow(org_image[:, :, m], cmap="gray", norm=NoNorm())
                # axs[0, m].axis('off')
                # axs[1, m].imshow(test_image[:, :, m], cmap="gray", norm=NoNorm())
                # axs[1, m].axis('off')
                # axs[2, m].imshow(gen_image1[:, :, m], cmap="gray", norm=NoNorm())
                # axs[2, m].axis('off')
                # axs[3, m].imshow(gen_image[:, :, m], cmap="gray", norm=NoNorm())
                # axs[3, m].axis('off')

                modal_label[total_modals * i + m] = m + 1
                available_label[total_modals * i + m] = 0 if np.sum(test_images[i, 50:60, 50:60, m]) == 0 else 1

                ssim_values[total_modals * i + m] = ssim(org_image[:, :, m], gen_image[:, :, m])
                mse_values[total_modals * i + m] = mean_squared_error(org_image[:, :, m], gen_image[:, :, m])
                psnr_values[total_modals * i + m] = peak_signal_noise_ratio(org_image[:, :, m], gen_image[:, :, m])
                # print('MSE = ' + str(mse_values[total_modals*i + m]) + '| SSIM = ' + str(ssim_values[total_modals*i + m]))

            # fig.savefig(GAN_PATH + os.sep + EXPORT_FOLDER +
            #             '/test_results/' + str(i) + '.png')
            # plt.close()

        # mean_ssim = np.mean(ssim_values)
        # mean_mse = np.mean(mse_values)
        # mean_psnr = np.mean(psnr_values)

        return mse_values, psnr_values, ssim_values, modal_label, available_label

    def compare_test_samples(self, samples, iteration_list, sample_num=5):
        from skimage.metrics import structural_similarity as ssim

        idx = np.random.randint(0, len(samples), sample_num)
        test_images, _ = self.apply_modal_available_masks(samples[idx], max_mask_num=2, all_combinations=False)

        # (#itr, #samples, h, w, #ch)
        all_gen_data = np.zeros((len(iteration_list), sample_num, self.img_rows, self.img_cols, self.channels))
        # generated results from GAN
        for j, itr in enumerate(iteration_list):
            self.generator.load_weights(GAN_PATH + os.sep + EXPORT_FOLDER +
                                        '/saved_model/generator_' + str(itr) + '_weights.hdf5')
            # [gen_all_modals1, gen_all_modals] = self.generator.predict(test_images, batch_size=16)
            gen_all_modals = self.generator.predict(test_images, batch_size=16)
            gen_all_modals = 0.5 * np.squeeze(gen_all_modals) + 0.5
            all_gen_data[j] = gen_all_modals

        org_image = 0.5 * np.squeeze(samples[idx]) + 0.5
        test_image = 0.5 * np.squeeze(test_images) + 0.5

        for i in range(sample_num):
            fig, axs = plt.subplots(2 + len(iteration_list), 5, num=i, figsize=(10 ,10))
            for j, itr in enumerate(iteration_list):
                for m, modal in enumerate(modal_lists):
                    # org image and test image(applied non avalible masks)
                    axs[0, m].set_title(modal)
                    axs[0, m].imshow(org_image[i, :, :, m], cmap="gray", norm=NoNorm())
                    axs[0, m].axis('off')
                    axs[1, m].imshow(test_image[i, :, :, m], cmap="gray", norm=NoNorm())
                    axs[1, m].axis('off')

                    # plot generated results in each saved weight iteration
                    axs[j+2, m].imshow(all_gen_data[j, i, :, :, m], cmap="gray", norm=NoNorm())
                    axs[j+2, m].set_xlabel(str(round(ssim(org_image[i, :, :, m], all_gen_data[j, i, :, :, m]), 4)),
                                           fontsize=10)
                    axs[j+2, m].set_ylabel(str(itr), fontsize=10)
                    axs[j+2, m].set_xticks([])
                    axs[j+2, m].set_yticks([])
                    # axs[j+2, m].axis('off')

            fig.savefig(GAN_PATH + os.sep + EXPORT_FOLDER + '/samples_results' + str(i) + '.png')
            plt.close()

        # plt.show()


if __name__ == '__main__':
    from pandas import DataFrame


    def reduced_channel(images, ch_num):
        reduced_ch_images = np.zeros((len(images), images.shape[1], images.shape[2], ch_num), np.float32)

        for ch in range(ch_num):
            ip_ch = ch * 3
            reduced_ch_images[:, :, :, ch] = images[:, :, :, ip_ch]

        return reduced_ch_images


    print(tf.__version__)

    # load data
    modal_lists = ['pc', 'ec', 'dc', 'tm', 'am']
    x_data, labels = load_data_from_xlsx(PATH + DATA_FOLDER,
                                         modal_lists=modal_lists,
                                         cropped=True,
                                         image_type=1,
                                         image_size=IMAGE_SIZE)
    x_data = reduced_channel(x_data, 5)
    print(x_data.shape)
    print(labels.shape)

    # train/test split (sc)
    train_samples, label_train, test_samples, label_test = split_train_test(x_data, labels, val_ratio=0.2,
                                                                            random_sample=False)

    try:
        os.makedirs(GAN_PATH + os.sep + EXPORT_FOLDER)
        os.mkdir(GAN_PATH + os.sep + EXPORT_FOLDER + os.sep + 'samples')
        os.mkdir(GAN_PATH + os.sep + EXPORT_FOLDER + os.sep + 'saved_model')
        os.mkdir(GAN_PATH + os.sep + EXPORT_FOLDER + os.sep + 'test_results')
    except FileExistsError:
        print('folder exist')

    modal_generate = Pix2Pix()

    # trainining GAN
    number_of_masks = [0, 1, 2, 3]
    modal_generate.train(samples=[train_samples, label_train],
                         epochs=STEP,
                         batch_size=BATCH_SIZE,
                         sample_interval=SAMPLE_INTERVAL,
                         step_masks=False)

    itr_list = [3000, 6000, 9000, 12000, 15000]

    # generate samples test results
    modal_generate.compare_test_samples(samples=test_samples, iteration_list=itr_list, sample_num=10)

    for saved_itr in itr_list:

        iqa_results = {'modal': [], 'available': [], 'mse': [], 'psnr': [], 'ssim': []}
        iqa_summary = {'available': [], 'mse': [], 'psnr': [], 'ssim': []}


        # testing GAN
        mse, psnr, ssim, modal_label, available_label = modal_generate.predict(samples=test_samples,
                                                                               saved_itr=saved_itr)

        mse_non_mask = np.zeros((900, 1))
        mse_mask = np.zeros((600, 1))
        psnr_non_mask = np.zeros((900, 1))
        psnr_mask = np.zeros((600, 1))
        ssim_non_mask = np.zeros((900, 1))
        ssim_mask = np.zeros((600, 1))

        # mse_non_mask = np.zeros((600, 1))
        # mse_mask = np.zeros((150, 1))
        # psnr_non_mask = np.zeros((600, 1))
        # psnr_mask = np.zeros((150, 1))
        # ssim_non_mask = np.zeros((600, 1))
        # ssim_mask = np.zeros((150, 1))

        n = 0
        m = 0
        for i in range(len(mse)):
            iqa_results['modal'].append(modal_label[i])
            iqa_results['available'].append(available_label[i])
            iqa_results['mse'].append(mse[i])
            iqa_results['psnr'].append(psnr[i])
            iqa_results['ssim'].append(ssim[i])

            if available_label[i] == 1:
                mse_non_mask[n] = mse[i]
                psnr_non_mask[n] = psnr[i]
                ssim_non_mask[n] = ssim[i]
                n += 1
            elif available_label[i] == 0:
                mse_mask[m] = mse[i]
                psnr_mask[m] = psnr[i]
                ssim_mask[m] = ssim[i]
                m += 1

                # mean
        mean_ssim = np.mean(ssim)
        mean_mse = np.mean(mse)
        mean_psnr = np.mean(psnr)
        iqa_results['modal'].append('')
        iqa_results['available'].append('')
        iqa_results['mse'].append(mean_mse)
        iqa_results['psnr'].append(mean_psnr)
        iqa_results['ssim'].append(mean_ssim)

        # saved_data = DataFrame(iqa_results, columns=['modal', 'available', 'mse', 'psnr', 'ssim'])
        # saved_data.to_excel(GAN_PATH + os.sep + EXPORT_FOLDER + '/iqa_results.xlsx', index=None, header=True)

        iqa_summary['available'].append('non-avalable')
        iqa_summary['available'].append('available')
        iqa_summary['mse'].append(np.mean(mse_mask))
        iqa_summary['mse'].append(np.mean(mse_non_mask))
        iqa_summary['psnr'].append(np.mean(psnr_mask))
        iqa_summary['psnr'].append(np.mean(psnr_non_mask))
        iqa_summary['ssim'].append(np.mean(ssim_mask))
        iqa_summary['ssim'].append(np.mean(ssim_non_mask))

        saved_data = DataFrame(iqa_summary, columns=['available', 'mse', 'psnr', 'ssim'])
        saved_data.to_excel(GAN_PATH + os.sep + EXPORT_FOLDER + '/iqa_summary' + str(saved_itr) + '.xlsx', index=None,
                            header=True)