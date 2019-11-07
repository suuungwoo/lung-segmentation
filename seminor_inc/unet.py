#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, MaxPooling2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.utils.vis_utils import plot_model
import keras.backend as K

def getUnet(imgW, imgH, channel):
    firstUnitCnt = 64

    inputs = Input((imgH, imgW, channel))

    encLayerList = decLayerList = []
    filterCntList = []

    enc = Conv2D(firstUnitCnt, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    encLayerList.append(enc)
    filterCntList.append((firstUnitCnt))

    outW = imgW
    outH = imgH
    unitCnt = firstUnitCnt

    # Encoder
    while True:
        outW = int(outW / 2)
        outH = int(outH / 2)
        if (outW <= 4 or outH <= 4):
            break
        if (1 == outW % 2 or 1 == outH % 2):
            break
        if (unitCnt <= 512):
            unitCnt *= 2
        enc = Conv2D(unitCnt, (3, 3), padding='same', strides=1)(enc)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(0.2)(enc)
        enc = MaxPooling2D((2, 2))(enc)
        encLayerList.append(enc)
        filterCntList.append(unitCnt)
    encLayerNum = len(encLayerList)

    # Decoder
    dec = enc
    for i in range(encLayerNum - 1):
        dec = LeakyReLU(0.2)(dec)
        dec = Conv2DTranspose(filterCntList[encLayerNum - i - 1], 2, strides=2,
                              kernel_initializer='he_uniform')(dec)
        dec = BatchNormalization()(dec)
        dec = Dropout(0.5)(dec)
        dec = concatenate([dec, encLayerList[encLayerNum - i - 2]], axis=-1)

    # Output
    dec = LeakyReLU(0.2)(dec)
    dec = Conv2DTranspose(channel, 2, strides=2)(dec)
    dec = Activation(activation='sigmoid')(dec)

    UNet = Model(inputs=inputs, outputs=dec)
    return UNet

