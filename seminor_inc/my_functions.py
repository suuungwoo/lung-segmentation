import numpy as np
import keras.backend as K

def readRaw(raw_path, dataType=np.int16):
    fRaw = open(raw_path, "rb")
    data = np.fromfile(fRaw, dtype=dataType)
    fRaw.close()
    return data

def getSliceUint8(slice, wl, ww):
    w_min = wl - int(ww / 2)
    w_max = w_min + ww
    slice = np.clip(slice.copy(), w_min, w_max)
    minGray = slice.min()
    slice = slice - minGray
    slice = 255.0 * slice / (w_max - w_min)
    return slice

def getSliceByLut(slice, wl=0, ww=200, width=-1, height=-1):
    if slice.size != width * height:
        width = slice.shape[0]
        height = slice.shape[1]

    slice = getSliceUint8(slice, wl, ww)
    gray = slice.reshape((width, height)).astype("uint8")

    return gray


