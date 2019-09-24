
################ conv_model ################
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Reshape, Concatenate
from keras.activations import relu
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
from keras import backend as K


def ConvBR(x, filter, kernel=3):
    x = Conv2D(filter, (kernel, kernel), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def ConvBR_Pool(x, filter, kernel=3):
    x = Conv2D(filter, (kernel, kernel), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    return x


def conv_model(img_h, img_w):
    myIn = Input(shape=(img_h, img_w, 3))

    myOut = ConvBR(myIn, 16, 3)
    myOut = ConvBR(myOut, 16, 3)
    myOut = ConvBR_Pool(myOut, 16, 3)

    myOut = ConvBR(myOut, 32, 3)
    myOut = ConvBR(myOut, 32, 3)
    myOut = ConvBR(myOut, 32, 3)
    myOut = ConvBR(myOut, 32, 3)
    myOut = ConvBR_Pool(myOut, 32, 3)

    myOut = ConvBR(myOut, 64, 3)
    myOut = ConvBR(myOut, 64, 3)
    myOut = ConvBR(myOut, 64, 3)
    myOut = ConvBR(myOut, 64, 3)
    myOut = ConvBR_Pool(myOut, 64, 3)

    myOut = ConvBR(myOut, 128, 3)
    myOut = ConvBR(myOut, 128, 3)
    myOut = ConvBR(myOut, 128, 3)
    myOut = ConvBR(myOut, 128, 3)
    myOut = ConvBR(myOut, 128, 3)
    myOut = ConvBR(myOut, 128, 3)
    myOut = ConvBR_Pool(myOut, 128, 3)

    myOut = ConvBR(myOut, 256, 3)
    myOut = ConvBR(myOut, 256, 3)
    myOut = ConvBR(myOut, 256, 3)

    out_conf_1 = Conv2D(2 * 6, (1,5), padding='same')(myOut)
    out_loc_1 = Conv2D(4 * 6, (1,5), padding='same')(myOut)

    myOut = ConvBR(myOut, 256, 3)
    myOut = ConvBR(myOut, 256, 3)
    myOut = ConvBR_Pool(myOut, 256, 3)

    myOut = ConvBR(myOut, 256, 3)
    myOut = ConvBR(myOut, 256, 3)
    out_conf_2 = Conv2D(2 * 6, (1,5), padding='same')(myOut)
    out_loc_2 = Conv2D(4 * 6, (1,5), padding='same')(myOut)

    # ssd outputs
    out_conf_1 = Reshape((-1, 2))(out_conf_1)
    out_loc_1 = Reshape((-1, 4))(out_loc_1)
    out_conf_2 = Reshape((-1, 2))(out_conf_2)
    out_loc_2 = Reshape((-1, 4))(out_loc_2)

    pred_conf = Concatenate(axis=1)([out_conf_1, out_conf_2])
    pred_loc = Concatenate(axis=1)([out_loc_1, out_loc_2])

    pred_conf_softmax = Activation('softmax')(pred_conf)
    pred = Concatenate(axis=2)([pred_conf_softmax, pred_loc])

    # model
    model = Model(inputs=[myIn], outputs=[pred])

    return model 



if __name__ == '__main__':
    m = conv_model(512,512)

    m.summary()
################ conv_model ################



