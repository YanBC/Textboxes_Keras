################ densenet ################
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Reshape, Permute
# from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers import Conv2DTranspose, ZeroPadding2D
from keras.layers import SeparableConv2D as Conv2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten, Concatenate
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.layers import TimeDistributed


def conv_block(input_t, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input_t)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3,3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(weight_decay))(x)
    if(dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input_t, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input_t)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    if(dropout_rate):
        x = Dropout(dropout_rate)(x)
    
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    
    # if(pooltype == 2):
    #     x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    # elif(pooltype == 1):
    #     x = ZeroPadding2D(padding = (0, 1))(x)
    #     x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    # elif(pooltype == 3):
    #     x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter



def densenet_model(img_h, img_w):
    _dropout_rate=0.2
    # _weight_decay=1e-4
    _weight_decay = 0.0005
    _nb_filter=64

    input_t = Input(shape=(img_h, img_w, 3))
    x = Conv2D(_nb_filter, (7, 7), strides=(2, 2), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(_weight_decay))(input_t)
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, _weight_decay)

    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, _weight_decay) 

    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, _weight_decay) 

    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    out_conf_1 = Conv2D(2 * 6, (1,5), padding='same')(x)
    out_loc_1 = Conv2D(4 * 6, (1,5), padding='same')(x)
    x, _nb_filter = transition_block(x, 128, _dropout_rate, _weight_decay)

    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    out_conf_2 = Conv2D(2 * 6, (1,5), padding='same')(x)
    out_loc_2 = Conv2D(4 * 6, (1,5), padding='same')(x)


    # ssd outputs
    out_conf_1 = Reshape((-1, 2))(out_conf_1)
    out_loc_1 = Reshape((-1, 4))(out_loc_1)
    out_conf_2 = Reshape((-1, 2))(out_conf_2)
    out_loc_2 = Reshape((-1, 4))(out_loc_2)

    pred_conf = Concatenate(axis=1)([out_conf_1, out_conf_2])
    pred_loc = Concatenate(axis=1)([out_loc_1, out_loc_2])

    pred_conf_softmax = Activation('softmax')(pred_conf)
    pred = Concatenate(axis=2)([pred_conf_softmax, pred_loc])

    m = Model(inputs=[input_t], outputs=[pred])
    # m.summary(line_length=120)

    return m


if __name__ == '__main__':
    model = densenet_model(512, 512)

    model.summary(line_length=120)


################ densenet ################


