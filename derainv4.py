import numpy as np
import cv2
import keras
import os
import scipy
#import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Input, Dense ,Dropout
from keras.layers import Conv2D, Dense ,Conv2DTranspose ,Flatten ,Reshape
import keras.backend as K


DEBUG = True
PRINT = False

def psnr(base_image,altered_image):
    MSE=K.mean(K.square(base_image-altered_image))
    if(MSE==0):
       return 100
    else:
       return (20*K.log(255.0/K.sqrt(MSE)))/K.log(10.0)


class Derain(object):

    def __init__(self, folder,checkpoint_dir="./checkpoints"):
        self.images = []
        self.dim = []
        self.ori = []
        self.rainy = []

        self.mc=keras.callbacks.ModelCheckpoint(checkpoint_dir + "/" + "weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.n = 0;

        # Read all images
        for filename in os.listdir(folder):
            if (self.n == 700) & DEBUG:
                break

            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                self.images.append(img)
                self.dim.append(list(img.shape))
                self.dim[self.n][1] = self.dim[self.n][1]/2
                self.ori.append(img[:, 0:(self.dim[self.n][1] - 1), :])
                self.rainy.append(img[:, self.dim[self.n][1]:(self.dim[self.n][1]*2), :])
            self.n = self.n + 1

        self.x = self.dim[0][0]
        self.y = self.dim[0][1]

        # Find the largest reshapable size
        for i in range(self.n):
            self.x = min(self.x, self.dim[i][0])
            self.y = min(self.y, self.dim[i][1])

        # Reshape all images
        if DEBUG:
            self.x = 40
            self.y = 100

        self.x = 256
        self.y = 256

        for i in range(self.n):
            self.ori[i] = cv2.resize(self.ori[i], (self.y, self.x))
            self.rainy[i] = cv2.resize(self.rainy[i], (self.y, self.x))

    def loadmodel(self):
         #model = Sequential()
    	 file_name = "weights.40-678.80.hdf5"
    	 #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy',psnr])
         model1=keras.models.load_model(file_name,custom_objects = {'psnr':psnr});
         d={'rainy':np.asarray(self.rainy),'original':np.asarray(self.ori)}
         print model1.evaluate(d['rainy'].reshape(-1,256,256,3),d['original'].reshape(-1,256,256,3),batch_size = 1)
         #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy',psnr])
         for i in range(20):
            di=model1.predict(np.reshape(self.rainy[i],[1,256,256,3]))
            dd=np.reshape(di,[256,256,3])
            cv2.imwrite("./outputs/"+str(i)+"predict.jpg",dd)




    def train(self, training_steps=10):

        img_rows, img_cols, img_chns = 512, 512, 3
        filters = 64
        latent_dim = 2
        intermediate_dim = 128
        epsilon_std = 1.0
        epochs = 30
        batch_size = 10
        num_conv=3
        filtersize=2
        filtersize1=3
        '''

        step_start = 0
        for i in range(step_start, step_start+training_steps):
            # pick random line from file
            j = choice(train_indices)
            image = self.data_dir + 'train-' + str(j) + '.jpg'
            mask_image = self.data_dir + 'train-' + str(j) + '-mask.jpg'
            [image_tensor, mask_tensor, flat_labels, feed_dict_to_use]=self.get_image(image, mask_image)
            train_image, train_mask, flat_labels = self.session.run([image_tensor, mask_tensor, flat_labels],feed_dict=feed_dict_to_use)
            print('run train step: '+str(i))
            start = time.time()
            feed_dict_to_use = {self.x: [train_image], self.flat_labels:[flat_labels]}
            self.train_step.run(session=self.session, feed_dict=feed_dict_to_use)
            print('step {} finished in {:.2f} s with loss of {:.6f} '.format(i, time.time() - start, self.loss.eval(session=self.session, feed_dict=feed_dict_to_use)) )
            self.save_model(i)
            actual = tf.argmax(flat_labels,1)
            print( 'IOU : {:.6f} '.format( self.iou(self.prediction.eval(session=self.session, feed_dict=feed_dict_to_use), actual.eval(session = self.session))))
            print('Model {} saved'.format(i))

        '''
        d={'rainy':np.asarray(self.rainy),'original':np.asarray(self.ori)}
        model= Sequential()


        input_shape=(256,256,3)
        model.add(keras.layers.InputLayer(input_shape=input_shape))

        model.add(keras.layers.convolutional.Conv2D(64, filtersize, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        #model.add(Dropout(0.5))

        model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.1, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

        model.add(keras.layers.convolutional.Conv2D(128, filtersize, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

        model.add(keras.layers.convolutional.Conv2D(256, 3, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))



        model.add(keras.layers.convolutional.Conv2D(512, filtersize, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(Dropout(0.5))

        #model.add(keras.layers.convolutional.Conv2D(512,2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))





        # model.add(keras.layers.convolutional.Conv2D(128, 5, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))


        model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.1, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))


        #model.add(Conv2DTranspose(16, 5, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(keras.layers.UpSampling2D(size=(2, 2), data_format=None))

        #model.add(Conv2DTranspose(512, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(Conv2DTranspose(512, filtersize, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))
        model.add(keras.layers.UpSampling2D(size=(2, 2), data_format=None))


        model.add(Conv2DTranspose(256, 3, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(keras.layers.UpSampling2D(size=(2, 2), data_format=None))

        model.add(Conv2DTranspose(128, filtersize, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.add(Conv2DTranspose(3, filtersize, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        model.fit(d['rainy'],d['original'],  epochs=epochs, batch_size=batch_size,validation_split=0.3, callbacks=[self.mc])

        ii=[]
        ii.append(scipy.misc.imread("./training/1.jpg"))

        #ii=np.asarray(ii)
        #ii=ii[:512. :512, :]
        #ii=cv2.resize(ii,(512,512))
        
        
        
        # plt.imshow(np.reshape(self.rainy[0],[512,512,3]))

        '''plt.imshow(self.ori[0])
        plt.show()
        model.summary()
        '''

        '''
        mm=Model(self.rainy,self.ori)
        mm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        '''

        '''original_img_size = (img_rows, img_cols, img_chns)
        x = Input(shape=original_img_size)
        conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
        conv_2 = Conv2D(filters,
		        kernel_size=(2, 2),
		        padding='same', activation='relu',
		        strides=(2, 2))(conv_1)
        conv_3 = Conv2D(filters,
		        kernel_size=num_conv,
		        padding='same', activation='relu',
		        strides=1)(conv_2)
        conv_4 = Conv2D(filters,
		        kernel_size=num_conv,
		        padding='same', activation='relu',
		        strides=1)(conv_3)
        flat = Flatten()(conv_4)
        hidden = Dense(intermediate_dim, activation='relu')(flat)
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(filters * 14 * 14, activation='relu')
        output_shape = (batch_size, 14, 14, filters)
        decoder_upsample = Dense(filters * 14 * 14, activation='relu')
        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Conv2DTranspose(filters,
       	                           kernel_size=num_conv,
		                           padding='same',
		                           strides=1,
		                           activation='relu')
        decoder_deconv_2 = Conv2DTranspose(filters,
		                           kernel_size=num_conv,
		                           padding='same',
		                           strides=1,
	                                   activation='relu')
        output_shape = (batch_size, 29, 29, filters)
        decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
        decoder_mean_squash = Conv2D(img_chns,
		                     kernel_size=2,
		                     padding='valid',
		                     activation='sigmoid')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)'''

    def save_model(self, step):    #train function saves the model too
        
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """



    	# file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        
    

p = Derain("./test", "./checkpoints")
p.loadmodel()
