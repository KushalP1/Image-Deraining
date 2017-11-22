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
       return 20*K.log(255.0/K.sqrt(MSE))

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

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy',psnr])


        model.fit(d['rainy'],d['original'],  epochs=epochs, batch_size=batch_size,validation_split=0.3, callbacks=[self.mc])

        ii=[]
        ii.append(scipy.misc.imread("./training/1.jpg"))

        #ii=np.asarray(ii)
        #ii=ii[:512. :512, :]
        #ii=cv2.resize(ii,(512,512))
        di=model.predict(np.reshape(self.rainy[2],[1,256,256,3]))
        cv2.imshow('', np.reshape(di,[256,256,3]))
        cv2.waitKey(0)
        cv2.imshow('', np.reshape(self.rainy[2],[256,256,3]))
        cv2.waitKey(0)
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
'''
    def save_model(self, step):
        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """


    def load_model(self, **params):
    	# file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-oried instance of Segment class
        """

    def graph(self):

    	with tf.device('/gpu:1'):

            	self.x = tf.placeholder(tf.float32, shape=(1,None, None, 3))
	    	self.flat_labels = tf.placeholder(tf.int64, shape=(1,None,2))
	    	labels = tf.reshape(self.flat_labels, (-1, 2))

	    	#conv layer 1
	    	W_shape = [2, 2, 3, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_1 =  tf.nn.relu(tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		#conv layer 2
	    	W_shape = [3, 3, 64, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_2 =  tf.nn.relu(tf.nn.conv2d(conv_1, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		# pool layer 1
		pool_1, pool_1_argmax = tf.nn.max_pool_with_argmax(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		#conv layer 3
	    	W_shape = [3, 3, 64, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_3 =  tf.nn.relu(tf.nn.conv2d(pool_1, W, strides=[1, 1, 1, 1], padding='SAME') + b)

	    	#conv layer 4
	    	W_shape = [3, 3, 128, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_4 =  tf.nn.relu(tf.nn.conv2d(conv_3, W, strides=[1, 1, 1, 1], padding='SAME') + b)


		# pool layer 2
		pool_2, pool_2_argmax = tf.nn.max_pool_with_argmax(conv_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


	    	#conv layer 5
	    	W_shape = [3, 3, 128, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_5 =  tf.nn.relu(tf.nn.conv2d(pool_2, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		#conv layer 6
	    	W_shape = [3, 3, 256, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_6 =  tf.nn.relu(tf.nn.conv2d(conv_5, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		pool_3, pool_3_argmax = tf.nn.max_pool_with_argmax(conv_6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		dropout = tf.nn.dropout(pool_3, keep_prob = 0.5)

		#unpool 3
		padding = 'SAME'
		unpool_3 = self.unpool(dropout, pool_3_argmax, tf.shape(conv_6))


		    #deconv layer 6
	    	W_shape = [3, 3, 256, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = unpool_3
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
	        deconv_6 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


            	#deconv layer 5
	    	W_shape = [3, 3, 128, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_6
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_5 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	unpool_2 = self.unpool(deconv_5, pool_2_argmax, tf.shape(conv_4))

            	#deconv layer 4
	    	W_shape = [3, 3, 128, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = unpool_2
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_4 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


            	#deconv layer 3
	    	W_shape = [3, 3, 64, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_4
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_3 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	unpool_1 = self.unpool(deconv_3, pool_1_argmax, tf.shape(conv_2))

            	#deconv layer 2
            	W_shape = [3, 3, 64, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = unpool_1
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_2 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	#deconv layer 1
            	W_shape = [3, 3, 32, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_2
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_1 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


            	#deconv layer 0
            	W_shape = [1, 1, 2, 32]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_1
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	score_1 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	flat_logits = tf.reshape(score_1, (-1, 2))
            	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = flat_logits, labels = labels, name='x_entropy')
            	self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

            	self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

            	self.prediction = tf.argmax(flat_logits,1)


    def unpool(self, x, argmax, out_shape):
	shape = tf.to_int64(out_shape)
        argmax = tf.pack([argmax // (shape[2] * shape[3]), argmax % (shape[2] * shape[3]) // shape[3]])
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])
        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]
        indices1 = tf.squeeze(argmax)
        indices1 = tf.transpose(tf.pack((indices1[0], indices1[1]), axis=0), perm=[3, 1, 2, 0])
        indices2 = tf.tile(tf.to_int64(tf.range(channels)), [((width + 1) // 2) * ((height + 1) // 2)])
        indices2 = tf.transpose(tf.reshape(indices2, [-1, channels]), perm=[1, 0])
        indices2 = tf.reshape(indices2, [channels, (height + 1) // 2, (width + 1) // 2, 1])
        indices = tf.reshape(tf.concat(3,  [indices1, indices2]), [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])
        inter = tf.reshape(tf.squeeze(x), [-1, channels])
        values = tf.reshape(tf.transpose(inter, perm=[1, 0]), [-1])
        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)


if __name__ == "__main__":

        obj = Derain('./Training')
	# training the model
	X = range(0,164)
        k_fold = KFold(n_splits=5)
	i = 0
	for train_indices, test_indices in k_fold.split(X):
		print( '{} validation'.format(i))
		i += 1
    		print('Train: %s | test: %s' % (train_indices, test_indices))
		obj.train(train_indices)
		print 'Accuracy = {:.3f}%'.format(obj.test(test_indices) * 100)

       	# testing the model
	T = range(0,5)
	print( 'Accuracy = {:.3f}%'.format(obj.test(T) * 100))


'''
p = Derain("./training", "./checkpoints")
p.train()
