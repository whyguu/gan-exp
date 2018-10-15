# coding: utf-8
import tensorflow as tf
import numpy as np
import datetime
import os
import glob
import argparse


def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU id

    # config
    log_device_placement = False  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)

    return config


class NetBlocks(object):
    def __init__(self):
        # data should have sufficient shape(b, h, w, c) when using "trans" functions
        pass

    @staticmethod
    def conv2d(inputs, filters, kernel_size=(3, 3), strides=(1, 1), use_bn=True,
               padding='same', activation=tf.nn.relu, training=True, name=None):
        with tf.name_scope(name=name):
            out = inputs
            if use_bn:
                out = tf.layers.batch_normalization(out, training=training, name=name + '/bn')
            if activation is not None:
                out = activation(out, name=name + '/act')

            out = tf.layers.conv2d(out, filters, kernel_size, strides, padding,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name=name + '/conv')
            return out

    @staticmethod
    def conv2d_trans(inputs, filter_num, kernel_size, strides, out_shape, use_bn=True, activation=tf.nn.relu, training=True, name=None):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name=name+'/kernel', shape=kernel_size+(filter_num, inputs.shape[-1].value),
                                     dtype=inputs.dtype, initializer=tf.truncated_normal_initializer(stddev=0.1),)
            output_shape = (inputs.shape[0].value, ) + out_shape + (filter_num,)

            out = tf.nn.conv2d_transpose(inputs, kernel, output_shape, (1,)+strides+(1,), name=name+'/trans')
            # batch norm
            if use_bn:
                out = tf.layers.batch_normalization(out, training=training, name=name+'/bn')
            if activation is not None:
                out = tf.nn.relu(out, name=name+'/act')
            return out

    @staticmethod
    def residual2d(inputs, filters=(2, 4), same_size=True, same_channel=True, training=True, use_bn=True, name=None):
        strides = (1, 1) if same_size else (2, 2)
        with tf.name_scope(name):
            conv1 = NetBlocks.conv2d(inputs, filters[0], 1, (1, 1), use_bn=use_bn, training=training, name=name + '/conv1')
            conv2 = NetBlocks.conv2d(conv1, filters[0], 3, strides, use_bn=use_bn, training=training, name=name + '/conv2')
            conv3 = NetBlocks.conv2d(conv2, filters[1], 1, (1, 1), use_bn=use_bn, training=training, name=name + '/conv3')

            if not same_size:
                inputs = tf.layers.conv2d(inputs, filters[-1], (1, 1), strides,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          name=name + '/res')
                return inputs + conv3
            if not same_channel:
                inputs = tf.layers.conv2d(inputs, filters[-1], (1, 1), (1, 1),
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          name=name + '/res')
                return inputs + conv3
            return inputs + conv3

    @staticmethod
    def conv3d(inputs, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), use_bn=True, use_bias=True,
               padding='same', activation=tf.nn.relu, training=True, pre_act=False, name=None):
        out = inputs

        def act(ipt):
            act_out = ipt
            if use_bn:
                act_out = tf.layers.batch_normalization(ipt, training=training, name=name + '/bn')
            if activation is not None:
                act_out = tf.nn.relu(act_out, name=name + '/act')
            return act_out

        with tf.name_scope(name=name):
            if pre_act:
                out = act(inputs)

            out = tf.layers.conv3d(out, filters, kernel_size, strides, padding, use_bias=use_bias,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   name=name + '/conv')
            if not pre_act:
                out = act(out)
            return out

    @staticmethod
    def conv3d_trans(inputs, filter_num, kernel_size, strides, out_shape, use_bn=True, activation=tf.nn.relu,
                     pre_act=False, training=True, name=None):
        with tf.variable_scope(name):
            out = inputs

            def act(ipt):
                act_out = ipt
                # batch norm
                if use_bn:
                    act_out = tf.layers.batch_normalization(act_out, training=training, name=name + '/bn')
                if activation is not None:
                    act_out = tf.nn.relu(act_out, name=name + '/act')
                return act_out

            kernel = tf.get_variable(name=name+'/kernel', shape=kernel_size+(filter_num, inputs.shape[-1].value),
                                     dtype=inputs.dtype, initializer=tf.truncated_normal_initializer(stddev=0.1),)
            output_shape = (inputs.shape[0].value, ) + out_shape + (filter_num,)

            if pre_act:
                out = act(out)

            out = tf.nn.conv3d_transpose(out, kernel, output_shape, (1,)+strides+(1,), name=name + '/conv_trans')

            if not pre_act:
                out = act(out)
            return out

    @staticmethod
    def residual3d(inputs, filters=(2, 4), same_size=True, same_channel=True, training=True, stride=None, name=None):
        if same_size:
            strides = (1, 1, 1)
        elif stride is not None:
            strides = stride
        else:
            strides = (2, 2, 2)
        # strides = (1, 1, 1) if same_size else (2, 2, 2)
        with tf.name_scope(name):
            conv1 = NetBlocks.conv3d(inputs, filters[0], 1, (1, 1, 1), training, pre_act=True, name=name + '/conv1')
            conv2 = NetBlocks.conv3d(conv1, filters[0], 3, strides, training, pre_act=True, name=name + '/conv2')
            conv3 = NetBlocks.conv3d(conv2, filters[1], 1, (1, 1, 1), training, pre_act=True, name=name + '/conv3')

            if not same_size:
                inputs = tf.layers.conv3d(inputs, filters[-1], (1, 1, 1), strides,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          name=name + '/res')
                return inputs + conv3
            if not same_channel:
                inputs = tf.layers.conv3d(inputs, filters[-1], 1, 1,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          name=name + '/res')
                return inputs + conv3
            return inputs + conv3

    @staticmethod
    def up_concat3d(up, cat, filters, kernel_size=(5, 5, 5), strides=(2, 2, 2), training=True, name=None):
        with tf.name_scope(name=name):
            b, d, h, w, c = cat.shape
            ts = NetBlocks.conv3d_trans(up, filters, kernel_size, strides,
                                        (d.value, h.value, w.value), training, pre_act=True, name=name+'/trans')
            cat = tf.concat((cat, ts), axis=-1, name=name+'/concat')

            return cat

    @staticmethod
    def se3d(inputs, name='se'):
        with tf.name_scope(name):
            gb = tf.reduce_mean(inputs, axis=(2, 3), keepdims=True, name=name+'_mean')
            se1 = tf.layers.conv3d(gb, filters=inputs.shape[4].value / 4, kernel_size=1, activation=tf.nn.relu,
                                   use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            se2 = tf.layers.conv3d(se1, filters=inputs.shape[4].value, kernel_size=1, activation=tf.nn.sigmoid,
                                   use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

            return se2 * inputs


class ModelWGANGP(object):
    def __init__(self, height, width, channel_out):
        self.alpha = 0.5  # ratio of gram loss
        self.beta = 1500  # ratio MAE/MSE loss
        self.height = height
        self.width = width
        self.channel_out = channel_out

        # self.clip_range = 0.1
        self.gp_lambda = 0.1
        #
        self.data = None
        self.label = None

        self.g_out = None
        self.d_real = None
        self.real_feat_list = None
        self.d_fake = None
        self.fake_feat_list = None
        self.d_loss = None
        self.g_loss = None

        self.grad_penalty = None

    @staticmethod
    def calc_gram(feat_list):
        ls = []
        for feat in feat_list:
            b, h, w, c = feat.shape
            # ts = tf.transpose(feat, perm=[0, 2, 3, 1, 4])
            ts = tf.reshape(feat, shape=(b.value, h.value * w.value, c.value))
            ts = tf.matmul(ts, ts, transpose_a=True)
            ts = tf.reduce_mean(ts, axis=0) / (h.value * w.value * c.value)
            print('ts: ', ts)
            ls.append(ts)
        return ls

    def generator(self, inputs, reuse=False, variable_scope_name='generator', training=True):
        with tf.variable_scope(name_or_scope=variable_scope_name, reuse=reuse):
            with tf.name_scope(name='conv'):
                # input: 31*501*501*1
                g_conv1 = NetBlocks.conv3d(inputs, 8, (5, 7, 7), (2, 2, 2), training=training, name='conv3d_1')
                mp = tf.layers.max_pooling3d(g_conv1, pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')
                # input: 16*251*251*8
                g_conv2 = NetBlocks.residual3d(mp, [4, 16], same_channel=False, training=training, name='res_1')
                # input: 8*126*126*16
                g_conv3 = NetBlocks.residual3d(g_conv2, [4, 16], same_size=True, training=training, name='res_2')
                # input: 8*126*126*16
                g_conv4 = NetBlocks.residual3d(g_conv3, [8, 32], same_size=False, training=training, name='res_3')
                # input: 4*63*63*32
                g_conv5 = NetBlocks.residual3d(g_conv4, [8, 32], same_size=True, training=training, name='res_4')
                # input: 4*63*63*32
                # g_conv6 = NetBlocks.conv3d(g_conv5, 64, (3, 3, 3), (1, 2, 2), training=training, name='conv3d_2')
                g_conv6 = NetBlocks.residual3d(g_conv5, [16, 64], same_size=False, stride=(1, 2, 2), training=training, name='res_5')
                # input: 4*32*32*64
                # g_conv7 = NetBlocks.conv3d(g_conv6, 64, (1, 1, 1), (1, 1, 1), training=training, name='conv3d_3')
                g_conv7 = NetBlocks.residual3d(g_conv6, [16, 64], same_size=True, training=training, name='res_6')
                # input: 4*32*32*64
                print('g_conv1: ', g_conv1)
                print('g_conv2: ', g_conv2)
                print('g_conv3: ', g_conv3)
                print('g_conv4: ', g_conv4)
                print('g_conv5: ', g_conv5)
                print('g_conv6: ', g_conv6)
                print('g_conv7: ', g_conv7)

            with tf.name_scope(name='conv_trans'):
                g_trans4 = NetBlocks.up_concat3d(g_conv7, g_conv5, 64, (3, 3, 3), (1, 2, 2), name='trans4')
                g_trans4 = NetBlocks.residual3d(g_trans4, [16, 64], same_channel=False, training=training, name='trans4_res1')
                # g_trans4 = NetBlocks.residual3d(g_trans4, [8, 32], training=training, name='trans4_res2')
                # g_trans4 = NetBlocks.conv3d(g_trans4, 32, (3, 3, 3), training=training, name='trans4_conv1')
                # 4*63*63*32
                g_trans3 = NetBlocks.up_concat3d(g_trans4, g_conv3, 16, (3, 3, 3), (2, 2, 2), name='trans3')
                g_trans3 = NetBlocks.residual3d(g_trans3, [4, 16], same_channel=False, training=training, name='trans3_res1')
                # g_trans3 = NetBlocks.residual3d(g_trans3, [4, 16], training=training, name='trans3_res2')
                # g_trans3 = NetBlocks.conv3d(g_trans3, 16, (3, 3, 3), training=training, name='trans3_conv1')
                # 8*126*126*16
                g_trans2 = NetBlocks.up_concat3d(g_trans3, g_conv1, 8, (3, 3, 3), (2, 2, 2), name='trans2')
                g_trans2 = NetBlocks.residual3d(g_trans2, [4, 8], same_channel=False, training=training, name='trans2_res1')
                # g_trans2 = NetBlocks.residual3d(g_trans2, [4, 8], training=training, name='trans2_res2')
                # g_trans2 = NetBlocks.conv3d(g_trans2, 8, (3, 3, 3), training=training, name='trans2_conv1')
                # 16*251*251*8

                g_trans1 = NetBlocks.conv3d_trans(g_trans2, 8, (3, 3, 3), (1, 2, 2),
                                                  (g_trans2.shape[1].value, self.height, self.width),
                                                  pre_act=True, name='trans1')
                g_trans1 = NetBlocks.residual3d(g_trans1, [4, 8], same_size=True, training=training, name='trans1_res1')
                g_trans1 = NetBlocks.residual3d(g_trans1, [4, 8], same_size=True, training=training, name='trans1_res2')

                print('g_conv_trans4: ', g_trans4)
                print('g_conv_trans3: ', g_trans3)
                print('g_conv_trans2: ', g_trans2)
                print('g_conv_trans1: ', g_trans1)

                b, d, h, w, c = g_trans1.shape
                ts = tf.transpose(g_trans1, perm=(0, 2, 3, 1, 4))
                g_trans0 = tf.reshape(ts, shape=(b.value, h.value, w.value, d.value*c.value))
                out = NetBlocks.conv2d(g_trans0, 32, (3, 3), name='out1')
                out = NetBlocks.conv2d(out, self.channel_out, (1, 1), activation=None, use_bn=False, name='out2')
                # out = tf.nn.relu(out)
                out = tf.nn.sigmoid(out)
                print('g_out: ', out)

            return out

    @staticmethod
    def discriminator(inputs, reuse=False, training=True, use_bn=False, variable_scope_name='discriminator'):
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            d1 = NetBlocks.conv2d(inputs, 8, (3, 3), (2, 2), activation=None, use_bn=False, name='conv1')
            # d1 = tf.nn.relu(d1)
            d2 = NetBlocks.residual2d(d1, (4, 16), same_size=False, use_bn=use_bn, training=training, name='conv2')
            d3 = NetBlocks.residual2d(d2, (4, 16), same_size=False, use_bn=use_bn, training=training, name='conv3')
            d4 = NetBlocks.residual2d(d3, (8, 32), same_size=False, use_bn=use_bn, training=training, name='conv4')
            d5 = NetBlocks.residual2d(d4, (8, 32), same_size=False, use_bn=use_bn, training=training, name='conv5')
            d6 = NetBlocks.residual2d(d4, (8, 32), same_size=False, use_bn=use_bn, training=training, name='conv6')
            # d2 = NetBlocks.conv2d(d1, 8, (3, 3), (2, 2), training=training, use_bn=False, name='conv2')
            # d3 = NetBlocks.conv2d(d2, 16, (3, 3), (2, 2), training=training, use_bn=False, name='conv3')
            # d4 = NetBlocks.conv2d(d3, 32, (3, 3), (2, 2), training=training, use_bn=False, name='conv4')
            # d5 = NetBlocks.conv2d(d4, 32, (3, 3), (2, 2), training=training, use_bn=False, name='conv5')
            # d6 = NetBlocks.conv2d(d5, 32, (3, 3), (2, 2), training=training, use_bn=False, name='conv6')
            # mp = tf.reduce_max(d6, axis=[2, 3], keepdims=True)
            mp = tf.layers.max_pooling2d(d6, 2, 2)
            flt = tf.layers.flatten(mp)
            ds = tf.layers.dense(flt, 1, activation=None, use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1), name='ds')
            print('ds: ', ds)
        return ds, [d1, d2, d3, d4, d5]

    def build_model(self, inputs, label):
        self.data = inputs
        self.label = label
        self.g_out = self.generator(inputs, variable_scope_name='generator')
        self.d_real, self.real_feat_list = self.discriminator(label, variable_scope_name='discriminator')
        self.d_fake, self.fake_feat_list = self.discriminator(self.g_out, reuse=True)

        # gradient penalty
        epsilon = tf.random_uniform([self.data.shape[0].value, 1, 1, 1], 0.0, 1.0)
        interpolated = epsilon * label + (1 - epsilon) * self.g_out
        d_inter, _ = self.discriminator(interpolated, reuse=True)
        # tf.gradients returns a list of sum(dy/dx) for each x in xs.
        gradients = tf.gradients(d_inter, [interpolated, ], name="d_inter")[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))

        self.grad_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0), name='gd_penalty')

    def calc_loss(self, data, label):
        self.build_model(data, label)

        with tf.name_scope('gram_loss'):
            gram1 = self.calc_gram(self.real_feat_list)
            gram2 = self.calc_gram(self.fake_feat_list)
            gram_losses = []
            for idx in range(len(gram1)):
                # h, w = gram1[idx].shape
                gl = tf.norm(gram1[idx]-gram2[idx], ord=1)  # / h.value / w.value
                gram_losses.append(gl)
            gram_loss = tf.reduce_mean(gram_losses)

        with tf.name_scope('d_loss'):
            real_loss = -tf.reduce_mean(self.d_real) + tf.reduce_mean(self.d_fake) + self.gp_lambda * self.grad_penalty
            self.d_loss = real_loss  # - gram_loss*self.alpha

        with tf.name_scope('label_loss'):
            label_loss = tf.abs(tf.pow(label, 3) - tf.pow(self.g_out, 3)) * (0.05 + tf.pow(label, 2))
            label_loss = tf.reduce_mean(tf.reduce_mean(label_loss, axis=[1, 2, 3]), name='label_loss')
            # label_loss = tf.losses.mean_squared_error(label, self.g_out, weights=label, scope='label_loss')

        with tf.name_scope('g_loss'):
            fake_loss = -tf.reduce_mean(self.d_fake)
            self.g_loss = fake_loss + self.beta*label_loss + gram_loss*self.alpha

        tf.add_to_collection(name='defined_loss', value=self.g_loss)
        tf.add_to_collection(name='defined_loss', value=gram_loss)
        tf.add_to_collection(name='defined_loss', value=label_loss)
        tf.add_to_collection(name='defined_loss', value=self.grad_penalty)

        return self.d_loss, self.g_loss

    def make_summary(self):
        # make summary
        with tf.name_scope('define_losses'):
            for d_l in tf.get_collection(key='defined_loss'):
                tf.summary.scalar(name=d_l.name, tensor=d_l)
        with tf.name_scope('vars_hists'):
            summary_vars_list = []
            for variable in tf.trainable_variables():
                summary_vars_list.append(tf.summary.histogram(name=variable.name, values=variable))
            tf.summary.merge(inputs=summary_vars_list, )
        with tf.name_scope('d_g_losses'):
            tf.summary.scalar(name='d_loss', tensor=self.d_loss)
            tf.summary.scalar(name='g_loss', tensor=self.g_loss)
        with tf.name_scope('pred_img'):
            for i in range(6):
                tf.summary.image(name=str(i + 1), tensor=self.g_out[0:1, :, :, i:i+1], max_outputs=1)
        with tf.name_scope('label_img'):
            for i in range(6):
                tf.summary.image(name=str(i + 1), tensor=self.label[0:1, :, :, i:i+1], max_outputs=1)
        with tf.name_scope('img_hists'):
            tf.summary.histogram(name='label', values=self.label)
            tf.summary.histogram(name='pred', values=self.g_out)
        tf.summary.merge_all(scope='pred_img')
        tf.summary.merge_all(scope='label_img')


class DataLoader(object):
    def __init__(self, batch_size, height, width, depth_in, depth_out, train_file_list, val_file_list, data_path):
        self.data_path = data_path
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.depth_in = depth_in
        self.depth_out = depth_out
        self.train_file_list = train_file_list
        self.val_file_list = val_file_list
        self.handle = tf.placeholder(dtype=tf.string, shape=[])

    def get_iterator(self):
        with tf.name_scope('train_set_iterator'):
            train_set = tf.data.TextLineDataset(self.train_file_list, buffer_size=400,)
            train_set = train_set.shuffle(200).map(self.map_single_text_line, 4).repeat().batch(self.batch_size)
            train_iterator = train_set.make_one_shot_iterator()

        with tf.name_scope('val_set_iterator'):
            val_set = tf.data.TextLineDataset(self.val_file_list, buffer_size=100, )
            val_set = val_set.map(self.map_single_text_line).batch(self.batch_size)
            val_iterator = val_set.make_initializable_iterator()

        iterator = tf.data.Iterator.from_string_handle(self.handle, output_types=(tf.float32, tf.float32),
                                                       output_shapes=(
                                                           [self.batch_size, self.depth_in, self.height, self.width, 1],
                                                           [self.batch_size, self.height, self.width, self.depth_out],
                                                       ))

        return iterator, train_iterator, val_iterator

    def map_single_text_line(self, text):
        with tf.name_scope('map_fun'):
            path = self.data_path + text
            img_prefix = tf.string_split(tf.reshape(text, shape=[1]), delimiter='/').values[-1]
            images = []
            image_idx = [i for i in range(31)] + [35, 40, 45, 50, 55, 60]
            for idx in image_idx:
                img_suffix = '_{:03d}.png'.format(idx)
                img_content = tf.read_file(path+'/'+img_prefix+img_suffix, name='img_file_reader')
                img = tf.image.decode_png(img_content, channels=0, name='png_decoder')
                img = tf.where(tf.not_equal(img, 255), img, tf.zeros_like(img))
                images.append(tf.reshape(img, shape=(1, self.height, self.width, 1)))

            data = tf.concat(images[0:self.depth_in], axis=0)
            label = tf.concat(images[self.depth_in:], axis=-1)
            label = tf.squeeze(label, axis=[0])

        return tf.cast(data, tf.float32) / 80, tf.cast(label, tf.float32) / 80


def train():
    # config
    config = arg_config()
    # data
    data_loader = DataLoader(BatchSize, Height, Width, Depth_In, Depth_Out, TrainList, ValList, '../nas/SRAD2018/')
    iter_mask, train_iter, val_iter = data_loader.get_iterator()
    data, label = iter_mask.get_next()
    print('label: ', label)
    # model & loss
    model_wgan = ModelWGANGP(Height, Width, Depth_Out)
    d_loss, g_loss = model_wgan.calc_loss(data, label)

    # train
    with tf.name_scope('d_trainer'):
        # d_global_step = tf.Variable(0, trainable=False)
        # d_lr_schedule = tf.train.exponential_decay(0.001, d_global_step, 20000, 0.5, staircase=True)
        d_optimizer = tf.train.RMSPropOptimizer(1e-4)
    with tf.name_scope('g_trainer'):
        # g_global_step = tf.Variable(0, trainable=False)
        # g_lr_schedule = tf.train.exponential_decay(0.001, g_global_step, 10000, 0.5, staircase=True)
        g_optimizer = tf.train.RMSPropOptimizer(1e-4)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')):
        d_train = d_optimizer.minimize(d_loss, var_list=tf.trainable_variables(scope='discriminator'))
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')):
        g_train = g_optimizer.minimize(g_loss, var_list=tf.trainable_variables(scope='generator'))

    # summary
    model_wgan.make_summary()
    # with tf.name_scope('learning_rate'):
    #     tf.summary.scalar(name='d_lr', tensor=d_lr_schedule)
    #     tf.summary.scalar(name='g_lr', tensor=g_lr_schedule)
    summary_op = tf.summary.merge_all()
    # loss for print
    defined_loss = tf.get_collection(key='defined_loss')

    with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
        suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_writer = tf.summary.FileWriter('../nas/SRAD2018Model/summary/train_wgan_gp_'+suffix,
                                               graph=sess.graph, flush_secs=5)
        # saver and init
        saver = tf.train.Saver(max_to_keep=5, )
        sess.run(tf.global_variables_initializer())
        # tf.train.Saver(var_list=tf.trainable_variables(scope='generator')
        # ).restore(sess, '../nas/SRAD2018Model/weights/model_wgan_gp-24000', )
        # val_list_to_init = tf.global_variables()
        # for val in tf.trainable_variables(scope='generator'):
        #     val_list_to_init.remove(val)
        # sess.run(tf.variables_initializer(var_list=val_list_to_init))

        # handle
        train_handel = sess.run(train_iter.string_handle())
        val_handel = sess.run(val_iter.string_handle())

        current_step = 0
        max_step = 120000
        while current_step < max_step:
            # while True:
            current_step += 1
            for i in range(2):
                _, d_ls = sess.run([d_train, d_loss], feed_dict={data_loader.handle: train_handel})

            dl, _, g_ls, summary = sess.run([defined_loss, g_train, g_loss, summary_op],
                                            feed_dict={data_loader.handle: train_handel})

            print('step: {}, d_loss: {:.3f}, g_loss: {:.3f}'.format(current_step, d_ls, g_ls))
            for n, v in zip(defined_loss, dl):
                print('{}: {:.5f}'.format(n.name, v))
            print()

            summary_writer.add_summary(summary, global_step=current_step)
            # summary_writer.add_summary(summary_f, global_step=current_step)

            # if current_step % 100 == 0:
            #     sess.run(val_iter.initializer)
            #     while True:
            #         try:
            #             _, ls = sess.run([g_train, f_loss], feed_dict={data_loader.handle: val_handel})
            #             print('fake loss: ', ls)
            #         except tf.errors.OutOfRangeError:
            #             break
            if current_step % 1000 == 0:
                saver.save(sess, '../nas/SRAD2018Model/weights/model_wgan_gp', global_step=current_step)

        summary_writer.close()


# global variable###################################
BatchSize = 2
Height = 501
Width = 501
Depth_In = 31
Depth_Out = 6
TrainList = []
ValList = []


if __name__ == '__main__':
    TrainList = glob.glob('../nas/SRAD2018Model/train_val_files/train*')
    ValList = glob.glob('../nas/SRAD2018Model/train_val_files/val*')
    TrainList = sorted(TrainList, key=lambda x: int(x[-7:-4]))
    print(TrainList)
    train()

    # pp = '/Users/whyguu/Desktop/t4_RAD_296582494212544_004.png'
    # pp = '../nas/SRAD2018/SRAD2018_TRAIN_004/RAD_296582494212544/RAD_296582494212544_004.png'
    # im = io.imread(pp)
    # for i in range(10):
    #     print('sakljdflajkshfakhjflahf: ', im.shape)
    # content = tf.read_file(pp)
    # img = tf.image.decode_png(content, channels=0)
    # with tf.Session() as sess:
    #     try:
    #         ig = sess.run(img)
    #         print(ig.shape)
    #     except:
    #         print('haha')








