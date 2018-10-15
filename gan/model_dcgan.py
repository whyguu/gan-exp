import tensorflow as tf
import datetime
import os
import glob
import argparse
from gan.model_wgan_gp import NetBlocks


def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='2')
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


class ModelGAN(object):
    def __init__(self, height, width, channel_out):
        self.alpha = 0.5  # ratio of gram loss
        self.beta = 90  # ratio MAE/MSE loss
        self.height = height
        self.width = width
        self.channel_out = channel_out

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
                g_conv1 = NetBlocks.conv3d(inputs, 8, (3, 3, 3), (2, 2, 2), training=training, name='conv3d_1')
                # input: 16*251*251*8
                g_conv2 = NetBlocks.residual3d(g_conv1, [4, 16], same_size=False, training=training, name='res_1')
                # input: 8*126*126*16
                g_conv3 = NetBlocks.residual3d(g_conv2, [4, 16], True, True, training=training, name='res_2')
                # input: 8*126*126*16
                g_conv4 = NetBlocks.residual3d(g_conv3, [8, 32], same_size=False, training=training, name='res_3')
                # input: 4*63*63*32
                g_conv5 = NetBlocks.residual3d(g_conv4, [8, 32], same_size=True, training=training, name='res_4')
                # input: 4*63*63*32
                g_conv6 = NetBlocks.conv3d(g_conv5, 64, (3, 3, 3), (1, 2, 2), training=training, name='conv3d_2')
                # input: 4*32*32*64
                g_conv7 = NetBlocks.conv3d(g_conv6, 64, (1, 1, 1), (1, 1, 1), training=training, name='conv3d_3')
                # input: 4*32*32*64
                print('g_conv1: ', g_conv1)
                print('g_conv2: ', g_conv2)
                print('g_conv3: ', g_conv3)
                print('g_conv4: ', g_conv4)
                print('g_conv5: ', g_conv5)
                print('g_conv6: ', g_conv6)
                print('g_conv7: ', g_conv7)

            with tf.name_scope(name='conv_trans'):
                g_trans4 = NetBlocks.up_concat3d(g_conv7, g_conv5, [32, 32], (5, 5, 5), (1, 2, 2), name='trans4')
                # 4*63*63*32
                g_trans3 = NetBlocks.up_concat3d(g_trans4, g_conv3, [16, 16], (5, 5, 5), (2, 2, 2), name='trans3')
                # 8*126*126*16
                g_trans2 = NetBlocks.up_concat3d(g_trans3, g_conv1, [8, 8], (5, 5, 5), (2, 2, 2), name='trans2')
                # 16*251*251*8

                g_trans1 = NetBlocks.conv3d_trans(g_trans2, 8, (3, 5, 5), (1, 2, 2),
                                                  (g_trans2.shape[1].value, self.height, self.width), name='trans1')

                print('g_conv_trans4: ', g_trans4)
                print('g_conv_trans3: ', g_trans3)
                print('g_conv_trans2: ', g_trans2)
                print('g_conv_trans1: ', g_trans1)

                b, d, h, w, c = g_trans1.shape
                ts = tf.transpose(g_trans1, perm=(0, 2, 3, 1, 4))
                g_trans0 = tf.reshape(ts, shape=(b.value, h.value, w.value, d.value*c.value))
                out = NetBlocks.conv2d(g_trans0, 32, (3, 3), name='out1')
                out = NetBlocks.conv2d(out, self.channel_out, (1, 1), name='out2')
                # out = tf.nn.relu(out)
                out = tf.nn.sigmoid(out)
                print('g_out: ', out)

            return out

    @staticmethod
    def discriminator(inputs, reuse=False, training=True, variable_scope_name='discriminator'):
        with tf.variable_scope(variable_scope_name, reuse=reuse):
            d1 = NetBlocks.conv2d(inputs, 4, (3, 3), (2, 2), activation=None, training=training, name='conv1')
            d2 = NetBlocks.conv2d(d1, 8, (3, 3), (2, 2), training=training, name='conv2')
            d3 = NetBlocks.conv2d(d2, 16, (3, 3), (2, 2), training=training, name='conv3')
            d4 = NetBlocks.conv2d(d3, 32, (3, 3), (2, 2), training=training, name='conv4')
            d5 = NetBlocks.conv2d(d4, 32, (3, 3), (2, 2), training=training, name='conv5')
            d6 = NetBlocks.conv2d(d5, 32, (3, 3), (2, 2), training=training, name='conv6')
            # mp = tf.reduce_max(d6, axis=[2, 3], keepdims=True)
            mp = tf.layers.max_pooling2d(d6, 2, 2)
            flt = tf.layers.flatten(mp)
            ds = tf.layers.dense(flt, 1, activation=None, use_bias=False,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            print('ds: ', ds)
        return ds, [d1, d2, d3, d4, d5]

    def calc_loss(self, labels, gen_out, f_list1, f_list2, real_out, fake_out):
        with tf.name_scope('gram_loss'):
            gram1 = self.calc_gram(f_list1)
            gram2 = self.calc_gram(f_list2)
            gram_losses = []
            for idx in range(len(gram1)):
                # h, w = gram1[idx].shape
                gl = tf.norm(gram1[idx]-gram2[idx], ord=1)  # / h.value / w.value
                gram_losses.append(gl)
            gram_loss = tf.reduce_mean(gram_losses)

        with tf.name_scope('real_loss'):
            rl1 = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_out, dtype=tf.float32), real_out)
            rl2 = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_out, dtype=tf.float32), fake_out)
            real_loss_sigmoid = tf.reduce_mean(rl1 + rl2, name='real_loss_sigmoid')
            real_loss = real_loss_sigmoid  # - gram_loss*self.alpha

        with tf.name_scope('label_loss'):
            label_loss = tf.reduce_mean(tf.abs(labels-gen_out), name='label_loss')
            # label_loss = tf.losses.mean_squared_error(labels, gen_out, scope='label_loss')
        with tf.name_scope('fake_loss'):
            fake_loss_sigmoid = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_out, dtype=tf.float32), fake_out)
            fake_loss_sigmoid = tf.reduce_mean(fake_loss_sigmoid, name='fake_loss_sigmoid')
            fake_loss = fake_loss_sigmoid + gram_loss*self.alpha + self.beta*label_loss

        tf.add_to_collection(name='defined_loss', value=fake_loss_sigmoid)
        tf.add_to_collection(name='defined_loss', value=gram_loss)
        tf.add_to_collection(name='defined_loss', value=label_loss)

        return real_loss, fake_loss


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
            train_set = tf.data.TextLineDataset(self.train_file_list, buffer_size=200,)
            train_set = train_set.map(self.map_single_text_line, 4).repeat().batch(self.batch_size)
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
    # model
    model_gan = ModelGAN(Height, Width, Depth_Out)
    g_out = model_gan.generator(data, variable_scope_name='generator')
    d_real, real_feat_list = model_gan.discriminator(label, variable_scope_name='discriminator')
    d_fake, fake_feat_list = model_gan.discriminator(g_out, reuse=True, variable_scope_name='discriminator')
    # loss
    r_loss, f_loss = model_gan.calc_loss(label, g_out, real_feat_list, fake_feat_list, d_real, d_fake)
    # train
    with tf.name_scope('d_trainer'):
        d_global_step = tf.Variable(0, trainable=False)
        d_lr_schedule = tf.train.exponential_decay(0.001, d_global_step, 1000, 0.5, staircase=True)
        d_optimizer = tf.train.AdamOptimizer(d_lr_schedule)
    with tf.name_scope('g_trainer'):
        g_global_step = tf.Variable(0, trainable=False)
        g_lr_schedule = tf.train.exponential_decay(0.001, g_global_step, 2000, 0.5, staircase=True)
        g_optimizer = tf.train.AdamOptimizer(g_lr_schedule)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')):
        d_train = d_optimizer.minimize(r_loss, var_list=tf.trainable_variables(scope='discriminator'))
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')):
        g_train = g_optimizer.minimize(f_loss, var_list=tf.trainable_variables(scope='generator'))
    # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    # # print(tf.trainable_variables())
    # exit(0)
    # loss collection
    defined_loss = tf.get_collection(key='defined_loss')

    # make summary
    with tf.name_scope('my_summary'):
        tf.summary.scalar(name='real_loss', tensor=r_loss)
        tf.summary.scalar(name='fake_loss', tensor=f_loss)
        for d_l in defined_loss:
            tf.summary.scalar(name=d_l.name, tensor=d_l)
        summary_vars_list = []
        for i in range(6):
            tf.summary.image(name='pred_img/'+str(i+1), tensor=tf.expand_dims(g_out[:, :, :, i], axis=-1), max_outputs=1)
            tf.summary.image(name='label_img/'+str(i+1), tensor=tf.expand_dims(label[:, :, :, i], axis=-1), max_outputs=1)
        tf.summary.merge_all(scope='pred_img')
        tf.summary.merge_all(scope='label_img')
        for variable in tf.trainable_variables():
            summary_vars_list.append(tf.summary.histogram(name=variable.name, values=variable))
        tf.summary.merge(inputs=summary_vars_list, )
        summary_op = tf.summary.merge_all()

    with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
        suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_writer = tf.summary.FileWriter('../nas/SRAD2018Model/summary/train_gan'+suffix,
                                               graph=sess.graph, flush_secs=5)

        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, tf.train.latest_checkpoint('../nas/SRAD2018Model/weights/'))
        # handle
        train_handel = sess.run(train_iter.string_handle())
        val_handel = sess.run(val_iter.string_handle())

        current_step = 0
        max_step = 12000000
        while current_step < max_step:
            # while True:
            current_step += 1

            _, r_ls = sess.run([d_train, r_loss], feed_dict={data_loader.handle: train_handel})
            for i in range(2):

                dl, _, f_ls, summary = sess.run([defined_loss, g_train, f_loss, summary_op],
                                                feed_dict={data_loader.handle: train_handel})

            print('step: {}, real loss: {:.3f}, fake loss: {:.3f}'.format(current_step, r_ls, f_ls))
            print('{}: {:.5f}'.format(defined_loss[0].name, dl[0]))
            print('{}: {:.5f}'.format(defined_loss[1].name, dl[1]))
            print('{}: {:.5f}'.format(defined_loss[2].name, dl[2]))

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
            if current_step % 500 == 0:
                saver.save(sess, '../nas/SRAD2018Model/weights/model_gan', global_step=current_step)

        summary_writer.close()


# global variable###################################
BatchSize = 4
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












