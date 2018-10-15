import tensorflow as tf
import numpy as np
import os
from skimage import io
import argparse
import time


def arg_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    args = parser.parse_args()

    # config
    log_device_placement = False  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU id
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)

    return config


class TestLoader(object):
    def __init__(self):
        self.path = '/workspace/nas/SRAD2018/SRAD2018_Test_1/'
        self.height = 501
        self.width = 501
        self.channel = 31

    def data_gen(self):
        samples = os.listdir(self.path)
        for sp in samples:
            images = os.listdir(self.path+sp)
            images = sorted(images, key=lambda x: int(x[-7:-4]))
            data = []
            for img in images:
                data.append(np.expand_dims(io.imread(self.path+sp+'/'+img), axis=-1))
            return np.concatenate(data, axis=-1).astype(np.float32), sp

    def next_batch(self):
        ds = tf.data.Dataset.from_generator(generator=self.data_gen, output_types=tf.float32)
        data = ds.batch(1).make_one_shot_iterator().get_next()
        return data


def infer():
    data, img_name = TestLoader().next_batch()
    dst_path = '/workspace/nas/SRAD2018Model/test_1_rlt/'
    os.makedirs(dst_path, exist_ok=True)

    wgan = ModelWGAN(501, 501, 6)
    model_out = wgan.generator(data, training=False)
    saver = tf.train.Saver()
    weight_path = '/workspace/nas/SRAD2018Model/weights/model_gan-17000'
    with tf.Session() as sess:
        saver.restore(sess, save_path=weight_path)

        while True:
            try:
                tic = time.time()
                out, name = sess.run([model_out, img_name])
                out = np.ceil(out).astype(np.uint8)
                for i in range(6):
                    os.makedirs(dst_path+name, exist_ok=True)
                    io.imsave(dst_path+name+'/'+name+'_f{:03d}'.format(i+1), out[0, :, :, i])
                print('process time: {:.3f}, {}'.format(time.time()-tic, name))
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    infer()


