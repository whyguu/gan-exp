import shutil
import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import tensorflow as tf
import glob
from skimage import io
import warnings


def fake_result(samples):
    pid = os.getpid()
    print("pid :", pid)
    path = '../nas/SRAD2018/SRAD2018_Test_1/'
    for sample in tqdm(samples):
        # print(sample)
        os.mkdir('../nas/SRAD2018Model/result/fake_rlt/'+sample)

        for i in range(1, 7):
            suffix = '_f00{}.png'.format(i)
            shutil.copy(path+sample+'/'+sample+'_030.png',
                        '../nas/SRAD2018Model/result/fake_rlt/'+sample+'/'+sample+suffix)


def fake_result1(samples):
    mean_idx = 2
    rlt_dir = '../nas/SRAD2018Model/result/fake_rlt_mean3_at_{}/'.format(mean_idx)
    try:
        os.mkdir(rlt_dir)
    except:
        pass
    pid = os.getpid()
    print("pid :", pid)
    data_path = '../nas/SRAD2018/SRAD2018_Test_1/'
    imgs = np.zeros(shape=(501, 501, 3), dtype=np.uint8)
    dx = [1, 2, 3, 4, 5, 6]
    del dx[mean_idx - 1]
    print(dx)
    for sample in tqdm(samples):
        # print(sample)
        os.mkdir(rlt_dir+sample)

        # mean data
        for i in range(30, 27, -1):
            suffix = '_{:03d}.png'.format(i)
            imgs[:, :, i-30] = io.imread(data_path+sample+'/'+sample+suffix)
        mean_img = np.mean(imgs, axis=2)
        mean_img[mean_img > 80] = 255
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(rlt_dir+sample+'/'+sample+'_f00{}.png'.format(mean_idx), mean_img.astype(np.uint8))

        for i in dx:
            shutil.copy(data_path+sample+'/'+sample+'_030.png',
                        rlt_dir+sample+'/'+sample+'_f00{}.png'.format(i))


def parallel_():
    parallel_num = 16
    pools = multiprocessing.Pool(parallel_num)

    path = '../nas/SRAD2018/SRAD2018_Test_1/'
    samples = os.listdir(path)
    samples = [sample for sample in samples if os.path.isdir(path + sample)]

    ls = len(samples) // parallel_num
    print(ls)
    for i in range(parallel_num):
        if i == parallel_num-1:
            sp = samples[i * ls:]
            pools.apply_async(fake_result1, (sp,))

        else:
            sp = samples[i*ls:(i+1)*ls]
            pools.apply_async(fake_result1, (sp, ))
    pools.close()
    pools.join()


def zip_file():
    compacted_zip_name = 'haha_grad'
    dir_to_be_compact = 'unet_grad'
    shutil.make_archive(base_name='../nas/SRAD2018Model/result/'+compacted_zip_name, format='zip',
                        root_dir='../nas/SRAD2018Model/result/', base_dir=dir_to_be_compact)
    print('make archive done!')


def data_loader_test():
    train_list = glob.glob('../nas/SRAD2018Model/train_val_files/train*')
    train_list = sorted(train_list, key=lambda x: int(x[-7:-4]))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 使用 GPU id
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options=gpu_options)
    train_set = tf.data.TextLineDataset(train_list[3:], buffer_size=200, ).make_one_shot_iterator()
    txt = train_set.get_next()
    p = tf.placeholder(tf.string, )
    img_content = tf.read_file(p)
    img = tf.image.decode_image(img_content, channels=0)
    # f = open('loader_test.txt')
    with tf.Session(config=config) as sess:
        count = 0
        while True:
            path = sess.run(txt)
            path = np.array(path).astype(np.str)
            path = '../nas/SRAD2018/' + str(path)
            img_names = os.listdir(path)
            img_names = [name for name in img_names if '.png' in name]
            img_names = sorted(img_names, key=lambda x: int(x[-7:-4]))

            for idx, name in enumerate(img_names):
                try:
                    sess.run(img, feed_dict={p: path+'/'+name})
                except:
                    print(count, path+'/'+name)
            count += 1

            if count % 1000 == 0:
                print('decode image num: ', count)


if __name__ == '__main__':
    # parallel_()
    zip_file()
    # data_loader_test()
    pass

