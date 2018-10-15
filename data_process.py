import numpy as np
import tensorflow as tf
import os
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
import imageio
import multiprocessing
from tqdm import tqdm


class ViewData(object):
    @staticmethod
    def view_data():
        color_bar = [
            43, 131, 186,
            77, 155, 180,
            111, 179, 174,
            145, 203, 169,
            177, 224, 166,
            199, 233, 173,
            222, 242, 180,
            244, 251, 188,
            255, 245, 179,
            255, 223, 154,
            254, 201, 128,
            254, 180, 103,
            246, 144, 83,
            236, 104, 65,
            226, 64, 46,
            215, 25, 28,
            225, 10, 10,
        ]
        color_bar = np.array(color_bar, dtype=np.uint8).reshape(-1, 3)
        src_path = '/workspace/nas/SRAD2018/SRAD2018_TRAIN_001/'
        dst_path = '/workspace/nas/SRAD2018Model/view_data/'
        samples = os.listdir(src_path)
        for i in range(51, len(samples)):  # len(samples)
            try:
                os.mkdir(dst_path+samples[i])
            except FileExistsError:
                print('file exist ')
            imgs = os.listdir(src_path+samples[i])
            for img in imgs:
                # print(radar.shape)
                # print(radar.dtype)
                print(src_path+samples[i]+'/'+img)
                radar = io.imread(src_path + samples[i] + '/' + img)
                unique_data = np.unique(radar.ravel())
                # print(unique_data)
                if unique_data[-1] == 255:
                    stop_point = unique_data[-2]
                else:
                    stop_point = unique_data[-1]
                pseudo = np.zeros(shape=(501, 501, 3), dtype=np.uint8)
                index = 0
                for c in range(0, stop_point+5, 5):
                    idx = (radar < c+5) * (radar >= c)
                    # print(idx)
                    pseudo[idx, :] = color_bar[index, :]
                    index += 1
                io.imsave(dst_path+samples[i]+'/'+img, pseudo)

    @staticmethod
    def make_gif_example():
        path = '/Users/whyguu/Desktop/RAD_276482464229544'
        img_names = os.listdir(path)
        img_names = sorted(img_names, key=lambda x: int(x[-7:-4]))
        print(img_names)

        frames = []
        for name in img_names:
            img = io.imread(path+'/'+name)
            frames.append(img)
        imageio.mimsave('/Users/whyguu/Desktop/RAD_276482464229544.gif', ims=frames, format='gif', duration=0.2)


class DataGenerator(object):
    def generate_tfrecords(self):
        pools = multiprocessing.Pool(8)
        data_path = '../nas/SRAD2018/'
        paths = os.listdir(data_path)
        paths = [path for path in paths if 'TRAIN' in path and 'zip' not in path]
        print('paths: ', paths)
        for path in paths:  # SRAD2018_TRAIN_xxx/
            try:
                os.remove(data_path+path+'/.DS_Store')
            except:
                pass
            samples = os.listdir(data_path+path)
            train_sample, val_sample = train_test_split(samples, test_size=0.05, random_state=42)
            # print(len(train_sample))
            # print(type(train_sample))
            train_dst_path = '../nas/SRAD2018Model/train_val_records/train_{:03d}.tfrecords'.format(int(path[-3:]))
            val_dst_path = '../nas/SRAD2018Model/train_val_records/val_{:03d}.tfrecords'.format(int(path[-3:]))
            pools.apply_async(func=self.gen_tfrecords_file, args=(train_sample, train_dst_path, data_path+path))
            pools.apply_async(func=self.gen_tfrecords_file, args=(val_sample, val_dst_path, data_path+path))

        pools.close()
        pools.join()

    @staticmethod
    def gen_tfrecords_file(samples, dst_path, sample_path):
        tmp_data = np.zeros(shape=(501, 501, 61), dtype=np.uint8)
        writer = tf.python_io.TFRecordWriter(dst_path)
        print('make: ', dst_path)
        # print(samples)
        for sample in samples:  # thousands samples
            try:
                os.remove(os.path.join(sample_path, sample, '.DS_Store'))
            except:
                pass
            imgs = os.listdir(os.path.join(sample_path, sample))
            assert len(imgs) == 61
            imgs = sorted(imgs, key=lambda x: int(x[-7:-4]))
            # print(imgs)
            for idx, img in enumerate(imgs):  # img names
                tmp_data[:, :, idx] = io.imread(os.path.join(sample_path, sample, img)).astype(np.uint8)
            # ######################  data pre-process  start ######################
            # data, label = data_process(data, label)
            # ######################  data pre-process  end ######################
            # gen record
            tf_features = tf.train.Features(feature={
                # 'data_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(data.shape))),
                'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_data[:, :, 0:31].tostring()])),
                # 'label_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(label.shape))),
                'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_data[:, :, 31:].tostring()])),
            })

            example = tf.train.Example(features=tf_features)
            writer.write(record=example.SerializeToString())
        writer.close()

    @staticmethod
    def generate_sample_file():
        data_path = '../nas/SRAD2018/'
        paths = os.listdir(data_path)
        paths = [path for path in paths if 'TRAIN' in path and 'zip' not in path]

        for path in paths:
            print(path)
            samples = os.listdir(data_path+path)
            samples = [path+'/'+sample for sample in samples if os.path.isdir(os.path.join(data_path, path, sample))]
            train_sample, val_sample = train_test_split(samples, test_size=0.05, random_state=42)
            with open('../nas/SRAD2018Model/train_val_files/train_{:03d}.txt'.format(int(path[-3:])), 'w') as f:
                f.write('\n'.join(train_sample))
            with open('../nas/SRAD2018Model/train_val_files/val_{:03d}.txt'.format(int(path[-3:])), 'w') as f:
                f.write('\n'.join(val_sample))

    @staticmethod
    def gen_test_record():
        path = '../nas/SRAD2018/SRAD2018_Test_1/'
        samples = os.listdir(path)

        samples = [sample for sample in samples if os.path.isdir(path+sample)]
        smp = np.zeros((31, 501, 501), dtype=np.uint8)
        writer = tf.python_io.TFRecordWriter('../nas/SRAD2018Model/test/test_1.tfrecords')
        print(len(samples))
        for idx, sample in enumerate(samples):
            print(idx, sample)
            for i in range(31):
                smp[i, :, :] = io.imread(path+sample+'/'+sample+'_{:03d}.png'.format(i)).astype(np.uint8)

            feat = tf.train.Features(feature={
                    'sample_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[smp.tostring()])),
            })
            example = tf.train.Example(features=feat)
            writer.write(record=example.SerializeToString())
        writer.close()

    @staticmethod
    def only_cloud():
        # folders = [fd for fd in os.listdir('../nas/SRAD2018/') if os.path.isdir('../nas/SRAD2018/'+fd) and 'TRAIN' in fd]
        folders = ['SRAD2018_TRAIN_060']
        print(folders)

        for fd in folders:
            try:
                os.remove('../nas/SRAD2018/'+fd+'/.DS_Store')
            except FileNotFoundError:
                pass
            pools = multiprocessing.Pool(10)
            samples = os.listdir('../nas/SRAD2018/'+fd+'/')
            sps = [samples[i:i + 500] for i in range(0, len(samples), 500)]
            rlt = [pools.apply_async(DataGenerator.sample_process, args=(fd, sp)) for sp in sps]
            pools.close()
            pools.join()
            results = []
            for r in rlt:
                results += r.get()
            # print(results)
            print(1.0*len(results) / len(samples))
            pd.DataFrame(results).to_csv('../nas/SRAD2018Model/valid_data_files/train_{}.csv'.format(fd[-3:]), index=None, header=None)

    @staticmethod
    def sample_process(fd, samples):
        valid_samples = []
        # print(len(samples))
        for sample in tqdm(samples):
            img_names = os.listdir('../nas/SRAD2018/' + fd + '/' + sample)
            ratio = []
            for name in img_names[0:40]:
                img = io.imread('../nas/SRAD2018/' + fd + '/' + sample + '/' + name).astype(np.uint8)
                # ratio.append(np.mean(img == 255))
                ratio.append(np.mean(img > 0))
            # if np.mean(ratio) < 0.4:
            if np.mean(ratio) > 0.3:
                valid_samples.append(fd + '/' + sample)
        return valid_samples


if __name__ == '__main__':
    # vd = ViewData()
    # vd.view_data()

    dg = DataGenerator()
    # dg.gen_test_record()
    # dg.generate_tfrecords()
    # dg.generate_sample_file()
    dg.only_cloud()

    # files = os.listdir('../nas/SRAD2018Model/valid_data_files/')
    # count = 0
    # for f in files:
    #     count += len(pd.read_csv('../nas/SRAD2018Model/valid_data_files/'+f))
    # print(count)
    print('aaa')


