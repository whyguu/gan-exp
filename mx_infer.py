import mxnet as mx
from mxnet import gluon
from mx_model import ModelUNet
import os
import numpy as np
from mxnet import nd
from mxnet import image
from skimage import io
from tqdm import tqdm
import multiprocessing
import warnings
import cv2


class TestDataSet(gluon.data.Dataset):
    def __init__(self, samples, channel_in=15, channel_out=2, **kwargs):
        super(TestDataSet, self).__init__(**kwargs)
        self.data_root = '../nas/SRAD2018/SRAD2018_Test_2/'
        self.data_scale = 80
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.samples = samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_root = self.data_root+sample
        img_names = [im for im in os.listdir(sample_root) if 'png' in im]
        if len(img_names) != 31:
            print(self.data_root + sample, 'have {} images. not equal 31.'.format(len(img_names)))
            exit(0)

        img_names = sorted(img_names, key=lambda x: int(x[-7:-4]))[-15:]
        # print(idx, sample_root)
        # dt = []
        # for name in img_names:
        #     try:
        #         tp = io.imread(sample_root + '/' + name)
        #     except:
        #         print(self.samples[idx])
        #     tp = nd.expand_dims(nd.squeeze(nd.array(tp)), axis=0)
        #     dt.append(tp)
        # dd = nd.concat(*dt, dim=0)
        # dd *= (dd != 255)
        dt = []
        for name in img_names:
            tp = cv2.imread(sample_root+'/'+name, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
            tp[tp == 255] = 0
            dt.append(tp)
        dd = np.concatenate(tuple(dt), axis=-1)

        gx = cv2.Sobel(dd, cv2.CV_16S, 1, 0, 5)
        gy = cv2.Sobel(dd, cv2.CV_16S, 0, 1, 5)
        gx = cv2.convertScaleAbs(gx)
        gy = cv2.convertScaleAbs(gy)
        g = np.sqrt(gx * gx + gy * gy)

        dd = np.concatenate((dd, g), axis=-1).transpose([2, 0, 1])

        dd = nd.array(dd)
        return dd.astype(np.float32)/self.data_scale

    def __len__(self):
        return len(self.samples)


def infer(samples, out_channel=3):
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    save_path = '../nas/SRAD2018Model/result/unet_grad/'
    os.makedirs(save_path, exist_ok=True)

    test_loader = gluon.data.DataLoader(TestDataSet(samples), batch_size=1, )
    print('sample num: ', len(samples))
    context = mx.gpu(1)
    model = ModelUNet(out_channels=out_channel)
    print('modle built !')
    model.load_parameters('../nas/SRAD2018Model/weights/mx_ModelUNet_1to3_grad_epoch_45.param', ctx=context)
    print('param loaded')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for idx, dt in tqdm(enumerate(test_loader)):
            os.makedirs(save_path+samples[idx], exist_ok=True)
            # print(idx, samples[idx])
            out = model(dt.as_in_context(context))
            out = np.clip(np.round(out.asnumpy()*80), 0, 255).astype(np.uint8)
            out[out == 0] = 255
            for i in range(6):  #
                j = i if i < out_channel else out_channel-1
                io.imsave(save_path+samples[idx]+'/{}_f{:03d}.png'.format(samples[idx], i+1), out[0, j, :, :])
    print('done')


if __name__ == '__main__':
    data_root = '../nas/SRAD2018/SRAD2018_Test_2/'
    sps = sorted([sp for sp in os.listdir(data_root) if os.path.isdir(data_root + sp)])
    smps = [sps[i:i+1000] for i in range(0, len(sps), 1000)]

    while smps:
        pools = multiprocessing.Pool(4)
        for i in range(4):
            pools.apply_async(infer, args=(smps.pop(0), ))
        pools.close()
        pools.join()






