import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd
import numpy as np
import multiprocessing
import os
from mxnet import nd
from mxnet import gluon
import glob
from skimage import io
from mxboard import *
import datetime
from queue import Queue
import pandas as pd
from mxnet.gluon.data.vision import transforms
from mxnet import image
from tqdm import tqdm
import cv2


class ResidualBlock(nn.Block):
    def __init__(self, channels=(), short_conv=False, strides=(1, 1), groups=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.short_conv = short_conv
        self.b1 = nn.Sequential()
        self.b1.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=channels[0], kernel_size=(1, 1), strides=1, padding=(0, 0), use_bias=False, groups=groups),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=channels[0], kernel_size=(3, 3), strides=strides, padding=(1, 1), use_bias=False, groups=groups),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels=channels[1], kernel_size=(1, 1), strides=1, padding=(0, 0), groups=groups)
        )
        if self.short_conv:
            self.short_cut = nn.Conv2D(channels=channels[1], kernel_size=(1, 1), strides=strides, padding=(0, 0), groups=groups)

    def forward(self, *args):
        x = args[0]
        c1 = self.b1(x)
        if self.short_conv:
            x = self.short_cut(x)
        return x + c1


class TransConcat(nn.Block):
    def __init__(self, channels, pad=1, out_pad=1, conv_num=2, **kwargs):
        super(TransConcat, self).__init__(**kwargs)
        self.name_scope()
        self.ts = nn.Conv2DTranspose(channels=channels, kernel_size=3, strides=2, padding=pad, output_padding=out_pad)
        self.cv = nn.Sequential()
        self.cv.add(nn.Conv2D(channels, 3, 1, 1))
        for i in range(conv_num):
            self.cv.add(ResidualBlock(channels=(channels//4, channels)))

    def forward(self, *args):
        x, c = args[0]
        cat = nd.concat(self.ts(x), c, dim=1)
        # print(b.shape)
        out = self.cv(cat)
        # print(out.shape)
        return out


class ModelUNet(nn.Block):
    def __init__(self, height=501, width=501, out_channels=6, **kwargs):
        self.height = height
        self.width = width
        self.out_channels = out_channels
        super(ModelUNet, self).__init__(**kwargs)
        with self.name_scope():
            # self.head = self.make_head()
            self.b0 = nn.Sequential()
            self.b0.add(
                nn.Conv2D(channels=16, kernel_size=3, strides=1, padding=1, groups=2),
                ResidualBlock(channels=(8, 16), short_conv=True, strides=2, groups=2),
            )
            self.b1 = nn.Sequential()
            self.b1.add(
                ResidualBlock(channels=(8, 16), short_conv=True, strides=2),
                ResidualBlock(channels=(8, 16)),
                ResidualBlock(channels=(8, 16)),
            )

            self.b2 = nn.Sequential()
            self.b2.add(
                ResidualBlock(channels=(8, 32), short_conv=True, strides=2),
                ResidualBlock(channels=(8, 32), ),
                ResidualBlock(channels=(8, 32), ),
                ResidualBlock(channels=(8, 32), ),
            )

            self.b3 = nn.Sequential()
            self.b3.add(
                ResidualBlock(channels=(16, 64), short_conv=True, strides=2),
                ResidualBlock(channels=(16, 64), ),
                ResidualBlock(channels=(16, 64), ),
                ResidualBlock(channels=(16, 64), ),
                ResidualBlock(channels=(16, 64), ),
            )

            self.b4 = nn.Sequential()
            self.b4.add(
                ResidualBlock(channels=(32, 128), short_conv=True, strides=2),
                ResidualBlock(channels=(32, 128), ),
                ResidualBlock(channels=(32, 128), ),
            )

            self.t3 = TransConcat(64, conv_num=3)
            self.t2 = TransConcat(32, out_pad=0, conv_num=3)
            self.t1 = TransConcat(16, conv_num=2)

            self.tail = nn.Sequential()
            self.tail.add(
                nn.Conv2D(self.out_channels, 1, 1, activation='sigmoid'),
            )

        self.fwd = (self.b0, self.b1, self.b2, self.b3, self.b4)
        self.up = (self.t3, self.t2, self.t1)

    def forward(self, *args):
        x = args[0]
        outs = [x]
        for block in self.fwd:
            outs.append(block(outs[-1]))
            # print(outs[-1].shape)
        for idx, block in enumerate(self.up):
            outs.append(block([outs[-1], outs[4-idx]]))
            # print(outs[-1].shape)
        outs.append(self.tail(outs[-1]))
        out = nd.contrib.BilinearResize2D(outs[-1], height=self.height, width=self.width)
        return out

    @staticmethod
    def make_head():
        hd = nn.Sequential()
        # hd.add(nn.BatchNorm(scale=False, center=False))
        hd.add(nn.Conv2D(channels=8, kernel_size=7, strides=2, padding=(1, 1), use_bias=False),
               nn.BatchNorm(),
               nn.Activation('relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return hd


def custom_loss(pred, label):
    channel_weights = nd.arange(1, 1+label.shape[1], ctx=label.context).reshape(1, -1, 1, 1) * 0.5
    # point_weights = 0.1 + label + (nd.clip(label, 0.5, 1.0) - 0.5) * 10
    point_weights = nd.exp(label*4) - 0.8
    tp = nd.abs(pred - label) * point_weights * channel_weights
    cloud_ratio = nd.mean(label > 0.1, axis=(0, 1), exclude=True)
    bc = nd.mean(tp, axis=(0, 1), exclude=True) * cloud_ratio
    return nd.mean(bc, axis=0, exclude=True)


class CustomDataSet(gluon.data.Dataset):
    def __init__(self, file_list, channel_in=15, channel_out=2, **kwargs):
        # self.file_root = '../nas/SRAD2018Model/train_val_files/'
        self.data_root = '../nas/SRAD2018/'
        self.file_list = file_list
        self.data_scale = 80
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.data_frame = None
        self.count = 0
        # init
        self.data_list()
        super(CustomDataSet, self).__init__(**kwargs)

    def data_list(self):
        dfs = list()
        for file_name in self.file_list:
            dfs.append(pd.read_csv(file_name, header=None))
        self.data_frame = pd.concat(dfs)
        self.count = len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx, 0]
        sample_root = self.data_root+sample
        img_names = [im for im in os.listdir(sample_root) if 'png' in im]
        img_names = sorted(img_names, key=lambda x: int(x[-7:-4]))
        start_num = int(np.random.randint(low=0, high=61-self.channel_out*5-self.channel_in, size=1))
        # if len(img_names) == 0:
        #     print(sample_root)
        #     exit()
        data_names = img_names[start_num:start_num+self.channel_in]
        label_names = img_names[start_num+self.channel_in+4::5][0:self.channel_out]

        dt = []
        lb = []
        # print(idx, sample_root)
        # for name in data_names:
        #     tp = image.imread(sample_root+'/'+name, flag=0)
        #     tp = nd.expand_dims(nd.squeeze(tp, axis=2), axis=0)
        #     dt.append(tp)
        # for name in label_names:
        #     tp = image.imread(sample_root + '/' + name, flag=0)
        #     tp = nd.expand_dims(nd.squeeze(tp, axis=2), axis=0)
        #     lb.append(tp)
        #
        # dd = nd.concat(*dt, dim=0)
        # ll = nd.concat(*lb, dim=0)
        # #
        # dd *= (dd != 255)
        # ll *= (ll != 255)

        for name in data_names:
            tp = cv2.imread(sample_root+'/'+name, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
            tp[tp == 255] = 0
            dt.append(tp)
        for name in label_names:
            tp = cv2.imread(sample_root+'/'+name, cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :]
            tp[tp == 255] = 0
            lb.append(tp)

        dd = np.concatenate(tuple(dt), axis=-1)
        ll = np.concatenate(tuple(lb), axis=0)

        gx = cv2.Sobel(dd, cv2.CV_16S, 1, 0, 5)
        gy = cv2.Sobel(dd, cv2.CV_16S, 0, 1, 5)
        gx = cv2.convertScaleAbs(gx)
        gy = cv2.convertScaleAbs(gy)
        g = np.sqrt(gx*gx + gy*gy)

        dd = np.concatenate((dd, g), axis=-1).transpose([2, 0, 1])

        dd = nd.array(dd)
        ll = nd.array(ll)
        return dd.astype(np.float32)/self.data_scale, ll.astype(np.float32)/self.data_scale

    def __len__(self):
        return self.count


def train():
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
    ctx = mx.gpu(0)
    batch_size = 16
    channel_in = 15
    channel_out = 3
    epochs = 50
    suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = 'mx_ModelUNet_1to3_grad_'
    sw = SummaryWriter('../nas/SRAD2018Model/summary/'+prefix+suffix, flush_secs=5)

    # data
    # gluon.data.vision.ImageFolderDataset()
    # gluon.data.DataLoader()
    # train_file_list = glob.glob('../nas/SRAD2018Model/train_val_files/train*.csv')
    train_file_list = glob.glob('../nas/SRAD2018Model/valid_data_files/train*.csv')

    loader = gluon.data.DataLoader(CustomDataSet(train_file_list[::-1], channel_in, channel_out), batch_size=batch_size, num_workers=2)
    # model
    model = ModelUNet(out_channels=channel_out)
    model.initialize(init=mx.init.Xavier(), ctx=ctx)
    # model.load_parameters('../nas/SRAD2018Model/weights/mx_ModelUNet_1to3_expweight_continue_epoch_20.param', ctx=ctx)
    # model.summary(nd.zeros(shape=(batch_size, channel_in, 501, 501), dtype=np.float32, ctx=ctx))

    # sw.add_graph(model)
    # loss
    # loss = gluon.loss.L2Loss()
    loss = custom_loss
    # trainer
    # mx.optimizer.Adam(learning_rate=0.001, )
    lr = mx.lr_scheduler.MultiFactorScheduler(step=[4000, 8000, 12000], factor=0.5)
    trainer = gluon.Trainer(model.collect_params(), optimizer='adam',
                            optimizer_params={'learning_rate': 0.005, 'wd': 1e-6, 'lr_scheduler': lr})
    # summary
    params = model.collect_params()
    grads = []
    for k in params.keys():
        if params[k].grad_req != 'null' and 'weight' in params[k].name:
            grads.append([k, params[k]])
    # train net
    step = 0
    for epoch in range(1, epochs):
        print('epoch: {}'.format(epoch))
        for batch_data, batch_label in tqdm(loader):
            step += 1
            batch_data = nd.array(batch_data, ctx=ctx)
            batch_label = nd.array(batch_label, ctx=ctx)
            # gluon.utils.split_and_load()
            with autograd.record():
                md_out = model(batch_data)
                ls = loss(md_out, batch_label).mean()
            ls.backward()
            trainer.step(1)
            # print('step: {}, loss: {}'.format(step, ls))
            # summary
            sw.add_scalar(tag='loss', value=ls.asscalar(), global_step=step)
            if step % 200 == 0:
                for i in range(channel_out):
                    sw.add_image(tag='pred_img{}'.format(i), image=nd.clip(md_out[0, i, :, :], 0.01, 0.99), global_step=step)
                    sw.add_image(tag='label_img{}'.format(i), image=nd.clip(batch_label[0, i, :, :], 0.01, 0.99), global_step=step)
                sw.add_histogram(tag='hist_pred', values=md_out, global_step=step, bins='auto')
                sw.add_histogram(tag='hist_label', values=batch_label, global_step=step, bins='auto')
                # for name, g in grads:
                #     sw.add_histogram(tag=name, values=g.grad(), global_step=step, bins='auto')
            sw.flush()

        model.save_parameters('../nas/SRAD2018Model/weights/{}epoch_{}.param'.format(prefix, epoch))
    sw.close()


if __name__ == "__main__":
    train()
