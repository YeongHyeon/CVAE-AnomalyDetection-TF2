import os, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf

import source.datamanager as dman
import source.neuralnet as nn
import source.tf_process as tfp

def main():

    # dataset = dman.Dataset(normalize=FLAGS.datnorm)
    # neuralnet = nn.CVAE(height=dataset.height, width=dataset.width, channel=dataset.channel, \
    #     z_dim=FLAGS.z_dim, leaning_rate=FLAGS.lr)
    neuralnet = nn.CVAE(height=28, width=28, channel=1, \
        ksize=FLAGS.ksize, z_dim=FLAGS.z_dim, leaning_rate=FLAGS.lr)

    # tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)
    # tfp.test(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, batch_size=FLAGS.batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=3, help='Kernel size')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()
