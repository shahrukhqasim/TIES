import numpy as np
import configparser as cp
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import os
import torch
from torch.autograd import Variable
import cv2
import random
from conv_autoencoder.Autoencoder import Autoencoder
import sys


class DataLoader(implements(LoadInterface)):
    def __init__(self, num_crops):
        self.num_crops = num_crops
        self.K = cv2.getGaussianKernel(5, 5)

    def load_datum(self, full_path):
        image = cv2.imread(os.path.join(full_path, 'image.png'), 1)
        height, width, _ = np.shape(image)

        patches_x = np.zeros((self.num_crops, 3, 512, 512)).astype(np.float32)
        patches_y = np.zeros((self.num_crops, 3, 512, 512)).astype(np.float32)

        for j in range(self.num_crops):
            crop_width = random.randint(100, width)
            crop_height = random.randint(100, height)
            x = random.randint(0, width - crop_width)
            y = random.randint(0, height - crop_height)
            patch = image[y:y + crop_height, x:x + crop_width, :]
            patch_x = cv2.resize(patch, dsize=(512, 512))
            patch_y = cv2.filter2D(patch_x, -1, self.K)

            patches_x[j] = np.swapaxes(patch_x, 0, 2).astype(np.float32)
            patches_y[j] = np.swapaxes(patch_y, 0, 2).astype(np.float32)

        datum = dict()
        datum['X'] = patches_x
        datum['Y'] = patches_x

        return datum


class ConvolutionalAutoencoder:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['conv_auto_encoder']['train_data_path']
        self.test_path = config['conv_auto_encoder']['test_data_path']
        self.validation_data_path = config['conv_auto_encoder']['validation_data_path']
        self.learning_rate = float(config['conv_auto_encoder']['learning_rate'])
        self.from_scratch = int(config['conv_auto_encoder']['from_scratch']) == 1
        self.model_path = config['conv_auto_encoder']['model_path']
        self.save_after = int(config['conv_auto_encoder']['save_after'])
        self.batch_size = int(config['conv_auto_encoder']['batch_size'])
        self.manual_mode_loaded = False

    def train(self):
        assert not self.manual_mode_loaded
        dataset = FolderDataReader(self.train_path, DataLoader(self.batch_size))
        dataset.init()
        model = Autoencoder().cuda()

        if not self.from_scratch:
            print("Loaded")
            model.load_state_dict(torch.load(self.model_path))

        criterion = torch.nn.MSELoss(size_average=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for i in range(1000000):
            # Save model
            if (i % self.save_after) == 0:
                print("Saving model")
                torch.save(model.state_dict(), self.model_path)

            document, epoch, id = dataset.next_element()
            input_image = Variable(torch.from_numpy(document['X'])).cuda()
            expected_output = Variable(torch.from_numpy(document['Y'])).cuda()

            given_output = model(input_image)
            loss = criterion(given_output, expected_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("%5d Loss = %f" % (i, loss.data[0]))

    def prepare_for_manual_testing(self):
        self.manual_model = Autoencoder().cuda()
        self.manual_model .load_state_dict(torch.load(self.model_path))
        self.manual_mode_loaded = True


    # image: Numpy array (N,N,3)
    def get_feature_map(self, image):
        assert self.manual_mode_loaded
        patches_x = np.zeros((1, 3, 512, 512)).astype(np.float32)

        patch_x = cv2.resize(image, dsize=(512, 512))

        patches_x[0] = np.swapaxes(patch_x, 0, 2).astype(np.float32)

        input_image = Variable(torch.from_numpy(patches_x)).cuda()

        given_output = self.manual_model.encoder(input_image)

        return np.swapaxes((given_output.cpu().data.numpy())[0], 0, 2)

    def test(self):
        assert not self.manual_mode_loaded
        dataset = FolderDataReader(self.validation_data_path, DataLoader(1))
        dataset.init()
        model = Autoencoder().cuda()
        model.load_state_dict(torch.load(self.model_path))
        for i in range(1000000):
            document, epoch, id = dataset.next_element()
            input_image = Variable(torch.from_numpy(document['X'])).cuda()
            given_output = model(input_image)

            original = np.swapaxes(document['Y'][0], 0, 2).astype(np.uint8)
            result = np.swapaxes((given_output.cpu().data.numpy())[0], 0, 2)
            result = result / np.max(result)
            cv2.imshow('a', original)
            cv2.imshow('b', result)
            cv2.waitKey(0)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error")
        sys.exit()

    arg = sys.argv[1]

    if arg == 'train':
        train = True
    else:
        train = False

    trainer = ConvolutionalAutoencoder()
    if train:
        trainer.train()
    else:
        trainer.test()
