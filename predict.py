import configparser
from keras.models import load_model
from model import *
from data import *
from metrics import dice_coef, dice_coef_loss
from glob import glob
import numpy as np
import matplotliby.pyplot as plt






if __name__ == "__main__":
    patient_no = input("Patient no: ")

    # parse config file

    config = configparser.ConfigParser()
    config.read('config.ini')
    model_str = "{}_v{}".format(config['model']['name'], config['model']['ver'])
    model_dir = "models/{}".format(model_str)
    lr = 1e-4

    # load and compile model
    model = load_model("{}/{}".format(model_dir, model_str),
                       custom_objects={"dice_coef": dice_coef,
                                       "dice_coef_loss": dice_coef_loss})
    model.load_weights("{}/{}_w".format(model_dir, model_str))
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    # load and normalize patient scans
    root = config['general']['root_path']
    out_path = config['general']['image_out_path']
    path_str = "{}/*pat{}*".format(root, patient_no)
    path = glob(path_str)[1]
    scans = norm_scans(load_scans(path))
    pbar = tqdm(total = scans.shape[0])

    for slice_no in range(scans.shape[0]):
        test_slice = scans[slice_no:slice_no+1,:,:,:4]
        test_label = scans[slice_no:slice_no+1,:,:,4]
        prediction = model.predict(test_slice, batch_size=32)[0]
        prediction = np.around(prediction)
        prediction = np.argmax(prediction, axis=-1)
        gt.extend(test_label[0])
        pred.extend(prediction)


        im = plt.figure(figsize=(15,10))
        im = plt.subplot(131)
        im = plt.title('Input')
        im = plt.imshow(scan, cmap='gray')
        im = plt.subplot(132)
        im = plt.title('Ground Truth')
        im = plt.imshow(label, cmap='gray')
        im = plt.subplot(133)
        im = plt.title('Prediction')
        im = plt.imshow(prediction, cmap='gray')
        plt.savefig('outputs/{}_pat{}_slice{}.png'.format(model_str, patient_no, slice_no), bbox_inches='tight')
        plt.close(im)
        pbar.update(1)


