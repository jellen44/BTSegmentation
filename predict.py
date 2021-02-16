import configparser
from keras.models import load_model
from model import *
from data import *
from metrics import dice_coef, dice_coef_loss
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage import color, img_as_float
import imageio


def save_mask(bg, mask, patient_no, out_path, model_str, slice_no):
    ones = np.argwhere(mask == 1)
    twos = np.argwhere(mask == 2)
    threes = np.argwhere(mask == 3)
    fours = np.argwhere(mask == 4)

    # scale intensity between 0-1
    bg = (bg - np.min(bg)) / (np.max(bg) - np.min(bg))
#    bg = rescale_intensity(bg, in_range=(0,1))
    bg = adjust_gamma(color.gray2rgb(bg), 1)
    bg_copy = bg.copy()

    red = [1, 0.2, 0.2]
    yellow = [1, 1, 0.25]
    green = [0.35, 0.75, 0.25]
    blue = [0, 0.25, 0.9]

    fig = plt.figure(figsize=(3,3),
                     dpi=80,
                     frameon=False,
                     edgecolor='black')
    for i in range(ones.shape[0]):
        bg_copy[ones[i][0]][ones[i][1]] = red
    for i in range(twos.shape[0]):
        bg_copy[twos[i][0]][twos[i][1]] = green 
    for i in range(threes.shape[0]):
        bg_copy[threes[i][0]][threes[i][1]] = blue 
    for i in range(fours.shape[0]):
        bg_copy[fours[i][0]][fours[i][1]] = yellow
    fig.figimage(bg_copy)
    #plt.imshow(bg_copy)
    plt.savefig('{}/{}_pat{}_slice{}.png'.format(out_path, model_str, patient_no, slice_no))
    plt.close(fig)

def save_dual_mask(bg, pred, label, pat_no, out_path, model_str, slice_no):
    ones = np.argwhere(pred == 1)
    twos = np.argwhere(pred == 2)
    threes = np.argwhere(pred == 3)
    fours = np.argwhere(pred == 4)

    gt_ones = np.argwhere(label == 1)
    gt_twos = np.argwhere(label == 2)
    gt_threes = np.argwhere(label == 3)
    gt_fours = np.argwhere(label == 4)

    bg = (bg - np.min(bg)) / (np.max(bg) - np.min(bg))
    bg = adjust_gamma(color.gray2rgb(bg), 1)
    bg_copy = bg.copy()
    truth_bg = bg.copy()

    '''
    ones_color = [1,0.2,0.2] # red
    twos_color = [1,1,0.25] # yellow
    threes_color = [0.35,0.75,0.25] # green
    fours_color = [0,0.25,0.9] # blue
    '''

    fours_color = [42/255,152/255,143/255]
    twos_color = [233/255,196/255,106/255]
    threes_color = [244/255,162/255,97/255]
    ones_color = [231/255,111/255,81/255]

    fig = plt.figure(figsize=(6,3),dpi=80,edgecolor='black',frameon=False)

    for i in range(gt_ones.shape[0]):
        truth_bg[gt_ones[i][0]][gt_ones[i][1]] = ones_color
    for i in range(gt_twos.shape[0]):
        truth_bg[gt_twos[i][0]][gt_twos[i][1]] = twos_color
    for i in range(gt_threes.shape[0]):
        truth_bg[gt_threes[i][0]][gt_threes[i][1]] = threes_color
    for i in range(gt_fours.shape[0]):
        truth_bg[gt_fours[i][0]][gt_fours[i][1]] = fours_color
    truth_im = fig.figimage(truth_bg)
    for i in range(ones.shape[0]):
        bg_copy[ones[i][0]][ones[i][1]] = ones_color
    for i in range(twos.shape[0]):
        bg_copy[twos[i][0]][twos[i][1]] = twos_color
    for i in range(threes.shape[0]):
        bg_copy[threes[i][0]][threes[i][1]] = threes_color
    for i in range(fours.shape[0]):
        bg_copy[fours[i][0]][fours[i][1]] = fours_color
    pred_im = fig.figimage(bg_copy, xo=240)
    test1 = plt.subplot(121)
    gt_title = plt.title('Ground Truth', loc='left')
    test2 = plt.subplot(122)
    pred_title = plt.title('Prediction')


    plt.setp(gt_title, color="w")
    plt.setp(pred_title, color="w")

    truth_im.set_zorder(0)
    pred_im.set_zorder(0)
    test1.set_zorder(1)
    test2.set_zorder(1)
    test1.patch.set_visible(False)
    test2.patch.set_visible(False)
    gt_title.set_zorder(1)
    pred_title.set_zorder(1)

    plt.savefig('{}/{}_pat{}_slice{}_dual.png'.format(out_path, model_str, patient_no, slice_no))
    plt.close(fig)


def create_gif(out_path, model_str, patient_no):
    img_list = []
    for i in range(155):
        image = glob('{}/{}_pat{}_slice{}_dual.png'.format(out_path, model_str, patient_no, i))[0]
        img_list.append(imageio.imread(image))
    imageio.mimwrite('{}/{}_pat{}.gif'.format(out_path, model_str, patient_no), img_list)

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
    path = glob(path_str)[0]
    scans = norm_scans(load_scans(path))
    pbar = tqdm(total = scans.shape[0])
    # predict each slice and generate image from each prediction 
    for slice_no in range(scans.shape[0]):
        test_slice = scans[slice_no:slice_no+1,:,:,:4]
        test_label = scans[slice_no:slice_no+1,:,:,4]
        prediction = model.predict(test_slice, batch_size=32)[0]
        prediction = np.around(prediction)
        prediction = np.argmax(prediction, axis=-1)
        # [flair, t1, t1c, t2, gt]
        scan = test_slice[0,:,:,2] 
        save_dual_mask(scan, prediction, test_label[0], patient_no, out_path, model_str, slice_no)
        '''
        label = test_label[0]
        im = plt.figure(figsize=(15,10))
        plt.subplot(131)
        plt.title('Input')
        plt.imshow(scan, cmap='gray')
        plt.subplot(132)
        plt.title('Ground Truth')
        plt.imshow(label, cmap='gray')
        plt.subplot(133)
        plt.title('Prediction')
        plt.imshow(prediction, cmap='gray')
        plt.savefig('{}/{}_pat{}_slice{}.png'.format(out_path, model_str, patient_no, slice_no), bbox_inches='tight')
        plt.close(im)
        '''
        pbar.update(1)
    pbar.close()

    create_gif(out_path, model_str, patient_no)


