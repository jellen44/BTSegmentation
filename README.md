# Brain Tumor Segmentation using a 2D UNet

![tumor_tile](./graphics/tumor_tile.jpg)

This project demonstrates the implementation of a 2D UNet Convolution Neural Network to segment regions of High-Grade Glioma brain tumors. The model implemented is based on this [paper](https://arxiv.org/abs/1505.04597).  The model was trained using data from the 2015 MICCAI BRaTS Challenge.

![unet_v0.2_pat206](./outputs/unet_v0.2_pat206.gif)

## Background / Overview

### What is Brain Tumor Segmentation?

Glioma tumors, the most common type of brain tumor, significantly reduce life expectancy in their high-grade form. While low grade gliomas (LGG) are usually removable surgically with a promising survival rate, high grade gliomas (HGG) are much more invasive. The segmentation of these HGG tumors aims to differentiate tissue regions, including regions of active tumor, necrosis, edema (swelling around a tumor) and non-tumor, in order to inform treatment. The segmentation process consists of going through each MRI slice and classifying each 3-D pixel (voxel) as a different tissue type. While segmentation is generally performed manually by radiologists, it is a highly laborious task that requires significant technical experience. Further, since HGG tumors have more undefined and irregular boundaries, segmenting them can provide additional challenges. Thus, an effective automatic segmentation method could provide a much more efficient alternative, saving radiologists and patients valuable time.

### Convolutional Neural Networks for Brain Tumor Segmentation

Since 2013, the Perelman School of Medicine at the University of Pennsylvania has been hosting the Multimodal Brain Tumor Segmentation Challenge (BraTS), aimed at developing algorithms to automatically segment gliomas. The current state-of-the-art method for automatic brain tumor segmentation uses different forms of a deep learning algorithm called a convolutional neural network (CNN). CNNs are commonly used due to their ability to 'learn' highly non-linear functions by fine-tuning millions of weights in the network. Specifically, they are often applied to image-processing tasks because they can extract high-level features, such as edges and orientations, in a hierarchical manner. CNN-based approaches to brain tumor segmentation vary in a number of features, including dimensionality, preprocessing techniques, input structure and the order and structure of the CNN layers. 

### Our UNet Approach

Our project explores a newer type of CNN-based deep learning algorithm called a UNet. Introduced in May 2015 by Olaf Ronneberger and a team of researchers from the University of Freiburg in Germany, the architecture immediately stood out compared to other architectures due to how well it performed in a number of different biomedical segmentation challenges. The network is made up of a contracting (encoder) path that reduces the dimensionality of the input and a subsequent expansive (decoder) path that increases it. One key feature of the UNet architecture is the incorporation of skip connections, which allow for the concatenation of features directly from the contracting to the expansive path and play a crucial role in restoring the spatial resolution lost from the input due to downsampling. Lastly, the fully convolutional nature (no dense, fully connected layers) of the UNet allows for variably-sized inputs, while the use of transposed convolutions in the decoder path enables precise localization of features. 

We first attempted this segmentation task by using a multi-pathway CNN where we first needed to create many smaller patches of pixels as input to the network to classify one at a time (the network would classify the central pixel of each patch). However, a UNet can classify slice by slice instead of patch by patch, which makes the UNet much more computationally efficient.


### Data Structure

![brain_grid.jpg](./graphics/brain_grid.jpg)


All MRI scans of the brain used to validate our model were provided by the BRATS 2015 challenge database. This dataset consists of 220 HGG cases and 54 LGG cases. For the purposes of our experimentation, only HGG cases were utilized. Each MRI scan consists of 155 2D slices in four different modalities: T1, T1 with contrast, T2 and FLAIR. Thus, these four modalities sum to a total of 620 MRI slices for each patient. Further, each patient has a fifth image providing the 'ground truth' labels for each pixel. In this dataset, the voxel labels are as follows: '0' is non-tumor; '1' is necrosis; '2' is edema; '3' is non-enhancing tumor; '4' is enhancing tumor. There is a label for every pixel in each 240x240 voxel slice, generating 8,928,000 labels for each patient, and approximately 2 billion labels (1.96x109) for the 220 HGG cases overall. These ground truth segmentation labels are manually provided by radiologists.


![label_diagram](./graphics/label_diagram.jpg)


## Installation

### Requirements

* GNU Make
* Python 3.7.9
* Tensorflow 2.4.0
* Keras 2.4.0
* CUDA 11.0
* scitkit-learn
* scikit-image
* SimpleITK
* tqdm

### Setup

1. Install [Tensorflow](tensorflow.org), it is recommended that you have version 2.4.0+
2. Install [Keras](keras.io), it is recommended  that you have version 2.4.0+
3. Install [CUDA](https://www.tensorflow.org/install/gpu) and configure GPU support for tensorflow. It is recommended that you have version 11.0+
4. Clone the project
```
git clone https://github.com/toehmler/nunet.git
```
5. Use pip to install all the dependencies
```
pip3 install -r requirements.txt
```
6. Download the dataset from the [MICCAI BRaTS Website](https://www.med.upenn.edu/cbica/brats2020/data.html) 
7. Run the preprocessing script to perform N4ITK bias correction and rename the patient directories. Bias correction is only performed on the t1 and t1c modalities. The corrected scans are saved with the `_n4` suffix. Set `n4itk=False` to skip bias correction. (Note: the current script expects `.mha` files)
```
python3 process.py <path_to_data> <n4itk=False>
```
8. Set `path_to_data`  in `config.ini`  to be the full path to the downloaded dataset.

## Usage 

The project relies on the values in `config.ini` to operate properly. Updating these values will change which model is being trained, tested or used for predictions. Additional parameters for training can be found here as well (batch size, validation split, etc.).  See [Configuration Options](#Configuration-Options) for a more detailed explanation of each option.

### Quick Start

To perform the entire pipeline on a new model from scratch (compile, train and test), perform the following steps:

1. Make sure `path_to_data` in `config.ini` is set as the full path to the data
2. Set `model_name` and `version_no` in `config.ini` to be the name and version number that new model willbe saved as. (Keeping the default will load a pre-trained model)
3. Update any of the training and testing options as needed
4. Run the pipeline using these options by simply running `make`. 

This will compile a new model and save a summary and its architecture in a new directory in `models/` under the name and version number given in the configuration file. The model will be trained and the results of the testing script will be saved as `.csv` in this directory. Due to the large file size of the MRI scans and limitations on RAM, training is done in groups. See [Training](#Training) for more information.

### Available Commands
* `make`: Runs the entire pipeline (see [Quick Start](#Quick-Start)). 
* `make train`: Runs the training script for the model specified in `config.ini`. Compiles a new model if one is not found using the given name and version number in `config.ini`, otherwise loads the model and resumes training. A training log is kept in the model's directory and training params are appended to this log on each run. See [Training](#Training) for more information.
* `make test`: Runs the test script, saving the result as a `.csv` in the model's directory. Assumes the model specified in `config.ini` can be found in `/models`. See [Testing](#Testing) for more information on the metrics used.
* `make predict`: Runs the prediction script, will show prompt for input of patient number to predict. Creates prediction images using t1c modality as background. Saves predictions as `.png` files and creates a `.gif` from these and saves these outputs in the directory specified as `image_out_path` in `config.ini`.

### Configuration Options

**General**
* `path_to_data` : Path to dataset in full, assumes patient directories have been renamed in the format `pat{x}`.
* `image_out_path` : Path to save prediction images in full.

**Model**
* `name` : Name of the current model being trained, tested or used for predictions. 
* `ver` : The version number of the current model being trained, tested, or used for predictions
If a model of a given name and version already exists when training, that model is loaded, otherwise a new model is compiled.

**Training**   
* `start` : The index of the patient to start training with
* `end` : The index of the patient to end training with
* `interval` : How many patients to train in each group (if not an even divisor of the total number the last group will be the size of the remainder)
* `epochs` : Number of epochs to train for *per group*
* `batch_size` : The batch size to train the model using *per group*   
* `validation_split` : The split to use to create the validation data (0-1), done on *per group* basis

**Testing**
* `start` : The start index of the patient to test on
* `end` : The end index of the patient to test on


## Training and Testing

We trained our U-Net from scratch with a weighted Dice-coefficient loss function and an 'Adam' optimizer. The model was training 30 epochs at a time, with a batch size of 32 samples and a 75/25 training/validation split. The learning rate controls how much the parameters of a U-Net are changed for each iteration, so to balance accuracy and time, our model uses a learning rate of .0001. Layers of the U-Net used the rectified linear unit (ReLU) activation function, with the final output layer using the SoftMax function.

The model was trained on 40 patients at a time, which was repeated five times for a total of 200 training examples. The trained model was subsequently tested on the remaining 20 files (90/10 test/train split) that were not trained on. Models were primarily evaluated based on the dice (F1) score (the harmonic mean of sensitvity and precision), as is standard for BRATS competition data. Separate Dice coefficients can be generated for the whole tumor, the enhancing tumor and the core. Sensitivity and specificity were also calculated for each class. Definitions of these metrics are shown below:

Dice Coefficient (X,Y) = (2*|X ∩ Y|)/|X|+|Y|

Sensitivity Score (X,Y): |Xp ∩ Yp|/Yp

Specificity Score (X,Y): |Xn ∩ Yn|/Yn

where X is defined as all of our model’s tissue class predictions and Y as the ground truth labels, with Xp and Yp representing positive predictions, and Xn and Yn representing negative predictions. Thus, the calculated dice score gives us an overall picture of how accurate our class predictions are, while sensitivity and specificity can be used to ascertain whether certain tissue labels are being overpredicted or underpredicted by our model.


## Results

Results on each of the 20 test set files are shown in the table below. Averages are presented in the bottom row.

|   Patient Number  |   dice_whole  |   dice_enhancing  |   dice_core  |   sen_whole  |   sen_enhancing  |   sen_score  |   spec_whole  |   spec_enhancing  |   spec_core  |
|-------------------|---------------|-------------------|--------------|--------------|------------------|--------------|---------------|-------------------|--------------|
|   1               |   0.907       |   0.860           |   0.848      |   0.957      |   0.932          |   0.976      |   0.998       |   0.999           |   0.998      |
|   2               |   0.819       |   0.790           |   0.800      |   0.911      |   0.970          |   0.913      |   0.995       |   0.998           |   0.998      |
|   3               |   0.832       |   0.859           |   0.746      |   0.766      |   0.833          |   0.767      |   0.999       |   1.000           |   0.999      |
|   4               |   0.906       |   0.911           |   0.890      |   0.946      |   0.912          |   0.922      |   0.998       |   1.000           |   1.000      |
|   5               |   0.938       |   0.899           |   0.865      |   0.959      |   0.832          |   0.786      |   0.999       |   1.000           |   1.000      |
|   6               |   0.815       |   1.000           |   0.266      |   0.770      |   1.000          |   0.162      |   1.000       |   1.000           |   1.000      |
|   7               |   0.943       |   0.908           |   0.933      |   0.959      |   0.866          |   0.950      |   0.998       |   1.000           |   0.999      |
|   8               |   0.897       |   0.797           |   0.838      |   0.885      |   0.796          |   0.921      |   0.999       |   1.000           |   0.999      |
|   9               |   0.873       |   0.837           |   0.854      |   0.804      |   0.741          |   0.773      |   1.000       |   1.000           |   1.000      |
|   10              |   0.695       |   0.233           |   0.407      |   0.623      |   0.152          |   0.381      |   0.998       |   1.000           |   0.999      |
|   11              |   0.722       |   0.442           |   0.593      |   0.812      |   0.692          |   0.766      |   0.998       |   0.999           |   0.998      |
|   12              |   0.942       |   0.901           |   0.900      |   0.949      |   0.858          |   0.927      |   0.999       |   1.000           |   0.999      |
|   13              |   0.880       |   0.907           |   0.776      |   0.935      |   0.948          |   0.969      |   0.996       |   1.000           |   0.998      |
|   14              |   0.917       |   0.816           |   0.826      |   0.933      |   0.739          |   0.778      |   0.997       |   0.999           |   0.999      |
|   15              |   0.748       |   0.898           |   0.771      |   0.624      |   0.865          |   0.654      |   1.000       |   1.000           |   1.000      |
|   16              |   0.930       |   0.900           |   0.922      |   0.965      |   0.845          |   0.935      |   0.998       |   1.000           |   0.999      |
|   17              |   0.942       |   0.902           |   0.868      |   0.929      |   0.969          |   0.830      |   0.999       |   0.999           |   0.999      |
|   18              |   0.786       |   0.613           |   0.515      |   0.935      |   0.915          |   0.865      |   0.994       |   0.999           |   0.998      |
|   19              |   0.866       |   0.810           |   0.797      |   0.851      |   0.719          |   0.748      |   0.999       |   1.000           |   1.000      |
|   20              |   0.502       |   0.743           |   0.718      |   0.400      |   0.751          |   0.643      |   0.999       |   1.000           |   1.000      |
|   Mean Values     |   0.843       |   0.801           |   0.757      |   0.846      |   0.817          |   0.783      |   0.998       |   1.000           |   0.999      |

