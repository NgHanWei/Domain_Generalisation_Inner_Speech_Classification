# Domain_Generalisation_Inner_Speech_Classification

Selective Learning with Domain Generalization for Inner Speech Classification. A variational autoencoder which utilizes spatial and temporal filters is used to learn latent features of the EEG signal. Comparative contrastive loss is subsequently used to perform selective learning whereby data with more generalized features are favorably selected for training the baseline classifier. The same variational autoencoder serves a dual purpose of performing domain generalization on the selected data, which is then used to train the baseline inner speech classification model.

In subject-adaptive transfer learning, the same variational autoencoder is finetuned to learn the subject-specific domain features. Domain generalization is then applied on the adaptive data before performing model finetuning on the classficiation model. Layer freezing is used to increase the computational speed of the model finetuning.

## Results Overview

| Subject     | EEGNet | DeepConvNet | Proposed Method (EEGNet) | **Proposed Method (DeepConvNet)** |
|-------------|--------|-------------|--------------------------|-------------------------------|
| 1           | 27.33  | 24.67       | 31.33                    | 30.00                         |
| 2           | 21.58  | 27.89       | 24.21                    | 27.37                         |
| 3           | 32.31  | 26.15       | 26.15                    | 33.85                         |
| 4           | 26.84  | 21.05       | 25.79                    | 27.89                         |
| 5           | 20.53  | 25.26       | 28.95                    | 27.37                         |
| 6           | 30.12  | 25.30       | 29.52                    | 27.11                         |
| 7           | 26.84  | 23.16       | 23.16                    | 25.26                         |
| 8           | 28.67  | 26.67       | 26.00                    | 29.33                         |
| 9           | 27.37  | 21.58       | 25.79                    | 33.16                         |
| 10          | 25.26  | 23.68       | 31.05                    | 27.89                         |
| **Average** | 26.68  | 24.40       | 27.20                    | 28.92                         |

## Resources

Inner Speech Dataset: [Link](https://openneuro.org/datasets/ds003626/versions/2.1.2)

Original Inner Speech Github: [Link](https://github.com/N-Nieto/Inner_Speech_Dataset)

## Dependencies

## Run

Download inner speech dataset from the resources above and place in the same directory as the main folder.

### Selective Learning and Domain Generalization for Baseline

To perform selective learning and domain generalization.
```
usage: python predict_adapt.py [--eegnet] [--dgtrain] [--dgval] [--dgtest] [-coeff COEFF] [-subj SUBJ]

Performs selective learning and domain generalization on the selected data which is subsequently used to train the classifier.

Arguments:
    --eegnet                            Set the classifier model to EEGNet. Default uses DeepConvNet.
    --dgtrain                           Perform domain generalization on the training data.
    --dgval                             Perform domain generalization on the validation data.
    --dgtest                            Perform domain generalization on the evaluation data.
    -coeff COEFF                        Set coefficient threshold for selective learning against learnt general features. Setting to 0 turns off selective learning.
    -subj SUBJ                          Set subject to perform transfer learning adaptation on.
```

To recreate domain generalization on training and test data with selective learning on subject 1 using DeepConvNet, run:
```
python predict_adapt.py --dgtrain --dgtest --coeff 1 -subj 1
```

### Subject-Adaptive Transfer Learning and Domain Generalization
To perform domain generalization on adaptation data followed by subject-adaptive transfer learning.
```
usage: python train_adapt.py [MODELPATH] [OUTPATH] [--eegnet] [--dgtrain] [--dgval] [--dgtest] [-scheme SCHEME] [-trfrate TRFRATE] [-lr LR] [-gpu GPU]

Performs subject-adaptive transfer learning on a subject with or without meta-learning.

Positional Arguments:
    MODELPATH                           Path to folder containing the baseline models for adaptation
    OUTPATH                             Path to folder for saving the adaptation results in

Optional Arguments:
    --eegnet                            Set the classifier model to EEGNet. Default uses DeepConvNet.
    --dgtrain                           Perform domain generalization on the training data.
    --dgval                             Perform domain generalization on the validation data.
    --dgtest                            Perform domain generalization on the evaluation data.
    -scheme SCHEME                      Set scheme which determines layers of the model to be frozen
    -trfrate TRFRATE                    Set amount of target subject data to be used for subject-adaptive transfer learning
    -lr LR                              Set the learning rate of the transfer learning
    -gpu GPU                            Set gpu to use, default is 0
```

To recreate domain generalization and subject-adaptive transfer learning using DeepConvNet:
```
train_adapt.py MODELPATH OUTPATH -scheme 4 -trfrate 100 --dgtrain
```
