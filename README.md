# Domain_Generalisation_Inner_Speech_Classification

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

## Dependencies

## Run

Perform Data Selection and Domain Generalisation/Adaptation

python predict_adapt.py --dgtrain --dgtest --coeff 1 -subj

For no selective learning, set coeff to 0

Settings:
Perform DG on the data
--dgtrain
--dgval
--dgtest
Set threshold
--coeff
Set Subject
-subj

python train_adapt.py -scheme -lr

To recreate, scheme 4 and transfer rate 100%

Settings:
Perform DG on the data
--dgtrain
--dgval
--dgtest
