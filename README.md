# Domain_Generalisation_Inner_Speech_Classification

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
