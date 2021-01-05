## train deeplabv1
python v1/train_deeplabv1.py --gpu 0,1,2 --config config/deeplabv1_voc12.yaml
## test on trained model
python v1/test_deeplabv1.py --gpu 0 --crf True --config config/deeplabv1_voc12.yaml