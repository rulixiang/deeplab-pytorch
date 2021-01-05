## train deeplabv1
python v1/train_deeplabv1.py --gpu 6,7 --config config/deeplabv1_voc12.yaml
## test on trained model
python v1/test_deeplabv1.py --gpu 7 --crf True --config config/deeplabv1_voc12.yaml