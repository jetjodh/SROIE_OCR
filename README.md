# SROIE_OCR
* To train a model run command like : ```python3 train.py --train_data train --valid_data test --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --lr 0.0005 --adam --data_filtering_off --PAD --num_iter 1000000 --batch_size 64 --imgW 224```
* For predicting on test set: ```python3 predict.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --model_path saved_models/TPS-ResNet-BiLSTM-Attn-Seed2020/best_accuracy.pth```

### Arguments
* `--train_data`: folder path to training lmdb dataset.
* `--valid_data`: folder path to validation lmdb dataset.
* `--select_data`: select training data. 
* `--batch_ratio`: assign ratio for each selected data in the batch.
* `--Transformation`: select Transformation module [None | TPS].
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet].
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM].
* `--Prediction`: select Prediction module [CTC | Attn].
* `--saved_model`: assign saved model to evaluation.

This repository is based on https://github.com/clovaai/deep-text-recognition-benchmark.
