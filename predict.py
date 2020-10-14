"""
Script for inferencing on test images and creating prediction files
"""
import os
import argparse
import time
import math
import torch
import torch.nn.functional as F
from PIL import Image
from dataset import NormalizePAD
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter

device = torch.device("cpu")
OUTPUT_DIR = "results/"


def extract_bbox(txt_file) -> list:
    """
    Returning the bboxe coords for cropping images with text regions
    """
    bboxes = []
    with open(txt_file) as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split(",")
            try:
                x1 = int(elements[0])
                y1 = int(elements[1])
                x2 = int(elements[4])
                y2 = int(elements[5])
                bboxes.append([x1, y1, x2, y2])
            except:
                print("Coords not found")
    return bboxes


def transform(img_list, number_of_input_channel) -> list:
    """
    Transforms images for model input
    """
    transform = NormalizePAD((number_of_input_channel, opt.imgH, opt.imgW))
    transformed_images = []
    for image in img_list:
        w, h = image.size
        try:
            ratio = w / float(h)
            if math.ceil(opt.imgH * ratio) > opt.imgW:
                resized_w = opt.imgW
            else:
                resized_w = math.ceil(opt.imgH * ratio)
            resized_image = image.resize((resized_w, opt.imgH), Image.BICUBIC)
            transformed_images.append(transform(resized_image))
        except:
            print("Error by img crop")
    return transformed_images


def predict(img_path, model):
    """
    Predicts the text on image provided
    """
    # Setting up text file name
    txt = os.path.splitext(img_path.split("/")[-1])[0] + ".txt"
    if opt.rgb:
        img = Image.open(img_path).convert("RGB")  # for color image
    else:
        img = Image.open(img_path).convert("L")  # for grayscale image

    boxes = extract_bbox("gt/" + txt)
    images = []
    for box in boxes:
        images.append(img.crop((box[0], box[1], box[2], box[3])))

    infer_time = 0
    input_channel = 3 if img.mode == "RGB" else 1

    resized_images = transform(images, input_channel)

    image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0).to(device)
    batch_size = image_tensors.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = (
        torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    )

    start_time = time.time()
    if "CTC" in opt.Prediction:
        preds = model(image_tensors)
        forward_time = time.time() - start_time

        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)

    else:
        preds = model(image_tensors, text_for_pred, is_train=False)
        forward_time = time.time() - start_time

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    infer_time += forward_time

    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)

    fp = open(OUTPUT_DIR + txt, "w")

    for pred, pred_max_prob in zip(preds_str, preds_max_prob):
        if "Attn" in opt.Prediction:
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        print(f"{pred:25s}\t{confidence_score:0.4f}\t{forward_time:f}")
        for word in pred.split(" "):
            fp.write(f"{word:s}\n")
    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument("--rgb", action="store_true", help="use rgb input")
    parser.add_argument("--model_path", required=True, help="path of pre-trained model")
    parser.add_argument(
        "--batch_max_length", type=int, default=100, help="maximum-label-length"
    )
    parser.add_argument(
        "--character",
        type=str,
        default='" "\'~qazwsxedcrfvtgbyhnujmik\\,ol`.<>·pQAZWSXEDCRFVTGBYHNUJMIKOLP;^[]_/?{}|!-+=:$@#%*&)(0123456789',
        help="maximum-label-length",
    )
    # Model Architecture
    parser.add_argument(
        "--Transformation",
        type=str,
        required=True,
        help="Transformation stage. None|TPS",
    )
    parser.add_argument(
        "--FeatureExtraction",
        type=str,
        required=True,
        help="FeatureExtraction stage. VGG|RCNN|ResNet",
    )
    parser.add_argument(
        "--SequenceModeling",
        type=str,
        required=True,
        help="SequenceModeling stage. None|BiLSTM",
    )
    parser.add_argument(
        "--Prediction", type=str, required=True, help="Prediction stage. CTC|Attn"
    )
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=1,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    opt = parser.parse_args()

    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    model.load_state_dict(torch.load(opt.model_path))
    model.to(device)
    model.eval()

    for img_path in os.listdir("task1_2_test(361p)"):
        predict("task1_2_test(361p)/" + img_path, model)
