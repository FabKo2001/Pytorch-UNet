import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from utils.data_loading import BasicDataset
from unet import UNet


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        if net.n_classes == 1:
            probs = torch.sigmoid(output)
        else:
            probs = F.softmax(output, dim=1)

    return probs.squeeze(0).numpy()  # Shape: [C, H, W]


def save_probability_map(probabilities, out_filename_prefix):
    """
    Speichert für jede Klasse eine Heatmap der Wahrscheinlichkeiten.
    """
    num_classes, height, width = probabilities.shape
    for c in range(num_classes):
        plt.imshow(probabilities[c], cmap='viridis')
        plt.colorbar()
        plt.title(f'Class {c} Probability Map')
        plt.axis('off')
        plt.savefig(f'{out_filename_prefix}_class{c}_prob.png')
        plt.close()


def get_args():
    parser = argparse.ArgumentParser(description='Predict probability maps from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output maps')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)

    # Entferne unerwarteten Schlüssel
    if 'mask_values' in state_dict:
        del state_dict['mask_values']

    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        probs = predict_img(net=net,
                            full_img=img,
                            scale_factor=args.scale,
                            device=device)

        if not args.no_save:
            out_prefix = os.path.splitext(out_files[i])[0]
            save_probability_map(probs, out_prefix)
            logging.info(f'Probability maps saved to {out_prefix}_classX_prob.png')

        if args.viz:
            logging.info(f'Visualizing Class 1 probability map for image {filename}, close to continue...')
            if probs.shape[0] > 1:
                class_index = 1
            else:
                class_index = 0  # falls nur 1 Klasse existiert (Binary Case)

            probs[class_index]=abs(probs[class_index]-0.5)*2

            plt.imshow(probs[class_index], cmap='PiYG')
            plt.title('Confidence')
            plt.colorbar()
            plt.axis('off')
            plt.show()
