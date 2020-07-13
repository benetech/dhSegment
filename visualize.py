#!/usr/bin/env python

import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from imageio import imread, imsave
from tqdm import tqdm

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization, polygon_detection, bb_detection
import sys
import csv

# To output results in PAGE XML format (http://www.primaresearch.org/schema/PAGE/gts/pagecontent/2013-07-15/)
PAGE_XML_DIR = './page_xml'

#the colors corresponding to our classes
colors = [
    (255, 255, 255),
    (51, 102, 0),
    (6, 131, 255),
    (255, 189, 6),
    (255, 6, 6),
    (198, 6, 255),
    (134, 107, 46),
    (255, 148, 234),
    (128, 128, 128)
]

def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, kernel_size=5)
    return mask


def format_quad_to_string(quad):
    """
    Formats the corner points into a string.
    :param quad: coordinates of the quadrilateral
    :return:
    """
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


def main(args):
    print("args: ", args)
    #obtain the colors corresponding to the classes
    # colors = []
    # cat_file = open("../data/classes.txt","r")
    # for color in cat_file:
    #     print("color line: ",color )
    #     color = color.split(" ")
    #     colors.append((int(color[0]),int(color[1]),int(color[2])))
    # print(colors)

    #the model directory
    model_dir = args[1]
    

    input_files = glob(args[2] + "Algebra*")

    output_dir = args[3]
    print("output_dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # PAGE XML format output
    # output_pagexml_dir = os.path.join(output_dir, PAGE_XML_DIR)
    # os.makedirs(output_pagexml_dir, exist_ok=True)

    output = []
    # Store coordinates of page in a .txt file
    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')

        for filename in tqdm(input_files, desc='Processed files'):
            # For each image, predict each pixel's label
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            original_shape = prediction_outputs['original_shape']
            original_img = imread(filename, pilmode='RGB')

            #goes through each class and makes the bounding boxes for each one
            for cat in range(probs.shape[2]):
                print("\n category: ",cat)
                new_probs = probs[:, :, cat] 
                new_probs = new_probs / np.max(new_probs)  # Normalize to be in [0, 1]

                # Binarize the predictions
                page_bin = page_make_binary_mask(new_probs, threshold=0.5)

                # # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
                bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                        tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

                #find the bounding boxes
                # pred_page_coords = polygon_detection.find_polygonal_regions(bin_upscaled)
                pred_page_coords = bb_detection.find_bounding_boxes(bin_upscaled,min_area=0.001)

                print('ppc\n\n\n',pred_page_coords)
                # Draw page box on original image and export it. Add also box coordinates to the txt file
                
                #if it has actually found boxes, draw each box onto the page in the color of the category
                if pred_page_coords != [] and pred_page_coords != None:
                    for box in pred_page_coords:
                        # cv2.rectangle(original_img, (box[0][0],box[0][1]),(box[1][0],box[1][1]),colors[cat],5)
                        cv2.polylines(original_img,[box],True,colors[cat], 5)
                        # Write the points to a text file
                        # txt_coordinates += '{},{},{},{},\n'.format(filename,box[0],box[1],cat)
                        output.append([filename,box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1],cat])
                        # Create page region and XML file
                        # page_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(box[:, None, :]))
                else:
                    print('No box found in {}'.format(filename))

            basename = os.path.basename(filename).split('.')[0]
            imsave(os.path.join(output_dir, '{}_boxes.png'.format(basename)), original_img)

            # page_xml = PAGE.Page(image_filename=filename, image_width=original_shape[1], image_height=original_shape[0],
            #                     page_border=page_border)
            # xml_filename = os.path.join(output_pagexml_dir, '{}.xml'.format(basename))
            # page_xml.write_to_file(xml_filename, creator_name='PageExtractor')
    with open(os.path.join(output_dir,"pages.csv"),'w') as f:
        csv_writer = csv.writer(f)
        for row in output:
            csv_writer.writerow(row)






if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 visualize.py <model_directory> <image_directory> <output_directory>")
        exit()
    main(sys.argv)

    