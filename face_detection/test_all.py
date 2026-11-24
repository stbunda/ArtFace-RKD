import cv2
import sys
import numpy as np
import datetime
import os
import glob
import argparse
from retinaface import RetinaFace
from pathlib import Path
from face_detection.face_align import norm_crop


def main(args):
    thresh = 0.8
    scales = [1024, 1980]

    count = 1

    gpuid = args.gpu
    detector = RetinaFace(args.model, 0, gpuid, 'net3')
    folder_path = Path(args.folder)
    source_path = folder_path / 'paintings'
    crop_path = folder_path / 'faces'
    detection_path = folder_path / 'detections'

    # Check if the folders already exist
    crop_path.mkdir(exist_ok=True)
    detection_path.mkdir(exist_ok=True) if args.show_crop else None

    jpg_files = source_path.glob("*.jp*g")
    for jpg in jpg_files:
        print(f'Looking for faces in {jpg.stem}')
        img = cv2.imread(str(jpg))
        print(img.shape)
        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        print('im_scale', im_scale)

        im_scales = [im_scale]
        flip = False

        for c in range(count):
            faces, landmarks = detector.detect(img,
                                               thresh,
                                               scales=im_scales,
                                               do_flip=flip)
            print(c, faces.shape, landmarks.shape)

        if faces is not None:
            print(f'found {faces.shape[0]} faces')
            for i in range(faces.shape[0]):
                print(f'{jpg.stem}-{i} - score: {faces[i][4]}')

                face_crop = norm_crop(img, landmarks[i])

                # Save the cropped face
                crop_filename = crop_path / f'{jpg.stem}-{i}.jpg'
                print('saving cropped face:', crop_filename)
                cv2.imwrite(str(crop_filename), face_crop)

                if args.show_crop:
                    # Draw rectangle on the original image
                    color = (0, 0, 255)
                    box = faces[i].astype(int)

                    # Crop the face using the box coordinates
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    if landmarks is not None:
                        landmark5 = landmarks[i].astype(int)
                        for l in range(landmark5.shape[0]):
                            color = (0, 0, 255)
                            if l == 0 or l == 3:
                                color = (0, 255, 0)
                            cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            if args.show_crop:
                detection_filename = detection_path / f'{jpg.stem}.jpg'
                print('writing', detection_filename)
                cv2.imwrite(str(detection_filename), img)
        else:
            print(f'Could not find any faces in {jpg.stem}!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int, help='device')
    parser.add_argument('--folder', default='../data', type=str, help='Folder with the faces')
    parser.add_argument('--model', default='./model/R50/R50', type=str, help='Face Detector model')
    parser.add_argument('--show_crop', action='store_true', help='Show cropped face image in detections/')

    arguments = parser.parse_args()
    print(f'arguments: {arguments}')
    main(arguments)