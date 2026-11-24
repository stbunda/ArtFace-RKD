import argparse

import cv2
import numpy as np
import torch
import pandas as pd

from backbones import get_model

from pathlib import Path

@torch.no_grad()
def inference(weight, name, img, fp16):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=fp16)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    print('img: ', name)
    return feat

def collect_features(root_folder, weight, name, fp16):
    root_folder = Path(root_folder)
    records = []

    for subfolder in root_folder.iterdir():
        if subfolder.is_dir():
            for img_path in subfolder.glob("*.jp*g"):
                print(f"Processing {img_path}")
                feat = inference(weight, name, img_path, fp16)
                record = {
                    'folder': subfolder.name,
                    'filename': img_path.stem,
                    'feature': feat
                }
                records.append(record)

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='glint360k_cosface_r100/backbone.pth')
    parser.add_argument('--folder', type=str, default='sitters')
    parser.add_argument('--fp16', default=True, type=bool, help='')
    args = parser.parse_args()

    # Collect features
    df = collect_features(args.folder, args.weight, args.network, fp16=args.fp16)

    # Save to CSV (feature will be a long string)
    df.to_csv('features.csv', index=False, sep=';')
    df.to_pickle('features.pkl')
