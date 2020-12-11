from pathlib import Path
import shutil
import math
import json

from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from rasterio.transform import xy

from mmdet.datasets.pipelines import Compose
from mmdet.ops import rnms
from mmdet.core import rdets2points


TEST_PIPELINE = [
    {'type': 'LoadImageFromWebcam'},
    {'type': 'MultiScaleFlipAug', 'img_scale': (1024, 1024), 'flip': False,
     'transforms': [{'type': 'RResize', 'img_scale': (1024, 1024), 'keep_ratio': True},
                    {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
                    {'type': 'Pad', 'size_divisor': 32},
                    {'type': 'ImageToTensor', 'keys': ['img']},
                    {'type': 'Collect', 'keys': ['img']}]}
]


class Stream(Dataset):
    CLASSES = ['1', '2', '3', '4', '5']

    def __init__(self, imgs, chip_size=1024, stride=None,
                 pipeline=TEST_PIPELINE, classes=None):
        if classes:
            self.CLASSES = classes
        self.ext = ['.jpg', '.png', '.img', '.tif']
        self.pipeline = pipeline
        imgs = Path(imgs)
        assert imgs.exists(), "No such file or dir: {}".format(imgs)
        self.chip_size = chip_size
        self.stride = stride if stride is not None else chip_size // 4
        self.clip_stride = self.chip_size - self.stride
        self.img_seq = [img for img in imgs.iterdir() if img.suffix in self.ext]
        self.idx_map = {}
        self.transforms = {}
        for img_idx, img in enumerate(self.img_seq):
            idx_base = len(self.idx_map)
            with rasterio.open(img) as fp:
                self.transforms[str(img.name)] = fp.transform
                h, w = fp.shape
                chips_i = [_ for _ in range(math.ceil((w - self.chip_size) / self.clip_stride) + 1)]
                chips_j = [_ for _ in range(math.ceil((h - self.chip_size) / self.clip_stride) + 1)]
                chips = [(i, j) for i in chips_i for j in chips_j]
                for idx, (i, j) in enumerate(chips):
                    self.idx_map[idx+idx_base] = (img_idx, (i*self.clip_stride, j*self.clip_stride))
        imgs_map = {
            'chip_size': self.chip_size,
            'stride': self.stride,
            'img_count': len(self.img_seq),
            'idx_map': self.idx_map,
            'img_seq': self.img_seq
        }
        self.imgs_map = imgs_map

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        img_idx, (local_x, local_y) = self.idx_map[idx]
        img = self.img_seq[img_idx]
        with rasterio.open(img) as fp:
            bands = (3, 2, 1,) if len(fp.units) >= 3 else (1,)
            im = fp.read(
                bands,
                window=Window(local_x, local_y, self.chip_size, self.chip_size),
                boundless=True,
                fill_value=0,
            )
            if bands == (1,):
                im = np.concatenate((im, im, im), axis=0)
            im = im.transpose(1, 2, 0)  # 3xHxW to HxWx3
            test_pipeline = Compose(self.pipeline)
            im_data = dict(img=im)
            im_data = test_pipeline(im_data)
        return im_data

    def parse_dets(self, det_res):
        assert len(det_res) == len(self.idx_map)
        for idx, det in enumerate(det_res):
            _, (local_x, local_y) = self.idx_map[idx]
            for i in range(len(det)):
                det[i][:, 0] += local_x
                det[i][:, 1] += local_y
        det_res_dict = {}
        classes = len(det_res[0])
        for i in range(classes):
            last_ = 0
            init_array = np.array([])
            init_array.resize((0, 6))
            for idx, det in enumerate(det_res):
                img_idx, _ = self.idx_map[idx]
                if img_idx != last_:
                    img_filename = self.img_seq[last_].name
                    if img_filename not in det_res_dict.keys():
                        det_res_dict[img_filename] = [init_array]
                    else:
                        det_res_dict[img_filename].append(init_array)
                    # update img idx
                    last_ = img_idx
                    init_array = np.array([])
                    init_array.resize((0, 6))
                if det[i].size > 0:
                    init_array = np.concatenate((init_array, det[i]), axis=0)
            # update last img_det
            img_filename = self.img_seq[last_].name
            if img_filename not in det_res_dict.keys():
                det_res_dict[img_filename] = [init_array]
            else:
                det_res_dict[img_filename].append(init_array)
        return det_res_dict

    def nms_rotated(self, det_res):
        det_res_dict = self.parse_dets(det_res)
        for f in det_res_dict:
            for i in range(len(det_res_dict[f])):
                sub_ts = torch.from_numpy(det_res_dict[f][i])
                if sub_ts.numel() > 0:
                    det_ts, _ = rnms(sub_ts, iou_thr=0.2)
                else:
                    det_ts = sub_ts
                det_np = det_ts.numpy()
                det_res_dict[f][i] = det_np
        return det_res_dict

    def export_dets(self, det_res, outdir, thresh_conf=0.3):
        det_res_dict = self.nms_rotated(det_res)
        # mkdir for output
        if Path(outdir).exists():
            shutil.rmtree(outdir)
        Path(outdir).mkdir(parents=True)
        # handle img one by one
        for det in det_res_dict.keys():
            geojsons_features = []
            for cls_idx, cls in enumerate(self.CLASSES):
                det_cls = det_res_dict[det][cls_idx]
                if det_cls.size > 0:
                    det_poly = rdets2points(det_cls)
                    det_poly = det_poly[det_poly[:, -1] >= thresh_conf]
                    det_transform = self.transforms[det]
                    det_poly_x = det_poly[:, :8:2]
                    det_poly_y = det_poly[:, 1:8:2]
                    det_poly_x = list(map(int, det_poly_x.flatten()))
                    det_poly_y = list(map(int, det_poly_y.flatten()))
                    det_conf = list(det_poly[:, -1].flatten())
                    poly_x, poly_y = xy(det_transform, det_poly_y,
                                                det_poly_x)
                    geojson_features = self.geojson_feature(poly_x, poly_y, cls,
                                                            det_conf)
                    geojsons_features.extend(geojson_features)
            geojsons = {'type': 'FeatureCollection',
                        'features': geojsons_features}
            img_name = str(Path(det).stem)
            save_path_geojson = Path(outdir) / (img_name + '.geojson')
            with open(save_path_geojson, 'w') as f:
                json.dump(geojsons, f)

    def geojson_feature(self, poly_x, poly_y, class_name, scores=None):
        features = []
        assert len(poly_x) % 4 == 0 and len(poly_y) == 4 * len(scores), \
            "No valid length of rectangle polygon"

        for i in range(0, len(poly_x), 4):
            polygon = [
                [poly_x[i], poly_y[i]],
                [poly_x[i + 1], poly_y[i + 1]],
                [poly_x[i + 2], poly_y[i + 2]],
                [poly_x[i + 3], poly_y[i + 3]],
                [poly_x[i], poly_y[i]],
            ]
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [polygon]
                },
                'properties': {
                    'class_name': class_name
                }
            }
            idx = i // 4
            if scores[idx] is not None:
                box_scores = scores[idx]
                if box_scores is not None:
                    if type(box_scores) is list:
                        feature['properties']['scores'] = box_scores
                    else:
                        feature['properties']['score'] = box_scores

            features.append(feature)
        return features
