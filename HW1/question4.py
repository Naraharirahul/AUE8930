import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import open3d as o3d
import struct

import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

import nuscenes

def image_visualize():
    files = os.listdir("CAM_FRONT/")
    for file in files:
        file_name = 'CAM_FRONT/' + file
        image = cv2.imread(file_name)
        plt.imshow(image)
        plt.show()

# image_visualize()
def lidar_visualize():

    # seg_files = os.listdir("v1.0-mini/")
    # pcd_files = os.listdir("LIDAR_TOP/")
    
    seg_name='v1.0-mini/00d437419bec4cb48b8feac9a77875c8_lidarseg.bin'
    seg=np.fromfile(seg_name, dtype=np.uint8)
    
    #Semantic labeling

    # color = np.zeros([len(seg), 3])
    # color[:, 0] = seg/32
    # color[:, 1] = seg/32
    # color[:, 2] = seg/32

    pcd_name='LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'
    scan=np.fromfile(pcd_name, dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]

    

    # intensity
    # color = np.zeros([len(points), 3])
    # color[:, 0] = points[:,3]/255 + 0.9
    # color[:, 1] = points[:,3]/255 + 0.1
    # color[:, 2] = points[:,3]/255 + 0.1
 
    # z value
    color = np.zeros([len(points), 3])
    color[:, 0] = points[:,2]
    color[:, 1] = points[:,2] 
    color[:, 2] = points[:,2]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([pcd])

lidar_visualize()


def from_file( file_name : str):
    assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

    meta = []
    with open(file_name, 'rb') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            meta.append(line)
            if line.startswith('DATA'):
                break

        data_binary = f.read()

    # Get the header rows and check if they appear as expected.
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    counts = meta[5].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)
    assert width > 0
    assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
    assert height == 1, 'Error: height != 0 not supported!'
    assert data == 'binary'

    # Lookup table for how to decode the binaries.
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                        'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                        'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point.
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # A NaN in the first point indicates an empty pointcloud.
    point = np.array(points[0])
    if np.any(np.isnan(point)):
        return cls(np.zeros((feature_count, 0)))

        # Convert to numpy matrix.
    points = np.array(points).transpose()

    return points

radar_data_points = from_file('D:/Documents/dataset/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151603555991.pcd')

radar_data_points = radar_data_points.transpose()

color = np.zeros([len(radar_data_points), 3])

# For height uncomment below:

color[:, 0] = radar_data_points[:,2] + 0.1
color[:, 1] = radar_data_points[:,2] + 0.7
color[:, 2] = radar_data_points[:,2] + 0.1

# For velocity:

color[:, 0] = radar_data_points[:,8] + 0.1
color[:, 1] = radar_data_points[:,8] + 0.7
color[:, 2] = radar_data_points[:,8] + 0.1



pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(radar_data_points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])