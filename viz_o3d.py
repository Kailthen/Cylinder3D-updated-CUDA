# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from pypcd import pypcd
import yaml
import argparse
import glob
import os
from utils.pc_utils import read_to_df

def load_vertex(file_path):
    """ Load 3D points of a scan. The fileformat is the .bin format used in
        the KITTI dataset.
        Args:
            scan_path: the (full) filename of the scan file
        Returns:
            A nx4 numpy array of homogeneous points (x, y, z, 1).
    """
    if file_path.endswith(".pcd"):
        # df, _ = read_to_df(file_path, min_bound=None, max_bound=None) # for kitti
        df, _ = read_to_df(file_path) # for rsm1
        raw_data = np.empty((len(df), 4), dtype=np.float32)
        raw_data[:, 0] = df['x'].to_numpy()
        raw_data[:, 1] = df['y'].to_numpy()
        raw_data[:, 2] = df['z'].to_numpy()
        # raw_data[:, 3] = df['intensity'].to_numpy() / 255.0
    else:
        raw_data = np.fromfile(file_path, dtype=np.float32)
        raw_data = raw_data.reshape((-1, 4))
    current_vertex = raw_data
    # current_points = current_vertex[:, 0:3]
    # current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    # current_vertex[:, :-1] = current_points
    return current_vertex


def load_labels_gt(label_path):
    """ Load semantic and instance labels in SemanticKitti format.
    """
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    # sanity check
    assert ((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label


def load_labels(label_path):
    """ Load pred label
    """
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    return label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scan_path', type=str, help='')
    parser.add_argument('--label_path', type=str, help='')
    parser.add_argument('--config_path', type=str, default='config/semantic-kitti.yaml')

    args = parser.parse_args()

    color_map = yaml.unsafe_load(open(args.config_path, 'rt'))
    color_map = color_map['color_map']

    pcds = glob.glob(os.path.join(args.scan_path, "*.pcd"))
    pcds.sort()

    labels = glob.glob(os.path.join(args.label_path, "*.label"))
    labels.sort()
    assert len(labels) == len(pcds)

    for pcd_fn, label_fn in zip(pcds, labels):
        print(pcd_fn)

        scan = load_vertex(pcd_fn)
        label = load_labels(label_fn)

        assert len(label) == len(scan)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
        # pcd.paint_uniform_color([0.25, 0.25, 0.25])
        # colors = np.array(pcd.colors)
        # colors[label > 200] = [1.0, 0.0, 0.0]

        # pcd.colors = o3d.utility.Vector3dVector(colors)
        colors = np.zeros((label.shape[0], 3))
        for idx, c in enumerate(label):
            colors[idx, :] = color_map[c]
            colors[idx, :] = colors[idx, :] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f'pointcloud semantic', width=3000, height=2000)
        vis.add_geometry(pcd)
        # parameters = o3d.io.read_pinhole_camera_parameters("/home/user/Repo/LiDAR-MOS/ScreenCamera_2022-02-20-21-03-42.json")
        # ctr = vis.get_view_control()
        # ctr.convert_from_pinhole_camera_parameters(parameters)
        vis.run()
        vis.destroy_window()

        continue
