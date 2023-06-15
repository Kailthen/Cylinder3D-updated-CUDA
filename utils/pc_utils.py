import pandas as pd
import open3d as o3d
import os

# def read_to_df(pcd_fpn, min_bound=[3.0, -100.0, -5.0], max_bound=[100.0, 100.0, 5.0]):
def read_to_df(pcd_fpn, min_bound=None, max_bound=None):
    """
    """
    assert os.path.exists(pcd_fpn), f"pcd file {pcd_fpn} not exist"

    pcd = o3d.t.io.read_point_cloud(pcd_fpn)
    xyz = pcd.point['positions'].numpy()
    df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    for k, v in pcd.point.items():
        if k == 'colors':
            colors = v.numpy()
            df['r'] = colors[:, 0]
            df['g'] = colors[:, 1]
            df['b'] = colors[:, 2]
        elif k == 'normals':
            pass
        elif k == 'positions':
            pass
        else:
            df[k] = v.numpy()
    df = df.dropna(axis=0)
    if min_bound is not None and max_bound is not None:
        df = df[  (df['x'] >= min_bound[0]) & (df['x'] <= max_bound[0]) 
                & (df['y'] >= min_bound[1]) & (df['y'] <= max_bound[1])
                & (df['z'] >= min_bound[2]) & (df['z'] <= max_bound[2])
                ]
    return df, pcd