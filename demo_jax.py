"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution using JAX."""

import time
import cv2
import numpy as np
from skimage import measure

from fusion_jax import TSDFVolumeJAX, meshwrite, pcwrite, get_view_frustum


if __name__ == "__main__":
    print("Estimating voxel volume bounds...")
    n_imgs = 1000
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=" ")
    vol_bnds = np.zeros((3, 2))

    for i in range(n_imgs):
        depth_im = cv2.imread(f"data/frame-{i:06d}.depth.png", -1).astype(float)
        depth_im /= 1000.0
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt(f"data/frame-{i:06d}.pose.txt")
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    print("Initializing voxel volume...")
    tsdf_vol = TSDFVolumeJAX(vol_bnds, voxel_size=0.02)

    t0_elapse = time.time()
    for i in range(n_imgs):
        print(f"Fusing frame {i+1}/{n_imgs}")
        color_image = cv2.cvtColor(
            cv2.imread(f"data/frame-{i:06d}.color.jpg"), cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        depth_im = cv2.imread(f"data/frame-{i:06d}.depth.png", -1).astype(float)
        depth_im /= 1000.0
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt(f"data/frame-{i:06d}.pose.txt")
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.0)

    fps = n_imgs / (time.time() - t0_elapse)
    print(f"Average FPS: {fps:.2f}")

    print("Saving mesh to mesh.ply...")
    verts, faces, norms, vals = measure.marching_cubes(
        np.array(tsdf_vol.tsdf_vol), level=0
    )
    verts_ind = np.round(verts).astype(int)
    verts = verts * tsdf_vol.voxel_size + np.array(tsdf_vol.vol_origin)
    rgb_vals = np.array(tsdf_vol.color_vol)[
        verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]
    ]
    colors_b = np.floor(rgb_vals / tsdf_vol.color_const)
    colors_g = np.floor((rgb_vals - colors_b * tsdf_vol.color_const) / 256)
    colors_r = rgb_vals - colors_b * tsdf_vol.color_const - colors_g * 256
    colors = np.stack([colors_r, colors_g, colors_b], axis=1).astype(np.uint8)
    meshwrite("mesh.ply", verts, faces, norms, colors)

    print("Saving point cloud to pc.ply...")
    pc = tsdf_vol.get_point_cloud()
    pcwrite("pc.ply", pc)
