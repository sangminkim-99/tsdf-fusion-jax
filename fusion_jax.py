import jax
import jax.numpy as jnp
import numpy as np
from skimage import measure


def rigid_transform(xyz, transform):
    xyz_h = jnp.concatenate([xyz, jnp.ones((xyz.shape[0], 1))], axis=1)
    xyz_t = xyz_h @ transform.T
    return xyz_t[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    im_h, im_w = depth_im.shape
    max_depth = jnp.max(depth_im)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    x = (jnp.array([0, 0, im_w, im_w]) - cx) * max_depth / fx
    y = (jnp.array([0, im_h, 0, im_h]) - cy) * max_depth / fy
    z = max_depth * jnp.ones(4)

    corners = jnp.stack([jnp.zeros(4), x, y, z], axis=0)
    corners = jnp.pad(corners[1:], ((0, 0), (0, 1)), constant_values=0)
    corners = jnp.concatenate([jnp.zeros((3, 1)), corners], axis=1)
    return rigid_transform(corners.T, cam_pose).T


@jax.jit
def integrate_jit(
    tsdf_vol,
    weight_vol,
    color_vol,
    vox_coords,
    vol_origin,
    voxel_size,
    color_const,
    trunc_margin,
    obs_weight,
    color_im,
    depth_im,
    cam_intr,
    cam_pose,
):

    # Compute world coordinates of voxels
    world_pts = vol_origin + vox_coords * voxel_size
    # Transform world points into camera coordinates
    cam_pts = rigid_transform(world_pts, jnp.linalg.inv(cam_pose))

    # Project camera points to pixel coordinates
    pix_z = cam_pts[:, 2]
    pix_x = jnp.round((cam_pts[:, 0] * cam_intr[0, 0]) / pix_z + cam_intr[0, 2]).astype(
        jnp.int32
    )
    pix_y = jnp.round((cam_pts[:, 1] * cam_intr[1, 1]) / pix_z + cam_intr[1, 2]).astype(
        jnp.int32
    )

    pix_x, pix_y, pix_z = (
        pix_x.reshape(tsdf_vol.shape),
        pix_y.reshape(tsdf_vol.shape),
        pix_z.reshape(tsdf_vol.shape),
    )

    H, W = depth_im.shape
    # Check which pixels are inside image bounds and have positive depth
    in_bounds = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H) & (pix_z > 0)

    # Get depth values at projected pixels, zero if out of bounds
    depth_val = jnp.where(in_bounds, depth_im[pix_y, pix_x], 0.0)
    depth_diff = depth_val - pix_z
    # Valid pixels have positive depth and are within truncation margin
    valid = (depth_val > 0) & (depth_diff >= -trunc_margin)
    valid_mask = valid & in_bounds

    # Old weights and tsdf values
    w_old = weight_vol
    tsdf_old = tsdf_vol

    # Compute new weights and tsdf values only where valid
    w_new = w_old + obs_weight * valid_mask.astype(jnp.float32)
    tsdf_new = (
        tsdf_old * w_old
        + obs_weight
        * valid_mask.astype(jnp.float32)
        * (jnp.minimum(1.0, depth_diff / trunc_margin))
    ) / jnp.maximum(w_new, 1e-5)

    # Update tsdf and weight volumes using mask
    tsdf_vol = jnp.where(valid_mask, tsdf_new, tsdf_old)
    weight_vol = w_new

    # Old color values
    old_color = color_vol
    old_b = jnp.floor(old_color / color_const)
    old_g = jnp.floor((old_color - old_b * color_const) / 256)
    old_r = old_color - old_b * color_const - old_g * 256

    # New colors sampled at valid pixels; zero elsewhere
    new_color = jnp.where(valid_mask, color_im[pix_y, pix_x], 0.0)
    new_b = jnp.floor(new_color / color_const)
    new_g = jnp.floor((new_color - new_b * color_const) / 256)
    new_r = new_color - new_b * color_const - new_g * 256

    # Weighted average for color update
    new_b = jnp.minimum(
        255.0,
        jnp.round(
            (w_old * old_b + obs_weight * valid_mask.astype(jnp.float32) * new_b)
            / jnp.maximum(w_new, 1e-5)
        ),
    )
    new_g = jnp.minimum(
        255.0,
        jnp.round(
            (w_old * old_g + obs_weight * valid_mask.astype(jnp.float32) * new_g)
            / jnp.maximum(w_new, 1e-5)
        ),
    )
    new_r = jnp.minimum(
        255.0,
        jnp.round(
            (w_old * old_r + obs_weight * valid_mask.astype(jnp.float32) * new_r)
            / jnp.maximum(w_new, 1e-5)
        ),
    )

    # Combine updated color channels into single value
    new_color_val = new_b * color_const + new_g * 256 + new_r

    # Update color volume where valid
    color_vol = jnp.where(valid_mask, new_color_val, old_color)

    return tsdf_vol, weight_vol, color_vol


class TSDFVolumeJAX:
    def __init__(self, vol_bnds, voxel_size):
        vol_bnds = jnp.asarray(vol_bnds)
        self.voxel_size = voxel_size
        self.trunc_margin = 5 * voxel_size
        self.color_const = 256 * 256

        self.vol_dim = jnp.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).astype(
            int
        )
        self.vol_bnds = vol_bnds.at[:, 1].set(
            vol_bnds[:, 0] + self.vol_dim * voxel_size
        )
        self.vol_origin = vol_bnds[:, 0]

        self.tsdf_vol = jnp.ones(self.vol_dim, dtype=jnp.float32)
        self.weight_vol = jnp.zeros(self.vol_dim, dtype=jnp.float32)
        self.color_vol = jnp.zeros(self.vol_dim, dtype=jnp.float32)

        xv, yv, zv = jnp.meshgrid(
            jnp.arange(self.vol_dim[0]),
            jnp.arange(self.vol_dim[1]),
            jnp.arange(self.vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = jnp.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.0):
        color_im = jnp.floor(
            color_im[..., 2] * self.color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )
        color_im = jnp.array(color_im, dtype=jnp.float32)
        depth_im = jnp.array(depth_im, dtype=jnp.float32)
        cam_intr = jnp.array(cam_intr, dtype=jnp.float32)
        cam_pose = jnp.array(cam_pose, dtype=jnp.float32)

        self.tsdf_vol, self.weight_vol, self.color_vol = integrate_jit(
            self.tsdf_vol,
            self.weight_vol,
            self.color_vol,
            self.vox_coords,
            self.vol_origin,
            self.voxel_size,
            self.color_const,
            self.trunc_margin,
            obs_weight,
            color_im,
            depth_im,
            cam_intr,
            cam_pose,
        )

    def get_point_cloud(self):
        tsdf_np = np.array(self.tsdf_vol)
        color_np = np.array(self.color_vol)

        verts = measure.marching_cubes(tsdf_np, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_size + np.array(self.vol_origin)

        rgb_vals = color_np[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self.color_const)
        colors_g = np.floor((rgb_vals - colors_b * self.color_const) / 256)
        colors_r = rgb_vals - colors_b * self.color_const - colors_g * 256
        colors = np.stack([colors_r, colors_g, colors_b], axis=1).astype(np.uint8)

        return np.hstack([verts, colors])


# Utility functions


def meshwrite(filename, verts, faces, norms, colors):
    with open(filename, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {verts.shape[0]}\n")
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write("property float nx\nproperty float ny\nproperty float nz\n")
        ply_file.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        ply_file.write(f"element face {faces.shape[0]}\n")
        ply_file.write("property list uchar int vertex_index\nend_header\n")
        for i in range(verts.shape[0]):
            ply_file.write(
                f"{verts[i,0]} {verts[i,1]} {verts[i,2]} "
                f"{norms[i,0]} {norms[i,1]} {norms[i,2]} "
                f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n"
            )
        for i in range(faces.shape[0]):
            ply_file.write(f"3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n")


def pcwrite(filename, xyzrgb):
    with open(filename, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {xyzrgb.shape[0]}\n")
        ply_file.write("property float x\nproperty float y\nproperty float z\n")
        ply_file.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        ply_file.write("end_header\n")
        for i in range(xyzrgb.shape[0]):
            ply_file.write(
                f"{xyzrgb[i,0]} {xyzrgb[i,1]} {xyzrgb[i,2]} "
                f"{xyzrgb[i,3]} {xyzrgb[i,4]} {xyzrgb[i,5]}\n"
            )
