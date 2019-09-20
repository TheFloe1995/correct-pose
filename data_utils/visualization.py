import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import scipy.io

from data_utils import pose_features


class PoseVisualizer:
    colors = ['orange', 'blue', 'green', 'red', 'purple']

    @classmethod
    def single(cls, pose, visible_fingers=(0, 1, 2, 3, 4), label_fingers=False, thick=False):
        axes, fig = cls._prepare_3d_plot(pose, 1, 1, 12, 12)
        cls._plot_pose(axes.flatten()[0], pose, visible_fingers, label_fingers, cls.colors, thick)
        plt.show()

    @classmethod
    def pair(cls, poses, labels=('Pose', 'Reference'), visible_fingers=(0, 1, 2, 3, 4),
             label_fingers=False):
        cls.n_tuple(poses, labels, visible_fingers, label_fingers)

    @classmethod
    def triplet(cls, poses, labels=('Input', 'Prediction', 'Label'),
                visible_fingers=(0, 1, 2, 3, 4), label_fingers=False):
        cls.n_tuple(poses, labels, visible_fingers, label_fingers)

    @classmethod
    def n_tuple(cls, poses, labels, visible_fingers=(0, 1, 2, 3, 4), label_fingers=False):
        ax, fig = cls._prepare_3d_plot(poses, 1, 1, 12, 12)
        cls._plot_n_poses(ax[0, 0], poses, visible_fingers, label_fingers, labels)
        plt.show()

    @classmethod
    def single_batch(cls, poses, visible_fingers=(0, 1, 2, 3, 4), label_fingers=False):
        n_cols = 3
        n_rows = (poses.shape[0] - 1) // n_cols + 1

        axes, fig = cls._prepare_3d_plot(poses.reshape(-1, 21, 3), n_rows, n_cols)
        for i, pose in enumerate(poses):
            row_idx = i // n_cols
            col_idx = i % n_cols
            cls._plot_pose(axes[row_idx, col_idx], pose, visible_fingers, label_fingers, cls.colors)

        plt.show()

    @classmethod
    def pair_batch(cls, pose_pairs, labels=('Pose', 'Reference'), visible_fingers=(0, 1, 2, 3, 4),
                   label_fingers=False):
        cls.n_batch(pose_pairs, labels, visible_fingers, label_fingers)

    @classmethod
    def triplet_batch(cls, pose_triplets, visible_fingers=(0, 1, 2, 3, 4), label_fingers=False,
                      labels=('Input', 'Prediction', 'Label')):
        cls.n_batch(pose_triplets, labels, visible_fingers, label_fingers)

    @classmethod
    def n_batch(cls, pose_batch, labels, visible_fingers=(0, 1, 2, 3, 4), label_fingers=False):
        n_cols = 3
        n_rows = (pose_batch.shape[0] - 1) // n_cols + 1

        axes, fig = cls._prepare_3d_plot(pose_batch.reshape(-1, 21, 3), n_rows, n_cols)
        for i, poses in enumerate(pose_batch):
            row_idx = i // n_cols
            col_idx = i % n_cols
            cls._plot_n_poses(axes[row_idx, col_idx], poses, visible_fingers, label_fingers, labels)

        plt.show()

    @classmethod
    def depth_image_batch(cls, directory, n=None, uv_labels=None):
        image_paths = glob.glob(os.path.join(directory, '*.png'))
        if n is not None:
            image_paths = image_paths[:n]
        else:
            n = len(image_paths)

        images = []
        for image_path in sorted(image_paths):
            images.append(imageio.imread(image_path))

        if uv_labels is not None:
            cls._mark_keypoints(images, uv_labels)

        fig, ax = plt.subplots(int(np.ceil(n / 2.0)), 2, figsize=(15, int(np.ceil(n / 2.0)) * 5.0))
        for i, image in enumerate(images):
            row = i // 2
            col = i % 2
            ax[row, col].imshow(image)

        plt.show()

    @classmethod
    def nyu_rgb_with_keypoints(cls, image_dir, keypoint_file, span):
        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        image_paths = image_paths[span[0]:span[1]]
        n = len(image_paths)

        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path))

        label_data = scipy.io.loadmat(keypoint_file)
        kp_selection = [29,
                        0, 1, 3, 5,
                        6, 7, 9, 11,
                        12, 13, 15, 17,
                        18, 19, 21, 23,
                        24, 25, 27, 28]
        uv_keypoints = label_data['joint_uvd'][0, span[0]:span[1], :, :2]
        uv_keypoints = uv_keypoints[:, kp_selection]

        cls._mark_keypoints(images, uv_keypoints)

        fig, ax = plt.subplots(int(np.ceil(n / 2.0)), 2,
                               figsize=(2 * 12.8, int(np.ceil(n / 2.0)) * 9.6))
        for i, image in enumerate(images):
            row = i // 2
            col = i % 2
            ax[row, col].imshow(image)

        plt.show()

    @classmethod
    def _prepare_3d_plot(cls, poses, n_rows, n_cols, col_size=5, row_size=5):
        subplot_args = {'projection': '3d'}
        fig_size = (col_size * n_cols, row_size * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, subplot_kw=subplot_args,
                                 squeeze=False)
        cls._set_axis_scales(axes, poses)

        for ax in axes.flatten():
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        return axes, fig

    @classmethod
    def _set_axis_scales(cls, axes, poses):
        # ax.set_aspect('equal') doesn't work for 3D plots, therefore manually computing axis scales
        joints = poses.reshape(-1, 3)
        max_val = (joints.max(dim=0).values - joints.min(dim=0).values).max() * 1.2 / 2.0
        center = (joints.max(dim=0).values + joints.min(dim=0).values).squeeze() / 2.0
        for ax in axes.flatten():
            ax.set_xlim(center[0] - max_val, center[0] + max_val)
            ax.set_ylim(center[1] - max_val, center[1] + max_val)
            ax.set_zlim(center[2] - max_val, center[2] + max_val)

    @classmethod
    def _plot_n_poses(cls, ax, poses, visible_fingers, label_fingers, labels):
        patches = []
        for i, pose in enumerate(poses):
            cls._plot_pose(ax, pose, visible_fingers, label_fingers, [cls.colors[i]] * 5)
            patches.append(mpl_patches.Patch(color=cls.colors[i], label=labels[i]))

        plt.legend(handles=patches)

    @classmethod
    def _plot_pose(cls, ax, pose, visible_fingers, label_fingers, colors, thick=False):
        visible_fingers = list(visible_fingers)
        pose = pose.cpu()
        fingers = pose_features.joints_of_all_fingers(pose.reshape(1, 21, 3)).squeeze()
        for i, finger in enumerate(fingers[visible_fingers]):
            finger = finger.numpy()
            if thick:
                marker_size = 8.0
                line_width = 4.0
            else:
                marker_size = 3.0
                line_width = 1.0
            ax.plot(finger[:, 0], finger[:, 1], finger[:, 2], 'o-', color=colors[i],
                    markersize=marker_size, linewidth=line_width)
            if label_fingers:
                cls._label_finger(ax, finger, i, colors[i])

    @classmethod
    def _label_finger(cls, ax, finger, label, color):
        last_bone = finger[-1] - finger[-2]
        label_position = finger[-1] + 0.5 * last_bone
        ax.text(label_position[0], label_position[1], label_position[2], label, color=color,
                fontweight='bold')

    @classmethod
    def _mark_keypoints(cls, images, keypoints):
        for image, kps in zip(images, keypoints):
            for u, v in kps:
                u, v = int(u), int(v)
                image[v - 2:v + 2, u - 2:u + 2] = 2000
