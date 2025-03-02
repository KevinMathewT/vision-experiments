
import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob

from datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from utils.image import imread_cv2
from utils.misc import get_stride_distribution

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

def depth_read(filename):
    depth = np.load(filename)
    return depth

def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

class TarTanAirDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/tartanair',
                 dset='Hard',
                 use_augs=False,
                 S=2,
                 strides=[8],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 dist_type=None,
                 *args, 
                 **kwargs
                 ):

        print('loading tartanair dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'tartanair'
        self.split = dset
        self.S = S # number of frames
        self.verbose = verbose

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotations = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location)) #'data/tartanair'


        for subdir in self.subdirs:
            print(os.path.join(subdir, "*/", dset, "*/"))
            print(glob.glob(os.path.join(subdir, "*/", dset, "*/")))
        
        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/", dset, "*/")):
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        if quick:
           self.sequences = self.sequences[1:2] 
        
        for seq in self.sequences:
            # if self.verbose: 
            print('seq', seq)

            rgb_path = os.path.join(seq, 'image_left')
            depth_path = os.path.join(seq, 'depth_left')
            caminfo_path = os.path.join(seq, 'pose_left.txt')
            caminfo = np.loadtxt(caminfo_path)
            
            for stride in strides:
                for ii in range(0,len(os.listdir(rgb_path))-self.S*stride+1, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    self.rgb_paths.append([os.path.join(rgb_path, '%06d_left.png' % idx) for idx in full_idx])
                    self.depth_paths.append([os.path.join(depth_path, '%06d_left_depth.npy' % idx) for idx in full_idx])
                    self.annotations.append(caminfo[full_idx])
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)
                if self.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        
        fx = 320.0  # focal length x
        fy = 320.0  # focal length y
        cx = 320.0  # optical center x
        cy = 240.0  # optical center y

        width = 640
        height = 480

        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))
    
    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()

        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.annotations = [self.annotations[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        annotations = self.annotations[index]

        views = []
        for i in range(2):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]

            # load camera params
            camera_pose = np.array(xyzqxqyqxqw_to_c2w(annotations[i]), dtype=np.float32)
            # camera_pose = np.linalg.inv(camera_pose)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = depth_read(depthpath)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-5]+'-'+rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views

if __name__ == "__main__":
    import mediapy as media
    import matplotlib.pyplot as plt
    import rerun as rr
    import numpy as np
    import torch
    import random

    use_augs = False
    S = 2
    strides = [1,2,3,4,5,6,7,8,9]
    clip_step = 2
    quick = False  # Set to True for quick testing


    def show_3d_viz(dataset):
        i = random.randint(0, len(dataset))
        # show_one_sample(dataset, i, 0)
        # show_one_sample(dataset, i, 1)

        def render(d):
            pts3d = d['pts3d']
            img = d['img']

            pts3d = pts3d.reshape(-1, 3)

            img = ((img.permute(1,2,0).numpy()+1)/2*255).astype(np.uint8)
            colors = img.reshape(-1, 3)

            rr.init("viz", spawn=True)
            rr.log("scene", rr.Points3D(pts3d, colors=colors))

        render(dataset[i][0])
        render(dataset[i][1])
    
    def show_3d_viz_v2(ds):
        i = random.randint(0, len(ds)-1)
        smp = ds[i]  # smp has 2 frames: smp[0], smp[1]

        rr.init("viz", spawn=True)

        # Gather keypoint tracks across both frames
        # Suppose tracks_3d is shape (N,3) in each frame
        # => stacked we get (S,N,3), here S=2
        frs_3d = []
        for t in range(len(smp)):
            frs_3d.append(smp[t]['tracks_3d'])  # (N,3)
        frs_3d = np.stack(frs_3d, axis=0)      # (S,N,3)

        # Create line strips: one line per keypoint
        lines = []
        for n in range(frs_3d.shape[1]):
            line = frs_3d[:, n, :][None]       # shape (1,S,3)
            lines.append(line)
        lines = np.concatenate(lines, axis=0)  # shape (N,S,3)

        # Give each line a random color
        colors = np.random.randint(0, 255, (lines.shape[0], 3), dtype=np.uint8)

        rr.log("scene/tracks", rr.LineStrips3D(lines, colors=colors))

        # Log the dense depth-based points for both frames
        for t in range(len(smp)):
            pts = smp[t]['pts3d'].reshape(-1, 3)
            img = ((smp[t]['img'].permute(1,2,0).numpy() + 1)/2*255).astype(np.uint8)
            c   = img.reshape(-1, 3)
            rr.log(f"scene/dense_{t}", rr.Points3D(pts, colors=c))
            

    dataset = TarTanAirDUSt3R(
        dset='Easy',
        use_augs=use_augs,
        S=S,
        strides=strides,
        clip_step=clip_step,
        quick=quick,
        verbose=False,
        resolution=(512,384), 
        dist_type='linear_9_1',
        aug_crop=16)
    
    idxs = np.arange(0, len(dataset)-1, (len(dataset)-1)//10)

    print(dataset[idxs[0]][0].keys())
    for k, v in dataset[idxs[0]][0].items():
        if isinstance(v, np.ndarray):
            print(f"{k}:", dataset[idxs[0]][0][k].shape)
        elif isinstance(v, torch.Tensor):
            print(f"{k}:", dataset[idxs[0]][0][k].size())
        else:
            print(f"{k}:", dataset[idxs[0]][0][k])
    # for idx in idxs:
    #     print(f"Visualizing scene {idx}...")
    
    show_3d_viz_v2(dataset)