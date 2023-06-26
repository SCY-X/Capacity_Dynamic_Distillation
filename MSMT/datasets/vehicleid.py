import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class VehicleID(BaseImageDataset):
    """
       VehicleID

        Reference:
        Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.

        URL: `<https://pkuml.org/resources/pku-vehicleid.html>`_

        Train dataset statistics:
        - identities: 13164.
        - images: 113346.

       identities: 13164
       images: 113346 (train) + 5693 (samll query) + 800 (samll gallery)
       cameras: 20
    """
    dataset_dir = ''

    def __init__(self, root='../', verbose=True, **kwargs):
        super(VehicleID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, mode='train')
        query = self._process_dir(self.query_dir, relabel=False, mode='query')
        gallery = self._process_dir(self.gallery_dir, relabel=False, mode='gallery')

        if verbose:
            print("=> VehicleID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, mode='train'):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_')

        if relabel:
            pid_container = set()
            for img_path in img_paths:
                pid = map(int, pattern.search(img_path).groups())
                pid = list(pid)[0]
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = map(int, pattern.search(img_path).groups())
            pid = list(pid)[0]
            #print(pid)
            if mode == 'train' or mode == 'query':
                camid = 0
            else:
                camid = 1
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 13164  # pid == 0 means background
            #assert 1 <= camid <= 20
            #camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

