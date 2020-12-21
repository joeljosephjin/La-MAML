from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# they taking cifar10 from url or pytorch ??
from torchvision.datasets.vision import VisionDataset
# 
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
# import ipdb


# defining the mapping from parent classes to fine-grained classes in cifar
# in case one needs to split tasks by parent class
super_class_to_class = {
    'aquatic_mammals'   :  ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish'  :  ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers'  :   ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers'  :   ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables'  :  ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices' :   ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture'  :   ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects'  :   ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores'  :  ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things' :  ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes'  :  ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores' : ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals'  :  ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates'  :  ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people'  : ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles'  :  ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals'  : ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1' :  ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2'  : ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    # what is this good for?
    base_folder = 'cifar-10-batches-py'
    # how many places has this been pasted??
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # same garbage lots
    filename = "cifar-10-python.tar.gz"
    # garbage no.s
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    # list of garbage no.s {data_batch_<no.>:'78sadfasyd9asy7d9'}
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    # list of 1 garbage no.
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    # garbage
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    # root=root directory, train=what else?, transform, target_transform, download
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        # use the download function if necessary
        if download: self.download()

        # __check_integrity stores {no error or what} 
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # downloaded_list will contain train or test accordingly
        if self.train: downloaded_list = self.train_list
        else: downloaded_list = self.test_list

        # {data, targets, super_targets} and high_lvl_supervision
        self.high_level_supervise = True
        self.data = []
        self.targets = []
        self.super_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            # load the filename
            file_path = os.path.join(self.root, self.base_folder, file_name)
            # open the file
            with open(file_path, 'rb') as f:
                # pickle(load in dict format) according to system version
                if sys.version_info[0] == 2: entry = pickle.load(f)
                else: entry = pickle.load(f, encoding='latin1')

                # put data in self.data
                self.data.append(entry['data'])

                # put labels/fine_labels in self.targets
                if 'labels' in entry: self.targets.extend(entry['labels'])
                else: self.targets.extend(entry['fine_labels'])             

                # put coarse_labels in self.super_targets
                self.super_targets.extend(entry['coarse_labels'])           

        # 
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')

            self.classes = data[self.meta['key']]
            self.super_classes = data[self.meta['coarse_key']]

        self.get_class_ids()

    def get_class_ids(self):
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.super_class_to_idx = {_class: i for i, _class in enumerate(self.super_classes)}

        high_ids = []
        low_ids = []
        low_idxs = np.arange(len(self.classes))
        for key in super_class_to_class:
            for classes in super_class_to_class[key]:
                high_ids.append(self.super_class_to_idx[key])
                low_ids.append(self.class_to_idx[classes])

        high_ids_np = np.array(high_ids) 
        low_ids_np = np.array(low_ids)
        self.low_high_map = np.stack([low_ids_np, high_ids_np], axis = 1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, super_target = self.data[index], self.targets[index], self.super_targets[index]

        if(self.high_level_supervise):
            target = super_target


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, super_target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


# why this even required??
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    # does this folder exist in my pc??
    base_folder = 'cifar-100-python'
    # this is the famous cifar url
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    # obvious
    filename = "cifar-100-python.tar.gz"
    # garbage
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    # god knows what??
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    # god knows what 2??
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    # why all this garbage can info??
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'coarse_key': 'coarse_label_names',        
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
