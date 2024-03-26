class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/houxiaojun/Workspace/SDSTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/houxiaojun/Workspace/SDSTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/houxiaojun/Workspace/SDSTrack/pretrained_networks'
        self.got10k_val_dir = '/home/houxiaojun/Workspace/SDSTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/home/houxiaojun/Workspace/SDSTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/houxiaojun/Workspace/SDSTrack/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/houxiaojun/Workspace/SDSTrack/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/houxiaojun/Workspace/SDSTrack/data/coco_lmdb'
        self.coco_dir = '/home/houxiaojun/Workspace/SDSTrack/data/coco'
        self.lasot_dir = '/home/houxiaojun/Workspace/SDSTrack/data/lasot'
        self.got10k_dir = '/home/houxiaojun/Workspace/SDSTrack/data/got10k/train'
        self.trackingnet_dir = '/home/houxiaojun/Workspace/SDSTrack/data/trackingnet'
        self.depthtrack_dir = '/home/houxiaojun/Workspace/SDSTrack/data/depthtrack/train'
        self.lasher_dir = '/home/houxiaojun/Workspace/SDSTrack/data/lasher/trainingset'
        self.visevent_dir = '/home/houxiaojun/Workspace/SDSTrack/data/visevent/train'
