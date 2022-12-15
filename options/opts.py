from .base_opts import BaseOpts
class TrainOpts(BaseOpts):
    def __init__(self):
        super(TrainOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Training Arguments ####
        self.parser.add_argument('--milestones',  default=[5, 10, 15, 20, 25], nargs='+', type=int)
        self.parser.add_argument('--start_epoch', default=1,      type=int)
        self.parser.add_argument('--epochs',      default=30,     type=int)
        self.parser.add_argument('--batch',       default=8,     type=int)
        self.parser.add_argument('--init_lr',     default=0.0001, type=float)
        self.parser.add_argument('--lr_decay',    default=0.5,    type=float)
        self.parser.add_argument('--beta_1',      default=0.9,    type=float, help='adam')
        self.parser.add_argument('--beta_2',      default=0.999,  type=float, help='adam')
        self.parser.add_argument('--momentum',    default=0.9,    type=float, help='sgd')
        self.parser.add_argument('--w_decay',     default=4e-4,   type=float)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--test_phase', type=bool, default=False)
        
        
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--perceptual_layers', type=int, default=3,
                                 help='index of vgg layer for extracting perceptual features.')
        self.parser.add_argument('--percep_is_l1', type=int, default=0, help='type of perceptual loss: l1 or l2')
        self.parser.add_argument('--select_loss', type=int, default=0, help='type of perceptual loss: l1 or l2')

        self.parser.add_argument('--L1_type', type=str, default='origin',
                                 help='use which kind of L1 loss. (origin|l1_plus_perL1)')
    
        self.parser.add_argument('--n_bins', type=int, default=5, help='No. of light bins')
        self.parser.add_argument('--c_in', type=int, default=7, help='No. of channels')
        self.parser.add_argument('--l_mode', type=int, default=0, help='tarining mode for lighting estimation')

        self.parser.add_argument('--lambda_img', type=float, default=1.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_spec', type=float, default=2.0, help='weight for specular loss')
        self.parser.add_argument('--lambda_geo', type=float, default=1.0, help='weight of geometric constraint')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.5, help='weight of smoothness constraint')
        self.parser.add_argument('--lambda_light', type=float, default=0.5, help='weight of smoothness constraint')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for perceptual L1 loss')
        self.parser.add_argument('--lambda_GAN', type=float, default=5.0, help='weight of GAN loss')
        self.parser.add_argument('--alpha', type=float, default=1.0, help='weight of TV loss normal')
        self.parser.add_argument('--beta', type=float, default=1.0, help='weight of TV loss spec_albedo')
        self.parser.add_argument('--gamma', type=float, default=1.0, help='weight of TV oss diff_albedo')
  
        
        self.parser.add_argument('--checkpoints_dir', type=str, default='./train/checkpoints')
        self.parser.add_argument('--save_dir', type=str, default='./train/visuals')
        self.parser.add_argument('--obj', type=str, default='ball')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    
        self.parser.add_argument('--benchmark', default='UPS_DiLiGenT_main')
        self.parser.add_argument('--bm_dir',    default='data/datasets/DiLiGenT/pmsData_crop')
        self.parser.add_argument('--have_gt_n', default=True, action='store_true', help='Have GT surface normals?')
    
        

    def collectInfo(self):
        BaseOpts.collectInfo(self)
        self.args.val_keys  += ['in_img_num', 'test_h', 'test_w']
        self.args.bool_keys += ['int_aug', 'test_resc']

    def setDefault(self):
        if self.args.test_h != self.args.crop_h:
            self.args.test_h, self.args.test_w = self.args.crop_h, self.args.crop_w
        self.collectInfo()

    def parse(self):
        BaseOpts.parse(self)
        self.setDefault()
        return self.args
