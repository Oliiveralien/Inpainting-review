from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for displays 
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=3000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='if true, do not save intermediate training results to web') 
        self.parser.add_argument('--tf_log', action='store_true', help='if true, use tensorboard logging') 

        # for training 
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load?')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, or test')
        self.parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=90, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam') 

        # for discriminators 
        self.parser.add_argument('--num_D', type=int, default=2, help='# of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=2, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=0.01, help='weight for feature matching loss') 
        self.parser.add_argument('--lambda_vgg', type=float, default=0.05, help='weight for vgg feature matching loss') 
        self.parser.add_argument('--lambda_l1', type=float, default=5.0, help='weight for l1 loss')
        self.parser.add_argument('--lambda_tv', type=float, default=0.1, help='weight for tv loss') 
        self.parser.add_argument('--lambda_style', type=float, default=80.0, help='weight for style loss')      
        self.parser.add_argument('--lambda_gan', type=float, default=0.001, help='weight for g gan loss')           
        self.parser.add_argument('--no_ganFeat_loss', action='store_false', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        # use least square GAN
        self.parser.add_argument('--no_lsgan', action='store_true', default=True, help='do *not* use least square GAN, if false, use vanilla GAN') 
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images') 

        self.isTrain = True

