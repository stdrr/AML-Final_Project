# DATA
dataset='Tusimple'
data_root = '../tusimple/'

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9
force_lr_value = False # custom

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '26'
griding_num = 100
use_aux = True
backbone_checkpoint = 'checkpoints RedNet/rednet26-4948f75f.pth' # custom
finetune_backbone = False # custom
frozen_blocks = None # custom

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = '../checkpoints/tusimple/'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4