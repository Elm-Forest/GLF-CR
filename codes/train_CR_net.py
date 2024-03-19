from dataloader import *
from generic_train_test import *
from model_CR_net import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--input_data_folder', type=str, default='K:\dataset\selected_data_folder')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)  # only useful when is_use_cloudmask=True
parser.add_argument('--data_list_filepath', type=str, default='E:\Development Program\Pycharm Program\dsen2-cr\csv\datasetfilelist.csv')

parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=5, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--log_freq', type=int, default=1)
parser.add_argument('--save_model_dir', type=str, default='./checkpoints',
                    help='directory used to store trained networks')

parser.add_argument('--is_test', type=bool, default=False)

parser.add_argument('--gpu_ids', type=str, default='0')

opts = parser.parse_args()
print_options(opts)

##===================================================##
##****************** choose gpu *********************##
##===================================================##
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
seed_torch()

train_filelist, _, _ = get_train_val_test_filelists(opts.data_list_filepath)

train_data = AlignedDataset(opts, train_filelist)

train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_sz, shuffle=True)

##===================================================##
##****************** Create model *******************##
##===================================================##
model = ModelCRNet(opts)


##===================================================##
##**************** Train the network ****************##
##===================================================##
class Train(Generic_train_test):
    def decode_input(self, data):
        return data


Train(model, opts, train_dataloader).train()
