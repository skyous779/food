# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""global args for Transformer in Transformer(TNT)"""
import argparse
import ast
import os
import sys

import yaml

from src.configs import parser as _parser

args = None


def parse_arguments():
    """parse_arguments"""
    global args
    parser = argparse.ArgumentParser(description="MindSpore TNT Training")

    parser.add_argument("-a", "--arch", metavar="ARCH", default="Conformer-Ti", help="model architecture")
    parser.add_argument("--accumulation_step", default=1, type=int, help="accumulation step")
    parser.add_argument("--amp_level", default="O2", choices=["O0", "O1", "O2", "O3"], help="AMP Level")
    parser.add_argument("--batch_size", default=256, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all Devices on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--dataset_sink_mode", type=ast.literal_eval, default=False, help="dataset sink mode")
    parser.add_argument("--beta", default=[0.9, 0.999], type=lambda x: [float(a) for a in x.split(",")],
                        help="beta for optimizer")
    parser.add_argument("--with_ema", default=False, type=ast.literal_eval, help="training with ema")
    parser.add_argument("--ema_decay", default=0.99996, type=float, help="ema decay")
    parser.add_argument('--data_url', default="./data", help='location of data.')
    parser.add_argument("--device_id", default=0, type=int, help="device id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend"], type=str)
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--file_format", type=str, choices=["AIR", "MINDIR"], default="MINDIR", help="file format")
    parser.add_argument("--in_chans", default=3, type=int)
    parser.add_argument("--is_dynamic_loss_scale", default=1, type=int, help="is_dynamic_loss_scale ")
    parser.add_argument("--keep_checkpoint_max", default=20, type=int, help="keep checkpoint max num")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="adamw")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")

    parser.add_argument("--graph_mode", default=0, type=int, help="graph mode with 0, python with 1")

    parser.add_argument("--mix_up", default=0., type=float, help="mix up")
    parser.add_argument("--mlp_ratio", help="mlp ", default=4., type=float)
    parser.add_argument("-j", "--num_parallel_workers", default=20, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--warmup_length", default=0, type=int, help="number of warmup iterations")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warm up learning rate")
    parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W",
                        help="weight decay (default: 0.05)", dest="weight_decay")
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--lr_scheduler", default="cosine_annealing", help="schedule for the learning rate.")
    parser.add_argument("--lr_adjust", default=30, type=float, help="interval to drop lr")
    parser.add_argument("--lr_gamma", default=0.97, type=int, help="multistep multiplier")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--config_file", help="Config file to use (see configs dir)", default=None, required=False)
    parser.add_argument("--seed", default=42, type=int, help="seed for initializing training. ")
    parser.add_argument("--save_every", default=10, type=int, help="save every ___ epochs(default:2)")
    parser.add_argument("--label_smoothing", type=float, help="label smoothing to use, default 0.1", default=0.1)
    parser.add_argument("--image_size", default=224, help="image Size.", type=int)
    parser.add_argument("--crop_pct", default=0.875, help="crop pct.", type=float)
    parser.add_argument('--train_url', default="./", help='location of training outputs.')
    parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="whether run on modelarts")

    parser.add_argument("--run_openi", type=ast.literal_eval, default=False, help="Whether run on openi")
    parser.add_argument("--run_intelligent", type=ast.literal_eval, default=False, help="Whether run on openi")
    parser.add_argument('--ckpt_url', default='./ckpt_url', help='model to save/load')
    parser.add_argument('--result_url', default='./result', help='result folder to save/load')

    parser.add_argument("--deploy", type=ast.literal_eval, default=False, help="whether run deploy")
    parser.add_argument('--fold_train', default=0, type = int,  help='use fold train')
    parser.add_argument('--fold', default=0, type = int,  help='the rank of fold train')
    
    parser.add_argument('--eval', type = int, default=0)
    args = parser.parse_args()


    get_config()


def get_config():

    global args
    override_args = _parser.argv_to_vars(sys.argv)
    # backup
    data_url = args.data_url
    train_url = args.train_url
    # load yaml file
    if args.run_modelarts:
        import moxing as mox
        if not args.config_file.startswith("obs:/"):
            args.config_file = "obs:/" + args.config_file
        with mox.file.File(args.config_file, 'r') as f:
            yaml_txt = f.read()
    if args.run_openi:
        current_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_path)
        config_file_path = os.path.join(src_dir, "configs/{}.yaml".format(args.arch))
        if not os.path.exists(config_file_path):
            raise ValueError("model config file: {} not exists".format(config_file_path))
        yaml_txt = open(config_file_path).read()
    else:
        yaml_txt = open(args.config_file).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config_file}")

    args.__dict__.update(loaded_yaml)
    # restore
    if args.run_openi:
        args.data_url = data_url
        args.train_url = train_url
    print(args)

    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
        os.environ["RANK_SIZE"] = str(args.device_num)

def run_args():
    """run and get args"""
    global args
    if args is None:
        parse_arguments()


run_args()
# !python train.py --config src/configs/twin_s.yaml --device_id 0 --device_target Ascend --data_url ../imagenet