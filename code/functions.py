from mmclassification.mmcls.models.backbones.rednet import RedNet
import torch
import mmcv
from collections import OrderedDict
import json


def get_RedNetInvolution(depth:int, pretrained:bool, finetune:bool, checkpoint_name:str, frozen_blocks=[1,2]):
  """
  Get the RedNetInvolution introduced in https://arxiv.org/abs/2103.06255 .

  :param depth: number of layers of the RedNetInvolution; 
                one of [26, 38, 50, 101, 152]
  :param pretrained: load the pretrained checkpoint provided in checkpoint_name
  :param finetune: finetune the network; if False, set the .requires_grad attribute
                   of all the network parameters to False
  :param checkpoint_name: path to the checkpoint file to load
  :param frozen_blocks: number of the blocks of the RedNetInvolution for which disable training

  :return model
  """
  assert depth in [26, 38, 50, 101, 152]
  model = RedNet(depth)

  if pretrained:
    model = load_pretrained_model(model, checkpoint_name)
    if not finetune:
      set_parameter_requires_grad(model, True)
    else:
      set_parameter_requires_grad(model.stem, True)
      for num_block in frozen_blocks:
        set_parameter_requires_grad(eval(f'model.layer{num_block}'), True)
    
  return model


def load_pretrained_model(model, checkpoint_name):
  """
  Load the pretrained checkpoint given the file path checkpoint_name.

  :param checkpoint_name: path to the checkpoint file (.pth)

  :return pretrained model
  """
  state_dict = get_clean_state_dict(checkpoint_name)
  mmcv.runner.checkpoint.load_state_dict(model, state_dict)
  return model


def get_clean_state_dict(checkpoint_name):
  """
  Get the state dictionary for the backbone only. Indeed the checkpoints provided in 
  https://github.com/d-li14/involution#model-zoo contain backbone, head, ...

  :param checkpoint_name: path to the checkpoint file (.pth)

  :return state dictionary of the backbone (RedNetInvolution)
  """
  state_dict = torch.load(checkpoint_name)['state_dict']

  clean_state_dict = OrderedDict()
  for key in state_dict.keys():
    if 'backbone' in key:
      new_key = '.'.join(key.split('.')[1:])
      clean_state_dict[new_key] = state_dict[key]
  
  return clean_state_dict


def set_parameter_requires_grad(model, feature_extracting:bool):
  """
  Set the parameters' attribute .requires_grad to False if feature_extracting=True.

  :param model: model object whose parameters' values have to be frozen
  :param feature_extracting: if True, apply the function, leave the model unchanged otherwise

  :return (no return, the model is modified in place)
  """
  if feature_extracting:
      for param in model.parameters():
          param.requires_grad = False


def parse_configuration(cfg):
  """
  Parse the JSON configuration file by default in the root directory.
  For simplicity, define your configuration file as the default one.
  Customized file paths will require the edit of the UltraFastLaneDetection/model/model.py file
  """
  # with open(conf_file, 'r') as cfg_file:
  #   cfg_dict = json.load(cfg_file)
  
  # pretrained = bool(cfg_dict['pretrained'])
  # finetune = bool(cfg_dict['finetune'])
  # checkpoint_name = cfg_dict['checkpoint_name']

  # return pretrained, finetune, checkpoint_name
  cfg_dict = {}
  cfg_dict['pretrained'] = True if cfg.backbone_checkpoint is not None else False
  cfg_dict['checkpoint_name'] = cfg.backbone_checkpoint
  cfg_dict['finetune'] = cfg.finetune_backbone
  cfg_dict['frozen_blocks'] = cfg.frozen_blocks
  return cfg_dict
    



