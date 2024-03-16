import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FeaturesExtractorRMA(BaseFeaturesExtractor):
    def __init__(self, observation_space, env_name,
                 object_emb_dim=32, 
                 use_depth_base: bool=False, 
                 use_prop_history_base:bool=False,
                 no_obj_emb=False, only_dr=False, sys_iden=False,
                 without_adapt_module=False,
                 inc_obs_noise_in_priv=False) -> None:
        
        self.use_obj_emb = not no_obj_emb
        # priv_info aka env_info
        self.use_priv_info = (not only_dr) and (not without_adapt_module)\
                        and (not use_depth_base) and (not use_prop_history_base)
        self.sys_iden = sys_iden

        if self.use_priv_info:
            if env_name in ['PickCube', 'PickSingleYCB', 'PickSingleEGAD']:
                priv_enc_in_dim = 4 + 3 + 4
            elif env_name in ['TurnFaucet']:
                priv_enc_in_dim = 4 + 1 + 4 + 3
            elif env_name in ['PegInsertionSide']:
                priv_enc_in_dim = 4 + 3
            if self.use_obj_emb:
                priv_enc_in_dim += object_emb_dim * 2
            priv_env_out_dim = priv_enc_in_dim - 4

            if inc_obs_noise_in_priv:
                priv_enc_in_dim += 19 # 9 proprio, 7 obj, 3 ext. force
                priv_env_out_dim += 15
        else:
            priv_env_out_dim = 0

        if self.sys_iden:
            priv_env_out_dim = priv_enc_in_dim

        # the output dim of feature extractor
        features_dim = priv_env_out_dim
        for k, v in observation_space.items():
            if k in ['agent_state', 'object1_state', 'goal_info']:
                features_dim += v._shape[0]
        # if env_name in ['PickCube', 'PickSingleYCB', 'PickSingleEGAD']:
        #     features_dim = (9 + 9 + 7 + 7 + 6 + priv_env_out_dim + 3 * 3)
        # elif env_name in ['TurnFaucet']:
        #     features_dim = (9 + 9 + 7 + 7 + 6 + priv_env_out_dim + 1)
        # elif env_name in ['PegInsertionSide']:
        #     features_dim = (9 + 9 + 7 + 7 + 6 + priv_env_out_dim + 3 * 3 + 5)

        self.use_prop_history_base = use_prop_history_base
        self.use_depth_base = use_depth_base
        # if use depth than it's doesn't use object state and priv info
        if use_depth_base:
            cnn_output_dim = 64
            features_dim += cnn_output_dim + 41 - 6 # cam param + img embedding

        if use_prop_history_base:
            prop_cnn_out_dim = 16
            features_dim += prop_cnn_out_dim
        super().__init__(observation_space, features_dim)

        # instantiate neural networks
        if self.use_depth_base:
            self.img_cnn = DepthCNN(out_dim=cnn_output_dim)
        if use_prop_history_base:
            self.prop_cnn = nn.Sequential(
                                ProprioCNN(in_dim=50),
                                Flatten(),
                                nn.Linear(39*2, prop_cnn_out_dim)
                            )
        if self.use_priv_info:
            self.priv_enc = MLP(units=[128, 128, priv_env_out_dim], 
                            input_size=priv_enc_in_dim)
        if self.use_obj_emb:
            self.obj_id_emb = nn.Embedding(80, object_emb_dim)
            self.obj_type_emb = nn.Embedding(50, object_emb_dim)


    def forward(self, obs_dict, 
                use_pred_e: bool=False, 
                return_e_gt:bool=False,
                pred_e: th.Tensor=None
                ) -> th.Tensor:

        priv_enc_in = []

        if self.use_priv_info:
            if self.use_obj_emb:
                obj_type_emb = self.obj_type_emb(obs_dict['object1_type_id'].int()
                                            ).squeeze(1)
                obj_emb = self.obj_id_emb(obs_dict['object1_id'].int()).squeeze(1)
                priv_enc_in.extend([obj_type_emb, obj_emb])
            priv_enc_in.append(obs_dict['obj1_priv_info'])
            priv_enc_in = th.cat(priv_enc_in,dim=1)

            if self.sys_iden:
                e_gt = priv_enc_in
            else:
                e_gt = self.priv_enc(priv_enc_in)
            if use_pred_e:
                env_vec = pred_e
            else:
                env_vec = e_gt
            obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
                        env_vec, obs_dict['goal_info']]
        else:
            e_gt = None
            if self.use_depth_base:
                obs_list = [obs_dict['agent_state'], obs_dict['goal_info']]
                img_emb = self.img_cnn(obs_dict['image'])
                obs_list.extend([img_emb, obs_dict['camera_param']])
            else:
                obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
                        obs_dict['goal_info']]
            if self.use_prop_history_base:
                prop = self.prop_cnn(obs_dict['prop_act_history'])
                obs_list.append(prop)

        try:
            obs = th.cat(obs_list, dim=-1)
        except:
            print("error")
            breakpoint()

        if return_e_gt:
            return obs, e_gt
        else:
            return obs
# class FeaturesExtractorPickCubeRMA(BaseFeaturesExtractor):
#     '''Only working with PickCube environment'''
#     def __init__(self, observation_space) -> None:
#         priv_enc_in_dim = 4
#         priv_env_out_dim = 3
#         features_dim = (9 + 9 + 7 + 7 + # 7 + agent state: qpos(9), qvel, base, tcp pose
#                     (1 + 7 + 3) + # object 1 state
#                     priv_env_out_dim + # priv_info_projection
#                     3 * 3) # to make the same as default
#         super().__init__(observation_space, features_dim)
#         self.priv_enc = MLP(units=[256, 128, priv_env_out_dim], 
#                             input_size=priv_enc_in_dim)

#     def forward(self, obs_dict, adapt_trn:bool=False, e:th.Tensor=None
#                 ) -> th.Tensor:
#         priv_emb = self.priv_enc(obs_dict['obj1_priv_info'])

#         if adapt_trn:
#             env_vec = e
#         else:
#             env_vec = priv_emb
#         obs = th.cat([obs_dict['agent_state'], obs_dict['object1_state'],
#                        env_vec, obs_dict['goal_info']], 
#                        dim=-1)
#         if adapt_trn:
#             return obs, priv_emb
#         else:
#             return obs
# class FeaturesExtractorMultiRMA(BaseFeaturesExtractor):
#     def __init__(self, observation_space, 
#                 object_emb_dim=32, 
#                 use_depth_base: bool=False, 
#                 normalized_image: bool=False,
#                 no_obj_emb=False, only_dr=False, sys_iden=False,
#                 without_adapt_module=False,
#                 inc_obs_noise_in_priv=False) -> None:
        
#         self.use_obj_emb = not no_obj_emb
#         self.use_priv_info = (not only_dr) and (not without_adapt_module)\
#                                 and (not use_depth_base)# priv_info aka env_info
#         self.sys_iden = sys_iden
#         if self.use_priv_info:
#             priv_enc_in_dim = 4 + 3 + 4
#             if self.use_obj_emb:
#                 priv_enc_in_dim += object_emb_dim * 2
#             priv_env_out_dim = priv_enc_in_dim - 4
#             # else:
#             #     priv_env_out_dim = 3

#             if inc_obs_noise_in_priv:
#                 priv_enc_in_dim += 19 # 9 proprio, 7 obj, 3 ext. force
#                 priv_env_out_dim += 15
#         else:
#             priv_env_out_dim = 0

#         if self.sys_iden:
#             priv_env_out_dim = priv_enc_in_dim

#         # the output dim of feature extractor
#         features_dim = (9 + 9 + 7 + 7 + # 7 + # agent state: qpos(9), qvel, base, tcp pose
#                         6 + # object 1 state
#                         priv_env_out_dim + # priv_info_projection
#                         3 * 3 # goal information
#                         )

#         self.use_depth_base = use_depth_base
#         # if use depth than it's doesn't use object state and priv info
#         if use_depth_base:
#             cnn_output_dim = 64
#             features_dim += cnn_output_dim + 41 - 6 # cam param + img embedding
 
#         super().__init__(observation_space, features_dim)

#         if self.use_depth_base:
#             self.img_cnn = DepthCNN(out_dim=cnn_output_dim)
#         if self.use_priv_info:
#             self.priv_enc = MLP(units=[256, 128, priv_env_out_dim], 
#                             input_size=priv_enc_in_dim)
#         if self.use_obj_emb:
#             self.obj_id_emb = nn.Embedding(80, object_emb_dim)
#             self.obj_type_emb = nn.Embedding(50, object_emb_dim)


#     def forward(self, obs_dict, 
#                 use_pred_e: bool=False, 
#                 return_e_gt:bool=False,
#                 pred_e: th.Tensor=None
#                 ) -> th.Tensor:

#         priv_enc_in = []

#         if self.use_priv_info:
#             if self.use_obj_emb:
#                 obj_type_emb = self.obj_type_emb(obs_dict['object1_type_id'].int()
#                                             ).squeeze(1)
#                 obj_emb = self.obj_id_emb(obs_dict['object1_id'].int()).squeeze(1)
#                 priv_enc_in.extend([obj_type_emb, obj_emb])
#             priv_enc_in.append(obs_dict['obj1_priv_info'])
#             priv_enc_in = th.cat(priv_enc_in,dim=1)

#             if self.sys_iden:
#                 e_gt = priv_enc_in
#             else:
#                 e_gt = self.priv_enc(priv_enc_in)
#             if use_pred_e:
#                 env_vec = pred_e
#             else:
#                 env_vec = e_gt
#             obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
#                         env_vec, obs_dict['goal_info']]
#         else:
#             e_gt = None
#             if self.use_depth_base:
#                 obs_list = [obs_dict['agent_state'], obs_dict['goal_info']]
#                 img_emb = self.img_cnn(obs_dict['image'])
#                 obs_list.extend([img_emb, obs_dict['camera_param']])
#             else:
#                 obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
#                         obs_dict['goal_info']]

#         # if self.use_depth:
#         #     img_emb = self.img_cnn(obs_dict['image'])
#         #     obs_list.append(img_emb)
#         #     obs_list.append(obs_dict['camera_param'])

#         obs = th.cat(obs_list, dim=-1)

#         if return_e_gt:
#             return obs, e_gt
#         else:
#             return obs

# class FeaturesExtractorTurnFaucetRMA(BaseFeaturesExtractor):
#     def __init__(self, observation_space, 
#                  object_emb_dim=32, 
#                 # object_emb_dim=8, 
#                 # priv_env_out_dim=16,
#                 use_depth_base: bool=False, 
#                 normalized_image: bool=False,
#                 no_obj_emb=False, only_dr=False, sys_iden=False,
#                 without_adapt_module=False,
#                 inc_obs_noise_in_priv=False) -> None:
        
#         self.use_obj_emb = not no_obj_emb
#         self.use_priv_info = (not only_dr) and (not without_adapt_module)\
#                                 and (not use_depth_base)# priv_info aka env_info
#         self.sys_iden = sys_iden

#         if self.use_priv_info:
#             priv_enc_in_dim = 4 + 1 + 4 + 3
#             if self.use_obj_emb:
#                 priv_enc_in_dim += object_emb_dim * 2
#             priv_env_out_dim = priv_enc_in_dim - 4

#             if inc_obs_noise_in_priv:
#                 priv_enc_in_dim += 19 # 9 proprio, 7 obj, 3 ext. force
#                 priv_env_out_dim += 15
#         else:
#             priv_env_out_dim = 0

#         if self.sys_iden:
#             priv_env_out_dim = priv_enc_in_dim

#         # the output dim of feature extractor
#         features_dim = (9 + 9 + 7 + 7 + # 7 + # agent state: qpos(9), qvel, base, tcp pose
#                         6 + # object 1 state
#                         priv_env_out_dim + # priv_info_projection 12+64-4=72
#                         1 # goal information
#                         )

#         self.use_depth_base = use_depth_base
#         # if use depth than it's doesn't use object state and priv info
#         if use_depth_base:
#             cnn_output_dim = 64
#             features_dim += cnn_output_dim + 41 - 6 # cam param + img embedding
 
#         super().__init__(observation_space, features_dim)

#         if self.use_depth_base:
#             self.img_cnn = DepthCNN(out_dim=cnn_output_dim)
#         if self.use_priv_info:
#             self.priv_enc = MLP(units=[256, 128, priv_env_out_dim], 
#                             input_size=priv_enc_in_dim)
#         if self.use_obj_emb:
#             self.obj_id_emb = nn.Embedding(80, object_emb_dim)
#             self.obj_type_emb = nn.Embedding(50, object_emb_dim)


#     def forward(self, obs_dict, 
#                 use_pred_e: bool=False, 
#                 return_e_gt:bool=False,
#                 pred_e: th.Tensor=None
#                 ) -> th.Tensor:

#         priv_enc_in = []

#         if self.use_priv_info:
#             if self.use_obj_emb:
#                 obj_type_emb = self.obj_type_emb(obs_dict['object1_type_id'].int()
#                                             ).squeeze(1)
#                 obj_emb = self.obj_id_emb(obs_dict['object1_id'].int()).squeeze(1)
#                 priv_enc_in.extend([obj_type_emb, obj_emb])
#             priv_enc_in.append(obs_dict['obj1_priv_info'])
#             priv_enc_in = th.cat(priv_enc_in,dim=1)

#             if self.sys_iden:
#                 e_gt = priv_enc_in
#             else:
#                 e_gt = self.priv_enc(priv_enc_in)
#             if use_pred_e:
#                 env_vec = pred_e
#             else:
#                 env_vec = e_gt
#             obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
#                         env_vec, obs_dict['goal_info']]
#         else:
#             e_gt = None
#             if self.use_depth_base:
#                 obs_list = [obs_dict['agent_state'], obs_dict['goal_info']]
#                 img_emb = self.img_cnn(obs_dict['image'])
#                 obs_list.extend([img_emb, obs_dict['camera_param']])
#             else:
#                 obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
#                         obs_dict['goal_info']]

#         # if self.use_depth:
#         #     img_emb = self.img_cnn(obs_dict['image'])
#         #     obs_list.append(img_emb)
#         #     obs_list.append(obs_dict['camera_param'])

#         obs = th.cat(obs_list, dim=-1)

#         if return_e_gt:
#             return obs, e_gt
#         else:
#             return obs

# class FeaturesExtractorPegInsertRMA(BaseFeaturesExtractor):
#     def __init__(self, observation_space, object_emb_dim=8, 
#                  use_depth_base: bool=False, normalized_image: bool=False,
#                  no_obj_emb=False, only_dr=False, sys_iden=False,
#                  without_adapt_module=False,
#                  inc_obs_noise_in_priv=False) -> None:
        
#         self.use_obj_emb = not no_obj_emb
#         self.use_priv_info = (not only_dr) and (not without_adapt_module)\
#                                 and (not use_depth_base)# priv_info aka env_info
#         self.sys_iden = sys_iden
#         if self.use_priv_info:
#             priv_enc_in_dim = 4 + 3 #+ 4
#             # if self.use_obj_emb:
#             #     priv_enc_in_dim += object_emb_dim * 2
#             #     priv_env_out_dim = priv_enc_in_dim - 4
#             # else:
#             priv_env_out_dim = priv_enc_in_dim - 4

#             if inc_obs_noise_in_priv:
#                 priv_enc_in_dim += 19 # 9 proprio, 7 obj, 3 ext. force
#                 priv_env_out_dim += 15
#         else:
#             priv_env_out_dim = 0

#         if self.sys_iden:
#             priv_env_out_dim = priv_enc_in_dim

#         # the output dim of feature extractor
#         features_dim = (9 + 9 + 7 + 7 + # 7 + # agent state: qpos(9), qvel, base, tcp pose
#                         6 + # object 1 state
#                         priv_env_out_dim + # priv_info_projection
#                         3 * 3 + 5 # goal information +5 for quat and hole dim
#                         )

#         self.use_depth_base = use_depth_base
#         # if use depth than it's doesn't use object state and priv info
#         if use_depth_base:
#             cnn_output_dim = 64
#             features_dim += cnn_output_dim + 41 - 6 # cam param + img embedding

#         super().__init__(observation_space, features_dim)

#         if self.use_depth_base:
#             self.img_cnn = DepthCNN(out_dim=cnn_output_dim)
#         if self.use_priv_info:
#             # self.priv_enc = MLP(units=[256, 128, priv_env_out_dim], 
#             self.priv_enc = MLP(units=[128, 128, priv_env_out_dim], 
#                             input_size=priv_enc_in_dim)
#         if self.use_obj_emb:
#             self.obj_id_emb = nn.Embedding(80, object_emb_dim)
#             self.obj_type_emb = nn.Embedding(50, object_emb_dim)


#     def forward(self, obs_dict, 
#                 use_pred_e: bool=False, 
#                 return_e_gt:bool=False,
#                 pred_e: th.Tensor=None
#                 ) -> th.Tensor:
#         priv_enc_in = []

#         if self.use_priv_info:
#             # if self.use_obj_emb:
#             #     obj_type_emb = self.obj_type_emb(obs_dict['object1_type_id'].int()
#             #                                 ).squeeze(1)
#             #     obj_emb = self.obj_id_emb(obs_dict['object1_id'].int()).squeeze(1)
#             #     priv_enc_in.extend([obj_type_emb, obj_emb])
#             priv_enc_in.append(obs_dict['obj1_priv_info'])
#             priv_enc_in = th.cat(priv_enc_in, dim=1)

#             if self.sys_iden:
#                 e_gt = priv_enc_in
#             else:
#                 e_gt = self.priv_enc(priv_enc_in)
#             if use_pred_e:
#                 env_vec = pred_e
#             else:
#                 env_vec = e_gt
#             obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
#                         env_vec, obs_dict['goal_info']]
#         else:
#             e_gt = None
#             if self.use_depth_base:
#                 obs_list = [obs_dict['agent_state'], obs_dict['goal_info']]
#                 img_emb = self.img_cnn(obs_dict['image'])
#                 obs_list.extend([img_emb, obs_dict['camera_param']])
#             else:
#                 obs_list = [obs_dict['agent_state'], obs_dict['object1_state'],
#                         obs_dict['goal_info']]

#         # if self.use_depth:
#         #     img_emb = self.img_cnn(obs_dict['image'])
#         #     obs_list.append(img_emb)
#         #     obs_list.append(obs_dict['camera_param'])

#         obs = th.cat(obs_list, dim=-1)

#         if return_e_gt:
#             return obs, e_gt
#         else:
#             return obs

class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.LayerNorm(output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class AdaptationNet(nn.Module):
    def __init__(self, observation_space=None, in_dim=50, out_dim=16, 
                 use_depth=False):

        super(AdaptationNet, self).__init__()
        self.use_depth = use_depth

        dep_cnn_output_dim = 0
        camera_param_dim = 0
        
        if use_depth:
            dep_cnn_output_dim = 64
            camera_param_dim = 32 + 9 # 16 + 16 + 9
        else:
            dep_cnn_output_dim = 0
            camera_param_dim = 0
        self.perc_cnn = DepthCNN(out_dim=dep_cnn_output_dim)
        self.prop_cnn = ProprioCNN(in_dim)
        self.fc = nn.Linear(39 * 2 + camera_param_dim + dep_cnn_output_dim, 
                            out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # print x
        prop, perc, cparam = x['prop'], x['perc'], x['cparam']
        # print(f"prop mean {prop.mean():.3f} min {prop.min():.3f} max {prop.max()}")
        # print(f"perc mean {perc.mean():.3f} min {perc.min():.3f} max {perc.max()}")
        # print(f"cparam mean {cparam.mean():.3f} min {cparam.min():.3f} max {cparam.max()}")
        prop = self.prop_cnn(prop)
        # print(f"new prop mean {prop.mean():.3f} min {prop.min():.3f} max {prop.max()}")
        obs = [prop]
        if self.use_depth:
            perc = self.perc_cnn(perc)
            # print(f"new perc mean {prop.mean():.3f} min {perc.min():.3f} max {perc.max()}")
            obs.extend([perc, cparam])
        x = self.fc(th.cat(obs, dim=-1))
        x = self.fc2(self.relu(x))
        # print(f"pred_e mean {x.mean():.3f} min {x.min():.3f} max {x.max():.3f}")
        # print("")
        return x

class DepthCNN(nn.Module):
    def __init__(self, out_dim):
        super(DepthCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, out_dim)
        # self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(256 * 4 * 4, 512)
        # self.relu4 = nn.ReLU()
        # self.fc2 = nn.Linear(512, out_dim)


    def forward(self, x):
        # x has shape [n_env, times, 1, h, w]
        x = x.squeeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 128 * 4 * 4)
        # x = x.view(-1, 256 * 4 * 4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x

def calc_activation_shape_1d(
        dim, ksize, stride=1, dilation=1, padding=0
    ):
        def shape_each_dim():
            odim_i = dim + 2 * padding - dilation * (ksize - 1) - 1
            return (odim_i / stride) + 1
        return int(shape_each_dim())     
class ProprioCNN(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(39, 39),
            nn.LayerNorm(39),
            nn.ReLU(inplace=True),
            nn.Linear(39, 39),
            nn.LayerNorm(39),
            nn.ReLU(inplace=True),
        )
        # add layerNorm after each conv1d
        ln_shape = calc_activation_shape_1d(in_dim, 9, 2)
        # ln_shape = 21
        ln1 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 7, 2)
        # ln_shape = 17
        ln2 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 5, 1)
        # ln_shape = 13
        ln3 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 3, 1)
        # ln_shape = 11
        ln4 = nn.LayerNorm((39, ln_shape))
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(39, 39, (9,), stride=(2,)),
            ln1,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (7,), stride=(2,)),
            ln2,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (5,), stride=(1,)),
            ln3,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (3,), stride=(1,)),
            ln4,
            nn.ReLU(inplace=True),
        )
    def forward(self, prop):
        prop = self.channel_transform(prop)  # (N, 50, 39)
        prop = prop.permute((0, 2, 1))  # (N, 39, 50)
        prop = self.temporal_aggregation(prop)  # (N, 39, 3)
        prop = prop.flatten(1)
        return prop



def calc_activation_shape_2d(
    dim, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0)
):
    def shape_each_dim(i):
        odim_i = dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
        return (odim_i / stride[i]) + 1

    return int(np.floor(shape_each_dim(0))), int(np.floor(shape_each_dim(1)))
