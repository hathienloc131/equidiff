# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from matplotlib.pyplot import axis
import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange
from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup
from .backbone import build_backbone, build_backbone_equi
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
from equi_diffpo.model.common.rotation_transformer import RotationTransformer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class EquivariantEncoder(ModuleAttrMixin):
    def __init__(self,
        encoder,
        num_queries=1,
        hidden_dim=128,
        final_dim=32,
        N=8,
        initialize=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.final_dim = final_dim
        self.N = N
        self.initialize = initialize
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.token_type = enn.FieldType(self.group, self.hidden_dim * [self.group.regular_repr])
        
        self.encoder = encoder
        
        # extra cls token embedding
        self.cls_embed = nn.Embedding(1, self.hidden_dim)
        
        self.encoder_joint_proj = enn.Linear(
            self.getJointFieldType(),
            self.token_type,
        )
        self.encoder_action_proj = enn.Linear(
            self.getOutFieldType(),
            self.token_type,
        )
        
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
        self.latent_proj = nn.Linear(hidden_dim, self.final_dim*2) # project hidden state to latent std, var
        
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+N * (1+num_queries), hidden_dim)) # [CLS], qpos, a_seq
    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 
    
    def forward(self, obs, action, is_pad=None):
        batch_size = action.shape[0]
        t = action.shape[1]

        joint_features = self.getJointGeometricTensor(obs) # (bs, 8)
        joint_embed = self.encoder_joint_proj(joint_features).tensor # (bs, n * hidden_dim)
        joint_embed = rearrange(joint_embed, "b (n d) -> b n d", b=batch_size, n=self.N) # (bs, t * N, hidden_dim)
        
        action_features = self.getActionGeometricTensor(action)
        action_embed = self.encoder_action_proj(action_features).tensor  # (bs*t, hidden_dim)
        action_embed = rearrange(action_embed, "(b t) (n d) -> b (t n) d", b=batch_size, t=t, n=self.N) # (bs, t * N, hidden_dim)

        # cls token
        cls_embed = self.cls_embed.weight # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(batch_size, 1, 1) # (bs, 1, hidden_dim)
        encoder_input = torch.cat([cls_embed, joint_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
        
        encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
        # do not mask cls token
        cls_joint_is_pad = torch.full((batch_size, 1 + self.N), False).to(joint_embed.device) # False: not a padding
        is_pad = torch.cat([cls_joint_is_pad, is_pad.repeat(1, self.N)], axis=1)  # (bs, seq+1)
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        # query model
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
        encoder_output = encoder_output[0] # take cls output only
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, :self.final_dim]
        logvar = latent_info[:, self.final_dim:]
        latent_sample = reparametrize(mu, logvar)
    
        return latent_sample, mu, logvar, is_pad

    def getJointFieldType(self):
        return enn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # pos 3, rot 6
            + 3 * [self.group.trivial_repr], # gripper 1
        )
        
    def getOutFieldType(self):
        return enn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8
            + 2 * [self.group.trivial_repr], # 2
        )
    
    def getJointGeometricTensor(self, obs):
        ee_pos = obs['robot0_eef_pos'] # (bs, t, 3)
        ee_quat = obs['robot0_eef_quat'] # (bs, t, 4)
        ee_q = obs["robot0_gripper_qpos"] # (bs, t, 2)
        
        ee_rot = self.get6DRotation(ee_quat)
        pos_xy = ee_pos[:, 0:2] # 2
        pos_z = ee_pos[:, 2:3] # 1
        joint_features = torch.cat(
            [
                pos_xy,
                ee_rot[:, 0:1], # 1
                ee_rot[:, 3:4], # 1
                ee_rot[:, 1:2], # 1
                ee_rot[:, 4:5], # 1
                ee_rot[:, 2:3], # 1
                ee_rot[:, 5:6], # 1
                pos_z, # 1
                ee_q, # 2
            ],
            dim=1
        )

        return enn.GeometricTensor(joint_features, self.getJointFieldType())

    def getActionGeometricTensor(self, act):
        batch_size, t, *_ = act.shape
        xy = act[:, :, 0:2]
        z = act[:, :, 2:3]
        rot = act[:, :, 3:9]
        g = act[:, :, 9:]

        cat = torch.cat(
            (
                xy.reshape(batch_size, t, 2),
                rot[:, :, 0].reshape(batch_size, t, 1),
                rot[:, :, 3].reshape(batch_size, t, 1),
                rot[:, :, 1].reshape(batch_size, t, 1),
                rot[:, :, 4].reshape(batch_size, t, 1),
                rot[:, :, 2].reshape(batch_size, t, 1),
                rot[:, :, 5].reshape(batch_size, t, 1),
                z.reshape(batch_size, t, 1),
                g.reshape(batch_size, t, 1),
            ),
            dim=2,
        )
        
        cat = rearrange(cat, "b t d -> (b t) d")  # (bs*t, d)
        return enn.GeometricTensor(cat, self.getOutFieldType())

class EquivariantDecoder(ModuleAttrMixin):
    def __init__(self, backbones, transformer, num_queries, camera_names, N=8):
        super().__init__()
        self.N = N
        self.camera_names = camera_names
        self.transformer = transformer
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.image_group = gspaces.rot2dOnR2(N)
        hidden_dim = transformer.d_model
        self.token_type = enn.FieldType(self.group, hidden_dim * [self.group.regular_repr])
        self.query_type = enn.FieldType(self.group, hidden_dim * [self.group.trivial_repr])
        self.action_head = enn.Linear(
            self.token_type
            , self.getOutFieldType())
 

        # Better: Make query embeddings equivariant
        self.query_embed_base = nn.Embedding(num_queries, hidden_dim)
        self.query_proj = enn.Linear(
            self.query_type,
            self.token_type
        )

        self.backbones = nn.ModuleList(backbones)
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)

        self.input_proj_robot_state = enn.Linear(
            self.getJointFieldType(),
            self.token_type,
        )
        self.quaternion_to_sixd = RotationTransformer('quaternion', 'rotation_6d')
        
        self.additional_pos_embed = nn.Embedding(1 + N, hidden_dim)
        
    def get6DRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]]) 
    
    def getJointFieldType(self):
        return enn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # pos 3, rot 6
            + 3 * [self.group.trivial_repr], # gripper 1
        ) 
        
    def getOutput(self, conv_out, bs):
        conv_out = conv_out.tensor
        xy = conv_out[:, 0:2]
        cos1 = conv_out[:, 2:3]
        sin1 = conv_out[:, 3:4]
        cos2 = conv_out[:, 4:5]
        sin2 = conv_out[:, 5:6]
        cos3 = conv_out[:, 6:7]
        sin3 = conv_out[:, 7:8]
        z = conv_out[:, 8:9]
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
        action = rearrange(action, "(b t) d -> b t d", b=bs, d=action.shape[1]) # (bs, N, d)
        return action
    
    def forward(self, obs, image, latent_input):
        """
        image: batch, channel, height, width
        latent_input: batch, hidden_dim
        """
        bs = image.shape[0]
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id, (cam_name, is_equi) in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # proprioception features
        qpos_features = self.getJointGeometricTensor(obs)
        proprio_input = self.input_proj_robot_state(qpos_features).tensor
        proprio_input = rearrange(proprio_input, "b (n d) -> b n d", b=proprio_input.shape[0], n=self.N) # (bs, N, hidden_dim)

        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        latent_input = latent_input.unsqueeze(1)
        query_embed = self.query_proj(
            enn.GeometricTensor(self.query_embed_base.weight, self.query_type)
        ).tensor
        
        query_embed = rearrange(query_embed, "b (n d) -> (b n) d", b=query_embed.shape[0], n=self.N) # (bs* N, hidden_dim)
        
        hs = self.transformer(src, None, query_embed, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        hs = rearrange(hs, "b (t n) d -> (b t) (n d)", b=bs, n=self.N) # (bs * N, hidden_dim)
        hs = enn.GeometricTensor(hs, self.token_type) 
        a_hat = self.action_head(hs)
        a_hat = self.getOutput(a_hat, bs)
        return a_hat
    
    def getOutFieldType(self):
        return enn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8
            + 2 * [self.group.trivial_repr], # 2
        )

    def getJointGeometricTensor(self, obs):
        ee_pos = obs['robot0_eef_pos'] # (bs, t, 3)
        ee_quat = obs['robot0_eef_quat'] # (bs, t, 4)
        ee_q = obs["robot0_gripper_qpos"] # (bs, t, 2)
        
        ee_rot = self.get6DRotation(ee_quat)
        pos_xy = ee_pos[:, 0:2] # 2
        pos_z = ee_pos[:, 2:3] # 1

        joint_features = torch.cat(
            [
                pos_xy,
                ee_rot[:, 0:1], # 1
                ee_rot[:, 3:4], # 1
                ee_rot[:, 1:2], # 1
                ee_rot[:, 4:5], # 1
                ee_rot[:, 2:3], # 1
                ee_rot[:, 5:6], # 1
                pos_z, # 1
                ee_q, # 2
            ],
            dim=1
        )

        return enn.GeometricTensor(joint_features, self.getJointFieldType())

class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, N):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.N = N
        hidden_dim = transformer.d_model
        
        self.equivariant_decoder = EquivariantDecoder(
            backbones,
            transformer,
            num_queries=num_queries,
            camera_names=camera_names,
            N=N,
        )

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.equivariant_encoder = EquivariantEncoder(
            encoder,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            final_dim=self.latent_dim,
            N=N,
            initialize=True,
        )

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        
        

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, *_ = image.shape
        ### Obtain latent z from action sequence
        if is_training:
            latent_sample, mu, logvar, is_pad = self.equivariant_encoder(qpos, action=actions, is_pad=is_pad)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(image.device)
            latent_input = self.latent_out_proj(latent_sample)

        a_hat = self.equivariant_decoder(
            obs=qpos, 
            image=image, 
            latent_input=latent_input, 
        )
        return a_hat, None, [mu, logvar]



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 11 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    
    backbones = []
    new_camera_names = []
    for cam_nam in args.camera_names:
        if cam_nam in args.camera_equi:
            print("build backbone equi", cam_nam)
            backbone = build_backbone_equi(args)
            new_camera_names.append([cam_nam , 1])
        else:
            print("build backbone", cam_nam)
            backbone = build_backbone(args)
            new_camera_names.append([cam_nam , 0])
        backbones.append(backbone)
            
    print(len(backbones))
    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=new_camera_names,
        N=args.N,  # number of groups
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

