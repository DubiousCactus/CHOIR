import torch.nn as nn

from model.graspTTA.CVAE import VAE
from model.graspTTA.pointnet_encoder import PointNetEncoder


class affordanceNet(nn.Module):
    def __init__(
        self,
        mano_params_dim: int,
        obj_inchannel=3,
        cvae_encoder_sizes=[1024, 512, 256],
        cvae_latent_size=64,
        cvae_decoder_sizes=[1024, 256, 37],
        cvae_condition_size=1024,
        **kwargs,
    ):
        super(affordanceNet, self).__init__()

        self.obj_inchannel = obj_inchannel
        # cvae_encoder_sizes[0] = mano_params_dim
        self.cvae_encoder_sizes = cvae_encoder_sizes
        self.cvae_latent_size = cvae_latent_size
        cvae_decoder_sizes[-1] = mano_params_dim
        self.cvae_decoder_sizes = cvae_decoder_sizes
        self.cvae_condition_size = cvae_condition_size
        self.mano_params_dim = mano_params_dim

        self.obj_encoder = PointNetEncoder(
            global_feat=True, feature_transform=False, channel=self.obj_inchannel
        )
        self.hand_encoder = PointNetEncoder(
            global_feat=True, feature_transform=False, channel=3
        )
        self.cvae = VAE(
            encoder_layer_sizes=self.cvae_encoder_sizes,
            latent_size=self.cvae_latent_size,
            decoder_layer_sizes=self.cvae_decoder_sizes,
            condition_size=self.cvae_condition_size,
        )

    def forward(self, obj_pc, hand_xyz):
        """
        :param obj_pc: [B, 3+n, N1]
        :param hand_xyz: [B, 3, 778]
        :return: reconstructed hand vertex
        """
        B = obj_pc.size(0)
        obj_glb_feature, _, _ = self.obj_encoder(obj_pc)  # [B, 1024]
        hand_glb_feature, _, _ = self.hand_encoder(hand_xyz)  # [B, 1024]

        if self.training:
            recon, means, log_var, z = self.cvae(
                hand_glb_feature, obj_glb_feature
            )  # recon: [B, MANO_PARAMS_DIM]
            recon = recon.contiguous().view(B, self.mano_params_dim)
            return recon, means, log_var, z
        else:
            # inference
            recon = self.cvae.inference(B, obj_glb_feature)
            recon = recon.contiguous().view(B, self.mano_params_dim)
            return recon, None, None, None

    def inference(self, obj_pc):
        B = obj_pc.size(0)
        obj_glb_feature, _, _ = self.obj_encoder(obj_pc)
        recon = self.cvae.inference(B, obj_glb_feature)
        recon = recon.contiguous().view(B, self.mano_params_dim)
        return recon
