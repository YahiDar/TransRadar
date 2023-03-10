import torch
import torch.nn as nn
from mvrss.models.adaptive_directional_attention import ADA
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x




class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, k_size = 3):
        super().__init__()
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, k_size=(k_size,3,3),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)

        x2 = self.double_conv_block2(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x3





class TransRad(nn.Module):
    """ 
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames, deform_k = [3, 3, 3, 3, 3, 3, 3, 3], depth = 8, channels = 64):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.rd_encoding_branch = EncodingBranch('range_doppler', k_size = (n_frames//2 + 1))
        self.ra_encoding_branch = EncodingBranch('range_angle', k_size = (n_frames//2 + 1))
        self.ad_encoding_branch = EncodingBranch('angle_doppler', k_size = (n_frames//2 + 1))

        self.pre_axial1 = ConvBlock(128*3,(128*3)//2,1,0,1)
        self.pre_axial2 = ConvBlock((128*3)//2,channels,1,0,1)
        self.axial1 = ADA(dim=channels, depth = depth, reversible = True, deform_k = deform_k)
        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=channels, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=channels, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(128, 128, 2, stride=2)

        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)


    def forward(self, x_rd, x_ra, x_ad):
        # Backbone
        #print('x_rd is:', x_rd.shape)
        #print('x_ra is:', x_ra.shape)
        #print('x_ad is:', x_ad.shape)
        ra_latent = self.ra_encoding_branch(x_ra)
        rd_latent = self.rd_encoding_branch(x_rd)
        ad_latent = self.ad_encoding_branch(x_ad)
        #print('rafeatures,ra latent is:', ra_features.shape, ra_latent.shape)
        #print('rafeatures,ra latent is:', rd_features.shape, rd_latent.shape)
        #print('rafeatures,ra latent is:', ad_features.shape, ad_latent.shape)


        # Latent Space
        # Features join either the RD or the RA branch

        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3 = self.pre_axial2(self.pre_axial1(x3))
        x3 = self.axial1(x3)
        x4_rd = self.rd_single_conv_block2_1x1(x3)
        x4_ra = self.ra_single_conv_block2_1x1(x3)
        # print('x3 is:', x3.shape)
        # print('x3_rd is:', x3_rd.shape)
        # print('x3_ra is:', x3_ra.shape)



        # Parallel decoding branches with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)
        #print('x5_rd is:', x5_rd.shape)
        #print('x5_ra is:', x5_ra.shape)        
        #print('x6_rd is:', x6_rd.shape)
        #print('x6_ra is:', x6_ra.shape)

        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)
        #print('x7_rd is:', x7_rd.shape)
        #print('x7_ra is:', x7_ra.shape)        
        #print('x8_rd is:', x8_rd.shape)
        #print('x8_ra is:', x8_ra.shape)

        # Final 1D convolutions
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)    
        #print('x9_rd  is:', x9_rd.shape)
        #print('x9_ra is:', x9_ra.shape)
            
        #print('end')
        return x9_rd, x9_ra
