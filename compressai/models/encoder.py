import torch
import torch.nn as nn
from basicsr.archs.atd_arch import ATDB
from compressai.layers import GDN#改

class VAEEncoderWithATDB(nn.Module):
    def __init__(self):
        super(VAEEncoderWithATDB, self).__init__()
        dim1 = 192
        dim2 = 320
        input_resolution1 = (64, 64)  # conv2的输出分辨率
        input_resolution2 = (32, 32)  # conv3的输出分辨率
        depths = 3  # 每个 ATDB 包含6个 transformer 层
        num_heads = 8
        window_size = 8
        category_size = 128
        num_tokens = 64
        reducted_dim = 10
        convffn_kernel_size = 5
        mlp_ratio = 2.0
        qkv_bias = True
        norm_layer = nn.LayerNorm

        self.conv1 = nn.Conv2d(3, dim1, 3, stride=2, padding=1)
        self.act1 = GDN(dim1)
        self.conv2 = nn.Conv2d(dim1, dim1, 3, stride=2, padding=1)
        self.act2 = GDN(dim1)

        self.atdb1 = ATDB(
            dim=dim1,
            idx=0,
            input_resolution=input_resolution1,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            category_size=category_size,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            img_size=256,
            patch_size=1,
            resi_connection='1conv'
        )

        self.conv3 = nn.Conv2d(dim1, dim2, 3, stride=2, padding=1)
        self.act3 = GDN(dim2)

        self.atdb2 = ATDB(
            dim=dim2,
            idx=1,
            input_resolution=input_resolution2,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            category_size=category_size,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            img_size=256,
            patch_size=1,
            resi_connection='1conv'
        )
        self.conv4 = nn.Conv2d(dim2, dim2, 3, stride=2, padding=1)
        self.act4 = GDN(dim2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        #print(f"x shape after conv2: {x.shape}")
        x_size = (x.shape[2], x.shape[3])
        params = {}  # 根据需要传递参数
        #将x从（batch_size, channels, height, width)转换为（batch_size, seq_len, channels)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        attn_mask = self.atdb1.calculate_mask(x_size).to(x.device)
        params = {
            'rpi_sa': self.atdb1.relative_position_index_SA,  # 确保 rpi_sa 被正确传递
            'attn_mask': attn_mask
        }

        x = self.atdb1(x, x_size, params)
        #print(f"x shape after atdb1: {x.shape}")

        # 恢复为 (batch_size, channels, height, width)
        b, seq_len, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, *x_size)
        #print(f"x shape restored after atdb1: {x.shape}")

        x = self.conv3(x)
        x = self.act3(x)
        x_size = (x.shape[2], x.shape[3])
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        attn_mask = self.atdb2.calculate_mask(x_size).to(x.device)
        params = {
            'rpi_sa': self.atdb2.relative_position_index_SA,
            'attn_mask': attn_mask
        }

        x = self.atdb2(x, x_size, params)

        # 恢复为 (batch_size, channels, height, width)
        b, seq_len, c = x.shape
        x = x.permute(0, 2, 1).view(b, c, *x_size)
        #print(f"x shape restored after atdb2: {x.shape}")

        x = self.conv4(x)
        x = self.act4(x)

        return x

"""
# 示例使用
input_image = torch.randn(1, 3, 256, 256)
encoder = VAEEncoderWithATDB()
output = encoder(input_image)
print(output.shape)
"""

class VAEDecoderWithATDB(nn.Module):
    def __init__(self):
        super(VAEDecoderWithATDB, self).__init__()
        dim1 = 192
        dim2 = 320
        output_resolution1 = (64, 64)  # 对应反卷积后特征图的分辨率
        output_resolution2 = (128, 128)  # 对应反卷积后特征图的分辨率
        depths = 3
        num_heads = 8
        window_size = 8
        category_size = 128
        num_tokens = 64
        reducted_dim = 10
        convffn_kernel_size = 5
        mlp_ratio = 2.0
        qkv_bias = True
        norm_layer = nn.LayerNorm

        self.deconv1 = nn.ConvTranspose2d(dim2,dim2,3,stride=2, padding=1, output_padding=1)
        self.act1 = GDN(dim2)

        self.atdb1 = ATDB(
            dim=dim2,
            idx=0,
            input_resolution=output_resolution1,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            category_size=category_size,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            img_size=256,
            patch_size=1,
            resi_connection='1conv'
        )

        self.deconv2 = nn.ConvTranspose2d(dim2, dim1, 3, stride=2, padding=1, output_padding=1)
        self.act2 = GDN(dim1, inverse=True)

        self.atdb2 = ATDB(
            dim=dim1,
            idx=1,
            input_resolution=output_resolution2,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            category_size=category_size,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            img_size=256,
            patch_size=1,
            resi_connection='1conv'
        )

        self.deconv3 = nn.ConvTranspose2d(dim1, dim1, 3, stride=2, padding=1, output_padding=1)
        self.act3 = GDN(dim1, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(dim1, 3, 3, stride=2, padding=1, output_padding=1)
        self.act4 = GDN(dim1, inverse=True)

    def forward(self, x):

        x = self.deconv1(x)
        x = self.act1(x)
        #print(f"x shape after deconv1: {x.shape}")

        x_size = (x.shape[2], x.shape[3])
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        # ATDB1
        attn_mask = self.atdb1.calculate_mask(x_size).to(x.device)
        params = {
            'rpi_sa': self.atdb1.relative_position_index_SA,
            'attn_mask': attn_mask
        }
        x = self.atdb1(x, x_size, params)

        # 还原形状
        x = x.permute(0, 2, 1).view(b, c, h, w)
        #print(f"x shape restored after atdb1: {x.shape}")

        x = self.deconv2(x)
        x = self.act2(x)
        #print(f"x shape after deconv2: {x.shape}")

        x_size = (x.shape[2], x.shape[3])
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)

        # ATDB2
        attn_mask = self.atdb2.calculate_mask(x_size).to(x.device)
        params = {
            'rpi_sa': self.atdb2.relative_position_index_SA,
            'attn_mask': attn_mask
        }
        x = self.atdb2(x, x_size, params)


        # 还原形状
        x = x.permute(0, 2, 1).view(b, c, h, w)
        #print(f"x shape restored after atdb2: {x.shape}")

        x = self.deconv3(x)
        x = self.act3(x)
        #print(f"x shape after deconv3: {x.shape}")

        x= self.deconv4(x)


        return x

"""
# 示例使用
input_tensor = torch.randn(1, 320, 16, 16)  # 潜在表示输入
decoder = VAEDecoderWithATDB()
output = decoder(input_tensor)
print(output.shape)  # 应为 (1, 3, 256, 256)
"""