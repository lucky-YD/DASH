import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """将特征图分割成不重叠的窗口"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """将窗口合并回特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim **-0.5

        # 相对位置偏置参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 计算相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 确保索引非负
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B_, num_heads, N, C//num_heads)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww, Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # (B_, num_heads, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.device = device

        # 如果输入分辨率小于窗口大小，则不进行移位
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim).to(device)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        ).to(device)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim).to(device)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop).to(device)

        # 计算注意力掩码并移动到指定设备
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask.to(device))
        else:
            self.register_buffer("attn_mask", None)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与分辨率不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 移位操作
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口分区
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size^2, C

        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size^2, C

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # 残差连接
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, device=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.device = device
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False).to(device)
        self.norm = norm_layer(4 * dim).to(device)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与分辨率不匹配"
        assert H % 2 == 0 and W % 2 == 0, f"输入分辨率 {H}x{W} 不是偶数"

        x = x.view(B, H, W, C)

        # 合并相邻的2x2 patch
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, (H//2)*(W//2), 4*C)  # (B, H/2*W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2*C)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, device=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.device = device

        # 构建多个Swin Transformer Block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                device=device
            )
            for i in range(depth)])

        # 下采样层
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = PatchMerging(
                input_resolution=input_resolution, dim=dim, norm_layer=norm_layer, device=device
            )

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        
        # 保存当前stage的输出特征用于下采样前
        stage_output = x
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x, stage_output

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=32, norm_layer=None, device=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.device = device

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size).to(device)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim).to(device)
        else:
            self.norm = None

    def forward(self, x):
        x = x.to(self.device)
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像尺寸 ({H}x{W}) 与预期 ({self.img_size[0]}x{self.img_size[1]}) 不符"
        
        # 卷积投影
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinBranch(nn.Module):
    """单个Swin Transformer分支，输出每个stage的特征图"""
    def __init__(self, img_size=256, patch_size=4, in_chans=1,
                 embed_dim=32, depths=[2, 2, 6, 2], num_heads=[1, 2, 4, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, device=None,** kwargs):
        super().__init__()
        
        self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 **(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            device=self.device
        )
        num_patches = self.patch_embed.num_patches
        patch_grid = self.patch_embed.patch_grid
        self.patch_grid = patch_grid
        self.H, self.W = patch_grid  # 保存初始patch网格尺寸

        # 位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim, device=self.device))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate).to(self.device)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建各个阶段
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 最后一层不进行下采样
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else None
            layer = BasicLayer(
                dim=int(embed_dim * 2** i_layer),
                input_resolution=(patch_grid[0] // (2 **i_layer),
                                 patch_grid[1] // (2** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                device=self.device
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features).to(self.device)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.patch_embed(x)  # (B, N, C)
        
        # 保存patch嵌入后的特征作为第一个stage特征
        stage_features = []
        H, W = self.patch_grid
        stage_features.append(x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous())
        
        if self.ape and self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 经过各个阶段，收集每个stage的特征
        for i, layer in enumerate(self.layers):
            x, stage_out = layer(x)
            
            # 计算当前stage的特征图尺寸
            size = int(stage_out.shape[1] **0.5)
            # 转换为 [B, C, H, W] 格式并添加到列表
            stage_feat = stage_out.view(x.shape[0], size, size, -1).permute(0, 3, 1, 2).contiguous()
            stage_features.append(stage_feat)

        # 处理最终输出
        x = self.norm(x)  # (B, N, C)
        final_size = int(x.shape[1]** 0.5)
        final_feat = x.view(x.shape[0], final_size, final_size, -1).permute(0, 3, 1, 2).contiguous()
        
        # stage_features包含所有stage的特征，final_feat是最终输出特征
        return final_feat, stage_features

def mmd_loss(x, y, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
    """
    计算两个分布之间的最大均值差异(MMD)损失
    """
    def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) **2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples** 2 - n_samples)
        
        bandwidth /= kernel_mul **(kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul** i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    # 展平特征图为向量
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    
    if kernel_type == 'rbf':
        kernels = guassian_kernel(x_flat, y_flat, kernel_mul=kernel_mul, kernel_num=kernel_num)
    else:
        raise NotImplementedError(f"不支持的核函数类型: {kernel_type}")
    
    batch_size = x.size(0)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
    return loss

class TwoBranchSwinHeartRate(nn.Module):
    """双分支Swin Transformer用于心率预测，带各stage的MMD损失"""
    def __init__(self, img_size=256, patch_size=4, in_chans=1,
                 embed_dim=32, depths=[2, 2, 6, 2], num_heads=[1, 2, 4, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, device=None, mmd_weight=0.1,** kwargs):
        super().__init__()
        
        self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.mmd_weight = mmd_weight  # MMD损失权重
        
        # 创建两个并行的Swin分支
        self.branch1 = SwinBranch(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            device=self.device,
            **kwargs
        )
        
        self.branch2 = SwinBranch(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            device=self.device,** kwargs
        )
        
        # 计算融合后的通道数
        fused_channels = self.branch1.num_features * 2  # 两个分支拼接
        
        # 融合卷积层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, kernel_size=3, padding=1, device=self.device),
            nn.BatchNorm2d(fused_channels // 2, device=self.device),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels // 2, fused_channels // 4, kernel_size=3, padding=1, device=self.device),
            nn.BatchNorm2d(fused_channels // 4, device=self.device),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels // 4, 16, kernel_size=3, padding=1, device=self.device),
            nn.BatchNorm2d(16, device=self.device),
            nn.ReLU(inplace=True),
        )
        
        # 心率预测头
        self.predict_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 32, device=self.device),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, device=self.device)  # 输出单个心率值
        )
        
        self.to(self.device)

    def forward(self, x1, x2):
        """
        x1: 第一个分支的输入 (B, C, H, W)
        x2: 第二个分支的输入 (B, C, H, W)
        返回: 心率预测值, 总MMD损失, 各stage的MMD损失列表
        """
        # 两个分支分别处理输入，获取最终特征和各stage特征
        feat1, stage_feats1 = self.branch1(x1)
        feat2, stage_feats2 = self.branch2(x2)
        
        # 计算每个stage的MMD损失
        stage_mmd_losses = []
        for feat_a, feat_b in zip(stage_feats1, stage_feats2):
            # 对每个stage的特征计算MMD损失
            loss = mmd_loss(feat_a, feat_b)
            stage_mmd_losses.append(loss)
        
        # 计算总MMD损失
        total_mmd_loss = torch.mean(torch.stack(stage_mmd_losses)) * self.mmd_weight
        
        # 拼接特征图 [B, 2*C, 8, 8]
        fused_feat = torch.cat([feat1, feat2], dim=1)
        
        # 卷积融合
        fused_feat = self.fusion_conv(fused_feat)
        
        # 预测心率
        heart_rate = self.predict_head(fused_feat)
        
        return heart_rate, total_mmd_loss, stage_mmd_losses


# 测试模型
if __name__ == "__main__":
    # 指定运行设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = TwoBranchSwinHeartRate(
        img_size=256,
        patch_size=32,
        in_chans=1,
        embed_dim=4,
        depths=[2, 2, 6, 2],
        num_heads=[1, 2, 4, 8],
        window_size=8,
        device=device,
        mmd_weight=0.1
    )
    
    # 生成两个随机输入 (B, C, H, W)
    input1 = torch.randn(64, 1, 256, 256).to(device)
    input2 = torch.randn(64, 1, 256, 256).to(device)
    
    # 前向传播
    output, total_mmd, stage_mmds = model(input1, input2)
    print(f"输入1形状: {input1.shape}")
    print(f"输入2形状: {input2.shape}")
    print(f"输出心率预测形状: {output.shape}")  
    print(f"总MMD损失: {total_mmd.item()}")
    print(f"各stage MMD损失: {[loss.item() for loss in stage_mmds]}")

