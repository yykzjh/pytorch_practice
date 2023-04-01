import time
import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualModule(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super(ResidualModule, self).__init__()
        # 原生模块
        # 普通3X3卷积
        self.conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        # 点级1X1卷积
        self.conv_2d_pointwise = nn.Conv2d(in_channels, out_channels, 1)

        # 算子融合模块
        # 将点级1x1的卷积扩展为同样效果的3X3的卷积
        self.conv_2d_for_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        # 按照self.conv_2d_pointwise中的weight构造self.conv_2d_for_pointwise的参数weight
        conv_2d_for_pointwise_weight = F.pad(self.conv_2d_pointwise.weight.data, [1, 1, 1, 1, 0, 0, 0, 0]) # 2*2*1*1->2*2*3*3
        # 用新的weight和原来的bias替换掉self.conv_2d_for_pointwise中的weight和bias
        self.conv_2d_for_pointwise.weight = nn.Parameter(conv_2d_for_pointwise_weight)
        self.conv_2d_for_pointwise.bias = self.conv_2d_pointwise.bias

        # 将x恒等部分也扩展为一个3X3的卷积
        self.conv_2d_for_identity = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        # 创建一个3X3全为0的Tensor
        all_zero_tesnor = torch.zeros((kernel_size, kernel_size))
        # 创建一个3X3，只有中间一个元素为1，其他元素都为0的Tensor
        center_one_tensor = F.pad(torch.ones(1, 1), [1, 1, 1, 1])
        # 构建self.conv_2d_for_identity的weight的flatten形式的list
        weight_list = []
        for i in range(out_channels):
            for j in range(in_channels):
                if j == i:
                    weight_list.append(center_one_tensor)
                else:
                    weight_list.append(all_zero_tesnor)
        # 堆叠后并调整维度
        conv_2d_for_identity_weight = torch.stack(weight_list, dim=0)
        conv_2d_for_identity_weight = torch.reshape(conv_2d_for_identity_weight, (out_channels, in_channels, kernel_size, kernel_size))
        # 用新的weight和原来的bias替换掉self.conv_2d_for_identity中的weight和bias
        self.conv_2d_for_identity.weight = nn.Parameter(conv_2d_for_identity_weight)
        self.conv_2d_for_identity.bias = nn.Parameter(torch.zeros(out_channels))

        # 融合三个3X3的卷积算子
        self.residual_conv_perator_fusion = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        # 将三个3X3卷积模块的weight张量相加
        self.residual_conv_perator_fusion.weight = nn.Parameter(
            self.conv_2d.weight.data + self.conv_2d_for_pointwise.weight.data + self.conv_2d_for_identity.weight.data
        )
        # 将三个3X3卷积模块的bias张量相加
        self.residual_conv_perator_fusion.bias = nn.Parameter(
            self.conv_2d.bias.data + self.conv_2d_for_pointwise.bias.data + self.conv_2d_for_identity.bias.data
        )



    def forward(self, x):
        return self.conv_2d(x) + self.conv_2d_pointwise(x) + x



    def fusion(self, x):
        return self.residual_conv_perator_fusion(x)





if __name__ == '__main__':

    model = ResidualModule()

    x = torch.randn((4, 3, 224, 224))

    t1 = time.time()
    output_native = model(x)
    t2 = time.time()

    t3 = time.time()
    output_fusion = model.fusion(x)
    t4 = time.time()

    print(output_native[0, 0, ...])
    print(output_fusion[0, 0, ...])

    print("计算结果是否相似：", torch.all(torch.isclose(output_native, output_fusion, rtol=1e-05, atol=1e-06)))
    print("原生残差模块计算时间：", (t2 - t1)*1000, "ms")
    print("算子融合残差模块计算时间：", (t4 - t3)*1000, "ms")


























