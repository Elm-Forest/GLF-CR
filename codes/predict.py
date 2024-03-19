import argparse
import os

import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt
from rasterio.transform import from_origin

from dataloader import AlignedDataset, get_train_val_test_filelists
from metrics import PSNR, SSIM
from net_CR_RDN import RDN_residual_CR


##########################################################
def test(CR_net, opts):
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    print(test_filelist)

    data = AlignedDataset(opts, test_filelist)
    print(data)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz, shuffle=True)

    iters = 0
    for inputs in dataloader:
        cloudy_data = inputs['cloudy_data'].cuda()
        cloudfree_data = inputs['cloudfree_data'].cuda()
        SAR_data = inputs['SAR_data'].cuda()
        file_name = inputs['file_name'][0]

        pred_cloudfree_data = CR_net(cloudy_data, SAR_data)

        psnr_13 = PSNR(pred_cloudfree_data, cloudfree_data)

        ssim_13 = SSIM(pred_cloudfree_data, cloudfree_data).item()

        print(iters, '  psnr_13:', format(psnr_13, '.4f'), '  ssim_13:', format(ssim_13, '.4f'))
        vis(pred_cloudfree_data)
        vis(cloudy_data, msg='cloudy')
        vis(cloudfree_data, msg='cloudfree')
        # return pred_cloudfree_data


# ROIs1158_spring_134_p59.tif
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--input_data_folder', type=str, default='K:\dataset\selected_data_folder')
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)  # only useful when is_use_cloudmask=True

    opts = parser.parse_args()

    CR_net = RDN_residual_CR(opts.crop_size).cuda()
    checkpoint = torch.load('../ckpt/CR_net.pth')
    CR_net.load_state_dict(checkpoint['network'])

    CR_net.eval()
    for _, param in CR_net.named_parameters():
        param.requires_grad = False

    test(CR_net, opts)


def vis(pred, msg='predicted'):
    # 将张量从CUDA移动到CPU，并转换为NumPy数组
    pred_np = pred.cpu().numpy()

    # 选择RGB通道
    R_channel = pred_np[0, 3, :, :]
    G_channel = pred_np[0, 2, :, :]
    B_channel = pred_np[0, 1, :, :]

    # 标准化通道到[0, 1]范围内以便显示
    # R_channel_normalized = (R_channel - R_channel.min()) / (R_channel.max() - R_channel.min())
    # G_channel_normalized = (G_channel - G_channel.min()) / (G_channel.max() - G_channel.min())
    # B_channel_normalized = (B_channel - B_channel.min()) / (B_channel.max() - B_channel.min())
    from codes.show_img import get_rgb_preview
    rgb_image = get_rgb_preview(R_channel, G_channel, B_channel)
    # 组合RGB通道制作3通道RGB图像
    # rgb_image = np.stack([R_channel_normalized, G_channel_normalized, B_channel_normalized], axis=-1)

    # 绘制图像
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.title(msg)
    plt.axis('off')  # 关闭坐标轴标号和刻度
    plt.show()


def save_imgs(pred):
    # 假设pred是模型的输出，形状为(1, 13, 128, 128)
    # 首先将其转换为NumPy数组，并调整形状
    pred_np = pred.squeeze().cpu().numpy()  # 假设pred是在GPU上，先移动到CPU然后转换
    pred_np = np.moveaxis(pred_np, 0, -1)  # 重排维度为(128, 128, 13)

    # 图像左上角的经纬度坐标
    left = 100.0  # 左上角的经度
    top = 0.0  # 左上角的纬度

    # 每个像素的大小（以米为单位），这里假设经度和纬度方向的像素大小相同
    pixel_size = 10

    # 创建transform
    transform = from_origin(left, top, pixel_size, pixel_size)
    # 定义新GeoTIFF文件的属性
    # transform = from_origin(西南角经度, 西南角纬度, 像素大小(经度), 像素大小(纬度))
    new_dataset = rasterio.open(
        'output.tif',  # 输出文件路径
        'w',
        driver='GTiff',
        height=pred_np.shape[0],
        width=pred_np.shape[1],
        count=13,  # 波段数
        dtype=pred_np.dtype,
        crs='+proj=latlong',  # 假设使用WGS84坐标系，根据实际情况调整
        transform=transform
    )

    # 将数据写入新文件
    for i in range(1, 14):
        new_dataset.write(pred_np[:, :, i - 1], i)

    new_dataset.close()


if __name__ == "__main__":
    main()
