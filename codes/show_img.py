import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

# 使用百分位数进行标准化
def normalize_band(band):
    min_percentile = np.percentile(band, 2)
    max_percentile = np.percentile(band, 98)
    return np.clip((band - min_percentile) / (max_percentile - min_percentile), 0, 1)


def get_rgb_preview(r, g, b, sar_composite=False):
    if not sar_composite:

        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0:
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)

        return rgb.astype(np.uint8)

    else:
        # generate SAR composite
        HH = r
        HV = g

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6

        rgb = np.dstack((np.zeros_like(HH), HH, HV))

        return rgb.astype(np.uint8)


def getRGBImg(r, g, b, img_size=256):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
    out = np.zeros_like(bands,dtype = np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0 # np.min(band)
        b = 255 # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)

        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t<a] = a
        t[t>b] = b
        out[i, :, :] = t
    return out

def Get2Img(img_fake, img_truth, img_size=256):
    output_img = np.zeros((img_size, 2 * img_size, 3), dtype=np.uint8)
    img_fake =  uint16to8((torch.squeeze(img_fake).cpu().detach().numpy()*10000).astype("uint16")).transpose(1, 2, 0)
    img_truth =  uint16to8((torch.squeeze(img_truth).cpu().detach().numpy()*10000).astype("uint16")).transpose(1, 2, 0)

    img_fake_RGB = getRGBImg(img_fake[:, :, 2], img_fake[:, :, 1], img_fake[:, :, 0], img_size)
    img_truth_RGB = getRGBImg(img_truth[:, :, 2], img_truth[:, :, 1], img_truth[:, :, 0], img_size)

    output_img[:, 0 * img_size:1 * img_size, :] = img_fake_RGB
    output_img[:, 1 * img_size:2 * img_size, :] = img_truth_RGB
    return output_img


#
# # 假设我们有一个多波段图像的路径 "path_to_your_multispectral_image.tif"
# # 为了演示，这里使用一个假设的路径
# path_to_multispectral_image = "E:\Development Program\Pycharm Program\GLF-CR\data\s2_cloudfree\ROIs2017_winter_112_p797.tif"
# pt = 'K:\\test\\gg.tif'
# pt='K:\dataset\ensemble\dsen2\ROIs1158_spring_15_p173.tif'
# pt='K:\dataset\selected_data_folder\s2_cloudy\ROIs1158_spring_15_p173.tif'
#
# path_to_multispectral_image = pt
# # 使用rasterio打开图像
# with rasterio.open(path_to_multispectral_image) as src:
#     # 假设我们想要可视化RGB通道，通常对应于波段4, 3, 2
#     # 注意：实际的波段编号可能根据具体数据集而有所不同，请根据实际情况调整
#     r, g, b = src.read([4, 3, 2])
# # r_normalized = normalize_band(r)
# # g_normalized = normalize_band(g)
# # b_normalized = normalize_band(b)
# #
# # # 组合波段创建RGB图像
# # rgb = np.dstack((r_normalized, g_normalized, b_normalized))
# rgb = get_rgb_preview(r, g, b)
# # 显示图像
# plt.figure(figsize=(10, 10))
# plt.imshow(rgb)
# plt.title("Normalized RGB Visualization of Multispectral Image")
# plt.axis('off')
# plt.show()
