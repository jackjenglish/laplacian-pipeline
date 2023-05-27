import torch
import torch.nn as nn
import time

# https://github.com/linmc86/Deep-Reformulated-Laplacian-Tone-Mapping/blob/master/laplacianet/utils/utils_lap_pyramid.py
class Lap_Pyramid(nn.Module):
  def __init__(self, num_high=3):
    super(Lap_Pyramid, self).__init__()

    self.num_high = num_high
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.kernel = self.gauss_kernel(self.device)

  def gauss_kernel(self, device=torch.device('cuda'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

  def downsample(self, x):
    return x[:, :, ::2, ::2]

  # https://colab.research.google.com/drive/1VXY-9pJlh0JvAzgoDCuSbaVlg-1smSl-#scrollTo=1uO64yY_2Mfu
  def upsample(self, x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return self.conv_gauss(x_up, 4 * self.kernel)

  # useed in upsample & pyramid_decom
  def conv_gauss(self, img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

  # Laplacian pyramid decomposition downsampling 
  def pyramid_decom(self, img):
    current = img.to(self.device)
    pyr = []
    gyr = [current]
    for _ in range(self.num_high):
        filtered = self.conv_gauss(current, self.kernel)
        down = self.downsample(filtered)
        up = self.upsample(down)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        diff = diff.to(self.device)
        down = down.to(self.device)
        pyr.append(diff)
        gyr.append(down)
        current = down
    pyr.append(current)
    return pyr, gyr

  # Laplacian pyramid reconstruction upsampling 
  def pyramid_recons(self, pyr):
    image = pyr[-1]
    for level in reversed(pyr[:-1]):
      up = self.upsample(image)
      if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
        up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
      image = up + level
    return image