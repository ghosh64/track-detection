import torch.nn as nn
import torch

class WeBACNN(nn.Module):
    """
    Weighted Branch Aggregation Model
    """
    def __init__(self):
        super().__init__()

        # Branch 1: (larger stride/pool size/filter size) (General Feature)
        self.b1_conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=3),
                        nn.LeakyReLU(),
                        nn.AvgPool2d(kernel_size=3, stride=3),
                        nn.Dropout2d(),
                        nn.BatchNorm2d(64),
                        nn.Upsample(size=128)
                        )

        self.b1_conv2 = nn.Sequential(
                        nn.Conv2d(64, 256, kernel_size=3, stride=3),
                        nn.LeakyReLU(),
                        nn.AvgPool2d(kernel_size=3, stride=3),
                        nn.BatchNorm2d(256)
                        )

        self.b1_resize = nn.Sequential(
                            nn.ConvTranspose2d(256, 64, kernel_size=3, padding=(2,2)),
                            nn.Upsample(size=64),
                            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=(2,2)),
                            nn.Upsample(size=320)
                        )

        # Branch 2: local, smaller convolution kernels
        self.b2_conv1 = nn.Sequential(
                        nn.Conv2d(3, 128, kernel_size=2, stride=1),
                        nn.LeakyReLU(),
                        nn.AvgPool2d(kernel_size=2, stride=1),
                        nn.BatchNorm2d(128)
                        )

        self.b2_conv2 = nn.Sequential(
                        nn.Conv2d(128, 512, kernel_size=2, stride=2, padding=(2,2)),
                        nn.LeakyReLU(),
                        nn.AvgPool2d(kernel_size=2, stride=1),
                        nn.BatchNorm2d(512)
                        )

        self.b2_resize = nn.Sequential(
                        nn.ConvTranspose2d(512, 128, kernel_size=3, stride=3, padding=(3,3)),
                        nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=(2,2)), # only 1 feature channel --> grayscale
                        nn.Upsample(320)
                        )

        # Final smoothen layer
        self.fin_conv = nn.Sequential(
                        nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1),
                        nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=(2,2)),
                        nn.BatchNorm2d(1)
                        )

        self.sig = nn.Sigmoid()

    def forward(self, input):
        # Branch 1
        global_result = self.b1_conv1(input)
        global_result = self.b1_conv2(global_result)
        global_result = self.b1_resize(global_result)

        # Branch 2
        # Crop image (1/6 from top)
        height = input.shape[2]
        local_result = input[:,:,height//6:,:]

        local_result = self.b2_conv1(input)
        local_result = self.b2_conv2(local_result)

        # Fill cropped part with 0
        fill_tensor = torch.zeros((local_result.shape[0], local_result.shape[1], (local_result.shape[3]-local_result.shape[2]), local_result.shape[3]))
        device=torch.device('cuda:0')
        fill_tensor = fill_tensor.to(device)
        local_result = torch.cat((fill_tensor, local_result), 2) # concatenate top with 0

        local_result = self.b2_resize(local_result)

        # Crop global_result and local_result into top, middle, bottom
        global_result_top = global_result[:,:,:80,:] # 0 ~ 79 (top 1/4)
        global_result_mid = global_result[:,:,80:160,:] # middle 1/4
        global_result_bot = global_result[:,:,160:,:] # bottom 1/2

        local_result_top = local_result[:,:,:80,:] # 0 ~ 79 (top 1/4)
        local_result_mid = local_result[:,:,80:160,:] # middle 1/4
        local_result_bot = local_result[:,:,160:,:] # bottom 1/2

        # Apply weights to each branch and add them up
        input_top = 0.7 * global_result_top + 0.3 * local_result_top # global feature heavy (7:3)
        input_mid = 0.4 * global_result_mid + 0.6 * local_result_mid # 4:6
        input_bot = 0.3 * global_result_bot + 0.7 * local_result_bot # local feature heavy (3:7)

        # Concatenate top, mid, and bottom
        input = torch.cat((input_top, input_mid, input_bot), 2)

        # Smoothen convolutional layer
        final_result = self.fin_conv(input)

        return final_result, global_result, local_result