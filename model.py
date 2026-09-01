import torch
import torch.nn as nn


# -----------------------------
# Double Convolution Block
# -----------------------------
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# U-Net Model
# -----------------------------
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024, 512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(128, 64)

        # Final layer learns the enhancement details
        self.final = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )


    def forward(self, x):

        # Save original image
        original_image = x


        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )


        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )


        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)


        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        # Predict enhancement residual
        residual = self.final(d1)

        # Add learned enhancement to original image
        output = original_image + residual

        # Keep pixel values between 0 and 1
        output = torch.clamp(
            output,
            0,
            1
        )

        return output


# -----------------------------
# Test the Model
# -----------------------------
if __name__ == "__main__":

    model = UNet()

    print(model)

    # Create a dummy RGB image
    test_image = torch.randn(
        1, 3, 256, 256
    )

    output = model(test_image)

    print("\nInput shape :", test_image.shape)
    print("Output shape:", output.shape)