import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import UNet


# ==========================================
# 1. SETTINGS
# ==========================================

LOW_DIR = "dataset/low"
NORMAL_DIR = "dataset/normal"

MODEL_DIR = "models"

# New model filename
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lowlight_unet_sharp_256.pth"
)

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.0001


# ==========================================
# 2. DEVICE
# ==========================================

device = torch.device("cpu")

print("Using device:", device)


# ==========================================
# 3. DATASET
# ==========================================

class LowLightDataset(Dataset):

    def __init__(self, low_dir, normal_dir):

        self.low_dir = low_dir
        self.normal_dir = normal_dir

        self.low_images = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        self.normal_images = sorted([
            f for f in os.listdir(normal_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        if len(self.low_images) != len(self.normal_images):

            raise ValueError(
                "Number of low-light and normal images do not match!"
            )

        print(
            "Number of image pairs:",
            len(self.low_images)
        )

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])


    def __len__(self):

        return len(self.low_images)


    def __getitem__(self, index):

        low_path = os.path.join(
            self.low_dir,
            self.low_images[index]
        )

        normal_path = os.path.join(
            self.normal_dir,
            self.normal_images[index]
        )

        low_image = Image.open(
            low_path
        ).convert("RGB")

        normal_image = Image.open(
            normal_path
        ).convert("RGB")

        low_image = self.transform(
            low_image
        )

        normal_image = self.transform(
            normal_image
        )

        return low_image, normal_image


# ==========================================
# 4. LOAD DATASET
# ==========================================

dataset = LowLightDataset(
    LOW_DIR,
    NORMAL_DIR
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ==========================================
# 5. CREATE U-NET MODEL
# ==========================================

model = UNet()

model = model.to(device)

print(
    "\nU-Net model created successfully!"
)


# ==========================================
# 6. LOSS FUNCTIONS
# ==========================================

mse_criterion = nn.MSELoss()

l1_criterion = nn.L1Loss()


# ==========================================
# 7. SHARPNESS / EDGE LOSS
# ==========================================

def edge_loss(prediction, target):

    # Sobel filter for horizontal edges
    sobel_x = torch.tensor(
        [
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]
        ],
        dtype=torch.float32,
        device=device
    )

    # Sobel filter for vertical edges
    sobel_y = torch.tensor(
        [
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]
        ],
        dtype=torch.float32,
        device=device
    )

    # Repeat filters for RGB channels
    sobel_x = sobel_x.repeat(
        3, 1, 1, 1
    )

    sobel_y = sobel_y.repeat(
        3, 1, 1, 1
    )

    # Predicted image edges
    pred_x = F.conv2d(
        prediction,
        sobel_x,
        padding=1,
        groups=3
    )

    pred_y = F.conv2d(
        prediction,
        sobel_y,
        padding=1,
        groups=3
    )

    # Target image edges
    target_x = F.conv2d(
        target,
        sobel_x,
        padding=1,
        groups=3
    )

    target_y = F.conv2d(
        target,
        sobel_y,
        padding=1,
        groups=3
    )

    # Compare the edges
    loss_x = F.l1_loss(
        pred_x,
        target_x
    )

    loss_y = F.l1_loss(
        pred_y,
        target_y
    )

    return loss_x + loss_y


# ==========================================
# 8. OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 9. CREATE MODEL FOLDER
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================
# 10. TRAINING
# ==========================================

print("\nStarting training...")
print("Image size:", IMAGE_SIZE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print()

best_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_number, (
        low_images,
        normal_images
    ) in enumerate(dataloader):

        # Move images to device
        low_images = low_images.to(device)

        normal_images = normal_images.to(device)


        # Forward pass
        enhanced_images = model(
            low_images
        )


        # MSE loss
        mse_loss = mse_criterion(
            enhanced_images,
            normal_images
        )


        # L1 loss
        l1_loss = l1_criterion(
            enhanced_images,
            normal_images
        )


        # Edge loss for sharpness
        sharp_loss = edge_loss(
            enhanced_images,
            normal_images
        )


        # Combined loss
        loss = (
            0.1 * mse_loss
            +
            0.7 * l1_loss
            +
            0.2 * sharp_loss
        )


        # Clear old gradients
        optimizer.zero_grad()


        # Backpropagation
        loss.backward()


        # Update model weights
        optimizer.step()


        # Add batch loss
        total_loss += loss.item()


        # Print progress
        if (batch_number + 1) % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_number + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.6f}"
            )


    # Average loss
    average_loss = (
        total_loss / len(dataloader)
    )


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {average_loss:.6f}\n"
    )


    # Save best model
    if average_loss < best_loss:

        best_loss = average_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best sharp model saved!")


# ==========================================
# 11. TRAINING COMPLETED
# ==========================================

print("----------------------------------")
print("Training completed successfully!")
print("----------------------------------")

print(
    "\nBest sharp model saved at:"
)

print(
    MODEL_PATH
)