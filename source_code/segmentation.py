import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

class ImprovedUNet(nn.Module):
    """Improved U-Net for vessel segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super(ImprovedUNet, self).__init__()

        # Encoder
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        self.enc4 = self._make_layer(128, 256)

        # Bottleneck
        self.bottleneck = self._make_layer(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._make_layer(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_layer(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_layer(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._make_layer(64, 32)

        # Final output
        self.final = nn.Conv2d(32, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        b = self.dropout(b)

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        output = self.final(d1)
        return torch.sigmoid(output)

# ===== Setup device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Instantiate and load model =====
model = ImprovedUNet().to(device)
model.load_state_dict(torch.load("best_vessel_segmentation_model_state_dict.pth", map_location=device))
model.eval()

# ===== Preprocessing Function =====
def preprocess(image):
    # Resize to 512x512 and normalize to [-1, 1]
    image_resized = cv2.resize(image, (512, 512))
    image_tensor = transforms.ToTensor()(image_resized)  # [0,1]
    image_tensor = (image_tensor - 0.5) / 0.5           # Normalize to [-1,1]
    return image_tensor.unsqueeze(0)                     # Add batch dim

# ===== Segmentation Function =====
def segment_vessels(image_np):
    input_tensor = preprocess(image_np).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return output_mask
