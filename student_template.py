import torch
import torch.nn as nn
import numpy as np
import os

def _single_image_to_chw(image) -> torch.Tensor:
    """
    Convert one image to tensor [1, H, W].
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    image = image.detach().clone()

    if image.ndim == 2:
        image = image.unsqueeze(0)

    elif image.ndim == 3:
        # CHW format: [C, H, W]
        if image.shape[0] in (1, 3):
            pass

        # HWC format: [H, W, C]
        elif image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)

        else:
            raise ValueError(f"Cannot interpret image shape: {tuple(image.shape)}")

    else:
        raise ValueError(f"Expected 2D or 3D image, got shape: {tuple(image.shape)}")

    image = image.float()

    # RGB -> grayscale
    if image.shape[0] == 3:
        image = image.mean(dim=0, keepdim=True)

    # Normalize raw pixel values to [0, 1]
    if float(image.max().detach().cpu()) > 1.5:
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val + 1e-8)

    return image


def _images_to_bchw(images, device="cpu") -> torch.Tensor:
    """
    Convert image/list/batch to tensor [B, 1, H, W].
    """
    if isinstance(images, torch.Tensor):
        x = images.detach().clone()

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.ndim == 3:
            # [C, H, W]
            if x.shape[0] in (1, 3):
                x = x.unsqueeze(0)

            # [H, W, C]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1).unsqueeze(0)

            # [B, H, W]
            else:
                x = x.unsqueeze(1)

        elif x.ndim == 4:
            # [B, H, W, C]
            if x.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)

            # [B, C, H, W]
            elif x.shape[1] in (1, 3):
                pass

            else:
                raise ValueError(f"Cannot interpret batch shape: {tuple(x.shape)}")

        else:
            raise ValueError(f"Cannot interpret tensor shape: {tuple(x.shape)}")

        x = x.float()

        # RGB -> grayscale
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        # Normalize raw pixel values to [0, 1]
        if float(x.max().detach().cpu()) > 1.5:
            min_val = x.amin(dim=(1, 2, 3), keepdim=True)
            max_val = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - min_val) / (max_val - min_val + 1e-8)

        return x.to(device)

    if isinstance(images, np.ndarray):
        return _images_to_bchw(torch.from_numpy(images), device=device)

    tensors = [_single_image_to_chw(img) for img in images]
    x = torch.stack(tensors, dim=0)

    return x.to(device)

class SimplePneumoniaClassifier(nn.Module):
    """
    A pneumonia classification model that takes X-ray images and predicts whether pneumonia is present.
    
    Your model should follow this basic structure, but you are free to modify the internal architecture.
    """
    
    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Initialize your model.
        
        Args:
            checkpoint_dir (str): Directory where model checkpoints will be saved
        """
        super(SimplePneumoniaClassifier, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2)
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1)
        )

        self.register_buffer("threshold_default", torch.tensor(0.50))
        self.register_buffer("threshold_male", torch.tensor(0.50))
        self.register_buffer("threshold_female", torch.tensor(0.50))
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw logits for training with BCEWithLogitsLoss.
        """
        x = x.float()

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.ndim == 3:
            x = x.unsqueeze(1)

        elif x.ndim != 4:
            raise ValueError(f"Expected [B, 1, H, W], got {tuple(x.shape)}")

        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)

        if float(x.max().detach().cpu()) > 1.5:
            min_val = x.amin(dim=(1, 2, 3), keepdim=True)
            max_val = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - min_val) / (max_val - min_val + 1e-8)

        features = self.features(x)
        pooled = self.pool(features)
        logits = self.classifier(pooled)

        return logits
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, height, width] 
                             containing grayscale X-ray images
        
        Returns:
            torch.Tensor: Output tensor with shape [batch_size, 1] containing probabilities 
                         of pneumonia (values between 0 and 1)
        """
        logits = self.forward_logits(x)
        probabilities = torch.sigmoid(logits)
        return probabilities
        pass
    
    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """
        Load model weights from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            
        Returns:
            dict: Checkpoint data including 'epoch' and other training metadata
        """
        # Load checkpoint (the file should contain 'model_state_dict' and other info)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load the state dictionary into the model
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Return the entire checkpoint for additional information
        return checkpoint
    
    def predict(self, image, device='cpu'):
        """
        Make a prediction for a single image.
        
        Args:
            image: Input image (can be numpy array or tensor)
            device: Device to use for computation ('cpu', 'cuda', or 'mps')
            
        Returns:
            dict: Dictionary containing:
                - 'probability': Float value between 0 and 1
                - 'class': Binary class (0 or 1)
                - 'label': String label ('Normal' or 'Pneumonia')
        """
        self.to(device)
        self.eval()

        x = _images_to_bchw([image], device=device)

        with torch.no_grad():
            probability = float(self(x).item())

        threshold = float(self.threshold_default.item())
        pred_class = int(probability >= threshold)

        return {
            "probability": probability,
            "class": pred_class,
            "label": "Pneumonia" if pred_class == 1 else "Normal"
        }
        pass

def get_importance_heatmaps(model: SimplePneumoniaClassifier, 
                           images: list, 
                           window_size: int = 32, 
                           stride: int = 16) -> list:
    """
    Generate occlusion sensitivity maps for a batch of images.
    
    This function should create heatmaps that highlight regions important for 
    the model's prediction. For pneumonia cases, the heatmap should focus on
    the areas of the image that contain the pneumonia opacity.
    
    Args:
        model: Trained PyTorch model (SimplePneumoniaClassifier)
        images: List or tensor of input images
        window_size: Size of the occlusion window (default: 32)
        stride: Stride of the sliding window (default: 16)
        
    Returns:
        heatmaps: List of numpy arrays, each representing a sensitivity map 
                 with the same height and width as the original image.
                 Values should be normalized between 0 and 1.
    """

    device = next(model.parameters()).device
    model.eval()

    batch = _images_to_bchw(images, device=device)
    heatmaps = []

    for idx in range(batch.shape[0]):
        image = batch[idx:idx + 1]
        _, _, h, w = image.shape

        with torch.no_grad():
            base_prob = float(model(image).item())

        heat = torch.zeros((h, w), device=device)
        counts = torch.zeros((h, w), device=device)

        fill_value = image.mean()

        max_y = max(h - window_size, 0)
        max_x = max(w - window_size, 0)

        y_positions = list(range(0, max_y + 1, stride))
        x_positions = list(range(0, max_x + 1, stride))

        if len(y_positions) == 0:
            y_positions = [0]

        if len(x_positions) == 0:
            x_positions = [0]

        if y_positions[-1] != max_y:
            y_positions.append(max_y)

        if x_positions[-1] != max_x:
            x_positions.append(max_x)

        occluded_images = []
        positions = []

        for y in y_positions:
            for x in x_positions:
                y2 = min(y + window_size, h)
                x2 = min(x + window_size, w)

                occluded = image.clone()
                occluded[:, :, y:y2, x:x2] = fill_value

                occluded_images.append(occluded.squeeze(0))
                positions.append((y, y2, x, x2))

        chunk_size = 64

        for start in range(0, len(occluded_images), chunk_size):
            end = start + chunk_size
            chunk = torch.stack(occluded_images[start:end], dim=0)

            with torch.no_grad():
                occluded_probs = model(chunk).detach().view(-1)

            for local_i, occluded_prob in enumerate(occluded_probs):
                y, y2, x, x2 = positions[start + local_i]

                importance = max(base_prob - float(occluded_prob.item()), 0.0)

                heat[y:y2, x:x2] += importance
                counts[y:y2, x:x2] += 1.0

        heat = heat / counts.clamp_min(1.0)

        heat_np = heat.detach().cpu().numpy()
        heat_np = heat_np - heat_np.min()

        if heat_np.max() > 1e-8:
            heat_np = heat_np / heat_np.max()

        heatmaps.append(heat_np.astype(np.float32))

    return heatmaps
    pass

def fair_predict(model: SimplePneumoniaClassifier, 
                images: list, 
                sex_attribute: list = None) -> list:
    """
    Make fair predictions on demographic attributes.
    
    Args:
        model: Trained model (SimplePneumoniaClassifier)
        images: List or tensor of input images
        sex_attribute: List of sex attributes corresponding to images ('M' or 'F')
                      Can be None if demographic information is not available
        
    Returns:
        List of prediction dictionaries, each containing:
            - 'probability': Raw probability from model (float between 0 and 1)
            - 'threshold': Threshold used for this prediction
            - 'class': Binary prediction (0 or 1) after applying threshold
            - 'label': String label ('Normal' or 'Pneumonia')
    """
    device = next(model.parameters()).device
    model.eval()

    batch = _images_to_bchw(images, device=device)

    if sex_attribute is None:
        sex_attribute = [None] * batch.shape[0]

    if len(sex_attribute) != batch.shape[0]:
        raise ValueError("sex_attribute length must match number of images")

    with torch.no_grad():
        probabilities = model(batch).detach().cpu().numpy().reshape(-1)

    results = []

    for probability, sex in zip(probabilities, sex_attribute):
        sex_norm = str(sex).upper() if sex is not None else ""

        if sex_norm == "M":
            threshold = float(model.threshold_male.item())
        elif sex_norm == "F":
            threshold = float(model.threshold_female.item())
        else:
            threshold = float(model.threshold_default.item())

        pred_class = int(float(probability) >= threshold)

        results.append({
            "probability": float(probability),
            "threshold": threshold,
            "class": pred_class,
            "label": "Pneumonia" if pred_class == 1 else "Normal"
        })

    return results
def calibrate_fair_thresholds_from_arrays(
    model: SimplePneumoniaClassifier,
    probabilities,
    labels,
    sex_attribute,
    grid_size: int = 181
):
    """
    Choose male/female thresholds using validation predictions.
    This helper is used in train.ipynb.
    """
    probs = np.asarray(probabilities).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    sex = np.asarray([str(s).upper() for s in sex_attribute])

    thresholds = np.linspace(0.05, 0.95, grid_size)

    base_preds = (probs >= 0.50).astype(int)
    base_positive_rate = base_preds.mean()

    best_score = float("inf")
    best_m = 0.50
    best_f = 0.50
    best_metrics = {}

    male_mask = sex == "M"
    female_mask = sex == "F"

    if male_mask.sum() == 0 or female_mask.sum() == 0:
        model.threshold_male.fill_(0.50)
        model.threshold_female.fill_(0.50)

        return {
            "threshold_male": 0.50,
            "threshold_female": 0.50,
            "prediction_rate_gap": None,
            "tpr_gap": None,
            "note": "Could not calibrate because one sex group is missing."
        }

    for tm in thresholds:
        for tf in thresholds:
            preds = np.zeros_like(labels)

            other_mask = ~(male_mask | female_mask)

            preds[male_mask] = (probs[male_mask] >= tm).astype(int)
            preds[female_mask] = (probs[female_mask] >= tf).astype(int)
            preds[other_mask] = (probs[other_mask] >= 0.50).astype(int)

            pr_m = preds[male_mask].mean()
            pr_f = preds[female_mask].mean()
            pr_gap = abs(pr_m - pr_f)

            male_positives = male_mask & (labels == 1)
            female_positives = female_mask & (labels == 1)

            if male_positives.sum() == 0:
                tpr_m = 0.0
            else:
                tpr_m = preds[male_positives].mean()

            if female_positives.sum() == 0:
                tpr_f = 0.0
            else:
                tpr_f = preds[female_positives].mean()

            tpr_gap = abs(tpr_m - tpr_f)

            overall_positive_rate = preds.mean()

            score = (
                pr_gap
                + tpr_gap
                + 0.25 * abs(overall_positive_rate - base_positive_rate)
            )

            if score < best_score:
                best_score = score
                best_m = float(tm)
                best_f = float(tf)

                best_metrics = {
                    "threshold_male": best_m,
                    "threshold_female": best_f,
                    "prediction_rate_male": float(pr_m),
                    "prediction_rate_female": float(pr_f),
                    "prediction_rate_gap": float(pr_gap),
                    "tpr_male": float(tpr_m),
                    "tpr_female": float(tpr_f),
                    "tpr_gap": float(tpr_gap),
                    "overall_positive_rate": float(overall_positive_rate),
                    "base_positive_rate": float(base_positive_rate)
                }

    model.threshold_male.fill_(best_m)
    model.threshold_female.fill_(best_f)

    return best_metrics
    pass
