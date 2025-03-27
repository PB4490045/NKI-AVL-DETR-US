import os
import numpy as np
import torch
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import train_test_split

# For model 
import torch.nn as nn
import timm

from torch.profiler import record_function

def load_data(input_path):
    # Construct full paths for each file
    
    images_path = os.path.join(input_path, 'X_collection_320_train.npy')
    labels_path = os.path.join(input_path, 'y_collection_320_train_adjusted.npy')
    
    # Load the .npy files
    images = np.load(images_path)
    labels = np.load(labels_path)
    
    return images, labels

def get_bounding_boxes(labels):
    """
    Extract bounding boxes from a segmentation mask using OpenCV.
    Assumes segmentation is a binary mask (or thresholded appropriately).
    Returns a list of bounding boxes in (x, y, w, h) format.
    """
    # Convert mask to uint8 if needed
    seg_uint8 = labels.astype(np.uint8)
    # Find external contours
    contours, _ = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Compute bounding rectangles for each contour
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    
    return boxes # Returns tuple (x, y, w, h)

# hier boundary boxes maken? Of in class?

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.x = torch.tensor(images, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.bboxes = [get_bounding_boxes(label) for label in labels]

    def __len__(self):
        # Return the size of the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Return the image and its label
        image = self.x[idx]
        boxes = self.bboxes[idx]
        
        return image, boxes
    
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding boxes.
    """
    images = torch.stack([item[0] for item in batch])  # Stack images into a batch
    bboxes = [torch.tensor(item[1], dtype=torch.float32) for item in batch]  # Keep bounding boxes as a list
    print(f"Batch size: {len(images)}, Number of bounding boxes: {[len(b) for b in bboxes]}")
    return images, bboxes
    
# =============================================================================

class testDETR(nn.Module):
    def __init__(self, config, hidden_dim=None, num_encoder_layers=1, num_decoder_layers=1, num_preds=1):
        super().__init__()
        # constants
        self.patch_dim = config.PATCH_DIM
        self.mean = torch.tensor([1]).to(device)
        self.std = torch.tensor([0.147]).to(device)
        # backbone to generate feature maps from input frame
        self.backbone = timm.create_model(
            # 'efficientvit_b0.r224_in1k',
            'convnextv2_atto.fcmae_ft_in1k',
            pretrained=config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool='',
        )
        # backbone hidden dimension
        self.hidden_dim_backbone = self.backbone.feature_info[-1]['num_chs']
        # dimension in detection transformer
        if hidden_dim is None:
            self.hidden_dim = self.backbone.feature_info[-1]['num_chs']
        else:
            self.hidden_dim = hidden_dim
        # compression layer
        self.proj = nn.Sequential(
                nn.Conv2d(self.hidden_dim_backbone, self.hidden_dim, kernel_size=1),
            )
        # Generic transformer for query with feature map interaction
        self.transformer = nn.Transformer(
                d_model=self.hidden_dim,
                dim_feedforward=self.hidden_dim*4,
                nhead=self.hidden_dim//16,
                dropout=0.00,
                activation='gelu',
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                batch_first=True,
            )
        # Classification layers
        self.linear_class = nn.Sequential(
                nn.Linear(self.hidden_dim, config.N_CLASSES),
            )
        self.linear_bbox = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 4),
            )
        # output positional encodings
        self.query_pos = nn.Parameter(torch.randn(1, num_preds, self.hidden_dim), requires_grad=True)
        # learnable spatial positional encodings
        self.pe_h = nn.Parameter(torch.randn(self.hidden_dim // 2, config.PATCH_DIM), requires_grad=True)
        self.pe_w = nn.Parameter(torch.randn(self.hidden_dim // 2, config.PATCH_DIM), requires_grad=True)

    def forward(self, inputs, debug=False): 
        # Normalize Input
        inputs = inputs.float() / 255.0
        # inputs = (inputs - self.mean) / self.std
        inputs = inputs.swapaxes(3, 1)
        # Acquire Batch Size
        B = inputs.shape[0]
        # Backbone
        x_backbone = self.backbone(inputs)
        # Projection to hidden dim
        x_proj = self.proj(x_backbone)
        # Positional Encoding
        pos = torch.cat([
            self.pe_h.unsqueeze(1).repeat(1, self.patch_dim, 1),
            self.pe_w.unsqueeze(2).repeat(1, 1, self.patch_dim),
        ], dim=0).unsqueeze(0)
        # construct positional encodings
        x_proj = (x_proj + pos).flatten(2).permute(2, 0, 1)

        # propagate through the transformer
        x = self.transformer(
                x_proj.swapaxes(1,0),
                self.query_pos.repeat(B,1,1),
            )
        # objectness prediction
        linear_cls = self.linear_class(x).squeeze(1)
        # coordinate prediction
        bb_xyhw = self.linear_bbox(x).squeeze(1).sigmoid()
        # convert xyhw → xyxy
        x_c, y_c, w, h = torch.chunk(bb_xyhw, chunks=4, dim=1)
        bb_xyxy = torch.concat([x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2], axis=1)
        # result
        if debug:
            return { 'obj': linear_cls, 'bb_xyhw': bb_xyhw, 'bb_xyxy': bb_xyxy, 'x_backbone': x_backbone, 'x_proj': x_proj }
        else:
            return { 'obj': linear_cls, 'bb_xyhw': bb_xyhw, 'bb_xyxy': bb_xyxy }
        

# Configuration class
class Config:
    PATCH_DIM = 16
    BACKBONE_PRETRAINED = True
    N_CLASSES = 1  # Number of object classes (e.g., 1 for binary classification)

def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10):
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started.")
        epoch_loss = 0.0
        for batch_idx, (images, bboxes) in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
            try:
                # Move data to GPU
                images = images.to(device)
                bboxes = [torch.tensor(box, dtype=torch.float32).to(device) for box in bboxes]

                # Forward pass
                outputs = model(images)
                pred_bboxes = outputs['bb_xyhw']  # Predicted bounding boxes

                # Compute loss (example: L1 loss for bounding boxes)
                loss = 0
                for pred, target in zip(pred_bboxes, bboxes):
                    target = target.view(-1, 4)  # Ensure target is in (x, y, w, h) format
                    loss += criterion(pred, target)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                print(f"Batch {batch_idx + 1} processed. Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Error in batch {batch_idx + 1}: {e}")
                continue

        print(f"Epoch {epoch + 1}/{num_epochs} completed. Epoch Loss: {epoch_loss:.4f}")
    print("Training completed.")

# Main script
if __name__ == "__main__":
    print("Loading dataset...")
    input_path = r"\\clin-storage\Group Ruers\Students personal folder\Bart van den Berg (M2)_TUDelft_2025\Data\Collection"
    images, labels = load_data(input_path)
    print("Dataset loaded successfully.")

    print("Creating dataset and dataloaders...")
    dataset = ImageDataset(images, labels)
    trainset, testset = random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])
    train_loader = DataLoader(trainset, shuffle=True, batch_size=64, collate_fn=custom_collate_fn)
    print("Dataset and dataloaders created.")

    print("Initializing model, optimizer, and loss function...")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = testDETR(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()  # Example loss function for bounding boxes
    print("Model, optimizer, and loss function initialized.")

    print("Starting training process...")
    train_model(model, train_loader, optimizer, criterion, device, num_epochs=10)
    print("Training process finished.") 
        

    
    
    
    
