import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from skimage.transform import rotate
from skimage.exposure import equalize_hist, adjust_gamma
from skimage.util import random_noise
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Directories
empty_dir = 'cropped_spots_more/empty'
not_empty_dir = 'cropped_spots_more/not_empty'

# Increase the image size to retain more features
IMG_SIZE = (32, 32)

X = []
y = []

def extract_features(img):
    """Extract visual features from the image"""
    # Convert to grayscale (optional)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # Calculate basic statistics
    mean = np.mean(gray)
    std = np.std(gray)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist = hist.flatten() / np.sum(hist)  # Normalize
    
    # Return the raw image and additional engineered features
    return np.concatenate([img.flatten(), [mean, std, edge_ratio], hist])

def augment_image(img, count):
    """Apply diverse augmentations to create varied samples."""
    aug = []
    h, w = img.shape[:2]
    
    while len(aug) < count:
        # Basic flip operations
        flipped_h = cv2.flip(img, 1)
        flipped_v = cv2.flip(img, 0)
        aug.extend([flipped_h, flipped_v])
        
        # Rotations with different angles
        for angle in [-20, -10, 10, 20]:
            rotated = rotate(img, angle=angle, mode='wrap', preserve_range=True).astype(np.uint8)
            aug.append(rotated)
            
        # Brightness variations
        bright = adjust_gamma(img, gamma=0.7)
        dark = adjust_gamma(img, gamma=1.3)
        aug.extend([bright.astype(np.uint8), dark.astype(np.uint8)])
        
        # Contrast enhancement
        equalized = equalize_hist(img)
        aug.append((equalized * 255).astype(np.uint8))
        
        # Add noise
        noisy = random_noise(img, mode='gaussian', var=0.01)
        aug.append((noisy * 255).astype(np.uint8))
        
        # Zoom variations
        crop_size = int(h * 0.8)
        start_i = (h - crop_size) // 2
        start_j = (w - crop_size) // 2
        cropped = img[start_i:start_i+crop_size, start_j:start_j+crop_size]
        zoomed = cv2.resize(cropped, (w, h))
        aug.append(zoomed)
        
    return aug[:count]  # Return only the requested number of augmentations

def visualize_samples(samples, title):
    """Visualize a few samples for debugging"""
    fig, axes = plt.subplots(1, min(len(samples), 5), figsize=(15, 3))
    for i, ax in enumerate(axes):
        if i < len(samples):
            ax.imshow(cv2.cvtColor(samples[i], cv2.COLOR_BGR2RGB))
            ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def load_and_process(dir_path, label, target_count):
    samples = []
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        samples.append(img)
    
    print(f"Loaded {len(samples)} original images from class {label}")
    
    # Visualize a few original samples (optional)
    if samples:
        visualize_samples(samples, f"Original Class {label}")
    
    # Original images
    for img in samples:
        X.append(extract_features(img))
        y.append(label)
    
    # Augmented images to reach target_count
    aug_needed = max(0, target_count - len(samples))
    if aug_needed > 0:
        print(f"Generating {aug_needed} augmented samples for class {label}")
        per_img = aug_needed // len(samples) + 1
        
        all_augs = []
        for img in samples:
            augmented_imgs = augment_image(img, per_img)
            all_augs.extend(augmented_imgs)
            if len(all_augs) >= aug_needed:
                break
        
        # Visualize a few augmented samples (optional)
        if all_augs:
            visualize_samples(all_augs[:5], f"Augmented Class {label}")
        
        for aug_img in all_augs[:aug_needed]:
            aug_img = cv2.resize(aug_img, IMG_SIZE)
            X.append(extract_features(aug_img))
            y.append(label)
    
    print(f"Total samples for class {label}: {len([i for i in y if i == label])}")

# Target: balance both classes to max count + extra buffer for robustness
num_empty = len(os.listdir(empty_dir))
num_not_empty = len(os.listdir(not_empty_dir))
target = max(num_empty, num_not_empty) * 5  # Create more augmentations for better generalization

print(f"Original dataset: Empty={num_empty}, Not Empty={num_not_empty}")
print(f"Target samples per class after augmentation: {target}")

# Load and process data
load_and_process(empty_dir, 0, target)
load_and_process(not_empty_dir, 1, target)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples after processing: {len(X)}")
print(f"Class distribution: 0 (Empty)={np.sum(y==0)}, 1 (Not Empty)={np.sum(y==1)}")

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Try multiple models
models = {
    'SVM': {
        'model': SVC(probability=True),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    }
}

best_models = {}
best_score = 0
best_model_name = None

for model_name, model_info in models.items():
    print(f"\n🔍 Training {model_name}...")
    grid = GridSearchCV(
        model_info['model'], 
        model_info['params'], 
        cv=cv, 
        n_jobs=-1, 
        verbose=1,
        scoring='f1_weighted'
    )
    grid.fit(X_train, y_train)
    best_models[model_name] = grid.best_estimator_
    
    # Evaluate on test set
    y_pred = grid.predict(X_test)
    score = np.mean(y_pred == y_test)
    print(f"\n📊 {model_name} - Best Parameters: {grid.best_params_}")
    print(f"📊 {model_name} - Accuracy: {score:.4f}")
    print(f"📊 {model_name} - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Track best model
    if score > best_score:
        best_score = score
        best_model_name = model_name

# Save the best model and scaler
print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_score:.4f})")
best_model = best_models[best_model_name]

# Save both models for comparison
for name, model in best_models.items():
    with open(f'model_{name.lower()}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved {name} model as model_{name.lower()}.pkl")

# Save the scaler for preprocessing new data
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved feature scaler as feature_scaler.pkl")

# Save the combined model package for easy deployment
model_package = {
    'scaler': scaler,
    'model': best_model,
    'image_size': IMG_SIZE,
    'feature_extractor': extract_features
}

with open('parking_detector_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✅ Saved complete model package as parking_detector_model.pkl")