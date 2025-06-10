import cv2
import numpy as np
import pickle
import os

# Global model cache
_model_package = None

def extract_features(img):
    """Extract visual features from the image as trained in the model"""
    # Resize image to match training size (32x32)
    img = cv2.resize(img, (32, 32))
    
    # Convert to grayscale if needed
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

def empty_or_not(spot_crop):
    """
    Determine if a parking spot is empty or not using the trained model
    
    Args:
        spot_crop: Image crop of the parking spot
        
    Returns:
        bool: True if spot is empty, False if occupied
    """
    global _model_package
    
    try:
        # Load model if not already loaded
        if _model_package is None:
            model_path = r'C:\Users\SadiqAliBabar\Downloads\ParkingDetectorComplete\ParkingDetectorComplete\models\parking_detector_model.pkl'
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Using fallback method.")
                # Fallback to a simple method if model not found
                gray = cv2.cvtColor(spot_crop, cv2.COLOR_BGR2GRAY)
                return np.mean(gray) > 100  # Simple threshold
            
            with open(model_path, 'rb') as f:
                _model_package = pickle.load(f)
                print("✅ Model loaded successfully!")
        
        # Extract features
        features = extract_features(spot_crop)
        
        # Scale features
        scaler = _model_package['scaler']
        features_scaled = scaler.transform([features])
        
        # Predict
        model = _model_package['model']  
        prediction = model.predict(features_scaled)
        
        # Class 0 = Empty, Class 1 = Not Empty
        return prediction[0] == 0
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback to a simple method in case of error
        try:
            gray = cv2.cvtColor(spot_crop, cv2.COLOR_BGR2GRAY)
            return np.mean(gray) > 100  # Simple threshold as fallback
        except:
            return False  # Default to occupied if everything fails

def get_parking_spots_bboxes(connected_components):
    """
    Get bounding boxes for all parking spots
    
    Args:
        connected_components: Output from cv2.connectedComponentsWithStats
        
    Returns:
        list of tuples: Each tuple contains (x, y, w, h) of a parking spot
    """
    (totalLabels, label_ids, values, centroid) = connected_components
    
    slots = []
    
    # Skip label 0 which is the background
    for i in range(1, totalLabels):
        # Get the bounding box coordinates and attributes
        x = int(values[i, cv2.CC_STAT_LEFT])
        y = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        area = values[i, cv2.CC_STAT_AREA]
        
        # You can filter spots based on size if needed
        # if area < min_area or area > max_area:
        #     continue
            
        slots.append((x, y, w, h))
        
    return slots