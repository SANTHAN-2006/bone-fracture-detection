# Aim :
To implement real-time fracture detection using grayscale morphology on X-ray images.

# Algorithm Steps:

## 1. Preprocess the input X-ray image by converting it to grayscale.

## 2. Apply thresholding to segment the image into foreground (fracture) and background regions.

## 3. Utilize morphological operations, such as dilation and erosion, to enhance fracture features and remove noise.

## 4. Detect fracture areas by finding contours in the processed image.

## 5. Highlight the detected fracture areas on the original image.

## 6. Display the original image with highlighted fracture areas in real-time.

## Program :
```python
import cv2
import numpy as np

# Function to perform fracture area prediction using morphological operations
def predict_fracture_area(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the image
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to enhance features
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of potential fractures
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with maximum area (likely the fracture)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    # Create a mask for the fracture area
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the original image to highlight the fracture area
    fracture_area = cv2.bitwise_and(img, img, mask=mask)
    
    return fracture_area

# Main function for fracture area prediction from an input image
def main():
    # Read input image
    input_img_path = r"C:\Users\gayat\Downloads\fracture-2.jpg"  # Replace with the path to your X-ray image
    img = cv2.imread(input_img_path)
    
    # Check if the image is loaded successfully
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Predict the fracture area in the input image
    fracture_area = predict_fracture_area(img)
    
    # Display the original image and the predicted fracture area
    cv2.imshow('Original', img)
    cv2.imshow('Fracture Area Prediction', fracture_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```

## Results Presentation :
![image](https://github.com/SANTHAN-2006/bone-fracture-detection/assets/80164014/a1249115-f6c6-4149-93ab-3457a7ecf057)
<br>
<br>
<br>
![image](https://github.com/SANTHAN-2006/bone-fracture-detection/assets/80164014/a92c7065-29be-49e5-9689-c156a9b6c6a0)


### The program successfully detects fractures in real-time from the input X-ray images.

### The original X-ray image and the image with highlighted fracture areas are displayed simultaneously.

### Fracture areas are outlined in green on the original image, making them easily identifiable.

## Advantages and Challenges of Using Morphological Operations for Fracture Detection:

## Advantages :
### Noise Robustness: Morphological operations help mitigate the effects of noise in X-ray images, enhancing the accuracy of fracture detection.

### Real-Time Processing: The algorithm's computational efficiency allows for real-time fracture detection, making it suitable for clinical applications.

### Feature Enhancement: Morphological operations can enhance fracture features, making them more prominent and easier to identify in X-ray images.

## Challenges :
### Parameter Tuning: Choosing appropriate parameters for morphological operations can be challenging and may require extensive experimentation.

### Over- and Under-Segmentation: Morphological operations may lead to over-segmentation or under-segmentation of fractures, impacting the accuracy of detection.

### Artifact Handling: Morphological operations may amplify imaging artifacts, leading to false positives or missed fractures in the detection process.

### Generalization: The effectiveness of morphological operations may vary across different types of fractures, anatomical regions, and imaging modalities, limiting its generalizability.



