import io
import cv2
import json
import base64
import requests
import urllib.request
import numpy as np

# Construct API parameters and URL. The API endpoint URL is https://image-analyser1.p.rapidapi.com/objectdetect
# Replace SIGN-UP-FOR-KEY below with your RapidAPI API-key which is obtained after subscription  
url = "https://image-analyser1.p.rapidapi.com/ocr"
headers = {
        "content-type": "application/json",
        "X-RapidAPI-Host": "image-analyser1.p.rapidapi.com", 
        "X-RapidAPI-Key": "SIGN-UP-FOR-KEY"
}

detection_boxes = []
detection_scores = []

def get_optimal_font_size(char_height):
    """Get character height in image and returns best font size to use.
    Args:
      char_height: Character height in image
    Returns:
      optimal font size
    """
    
    font_size = 0.5
    while(True):
        (text_width, text_height), baseline = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        if text_height > char_height:
            return font_size - 0.1 
        font_size = font_size + 0.1


req = urllib.request.urlopen("https://rapidapi-prod-collections.s3.amazonaws.com/org/003d6226-93ae-4afe-ad24-f50ea97455a3.png")
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1) # 'Load it as it is'

 
# Get image height and width
height = image.shape[0]
width = image.shape[1]
       
# Convert to JPEG and save in variable
image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

# Encode an image file into Base64 format
encoded_string = base64.b64encode(image_bytes).decode('utf-8')

# Create HTTP POST request body
payload = {"b64image": encoded_string, "annotate_words": "true", "normalisedvertices":"true", "min_conf": 10}
    
# Upload image to TenaxisAI API endpoint for analysis
http_response = requests.request("POST", url, json=payload, headers=headers)
response = http_response.json()

# Parse JSON response from TenaxisAI API
ocrAnnotations = response['response'][0]


Annotations = ocrAnnotations['ocrAnnotations']

# Window name in which image is displayed
window_name = "OCR"

# Draw annotations and detected text on image
index = 0
for anno_index in Annotations:
    vertices = Annotations[index]['vertices']
        
    # Starting coordinate
    # represents the top left corner of rectangle


    xmin = int(float(vertices['xmin'])*width)
    ymin = int(float(vertices['ymin'])*height)        
    start_point = (xmin, ymin)

  
    # Ending coordinate
    # represents the bottom right corner of rectangle
    xmax = int(float(vertices['xmax'])*width)
    ymax = int(float(vertices['ymax'])*height)    
    end_point = (xmax, ymax)
  
    # Blue color in BGR
    color = (255, 0, 0)
  
    # Line thickness of 1 px
    thickness = 1

    # Get word 
    word = Annotations[index]['word']

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 1 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    best_font_size = get_optimal_font_size(ymax-ymin)

    cv2.putText(image, word, (xmin, ymin+ymax-ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    index = index + 1

# Displaying the image 
cv2.imshow(window_name, image)
cv2.waitKey(0)
#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break

# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
