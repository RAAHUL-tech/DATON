# DATON
Detection and Tracking of Object Number Plates in real-time.

This Project is done as part of the AI club, at Puducherry Technological University. The main objective of this project is to build a deep-learning model to recognize and track the number plates that are entering and leaving the PTU premises. The information regarding the vehicle like the time entered, and license number is stored in a CSV file.

# Project Structure Overview
  For this prototype to work we give a video as input to our model. In real-time this video stream comes from the CCTV camera placed at the entrance of the campus. YOLO V8 nano model is used to detect cars, trucks, bikes, and other forms of vehicles from the video frame. Next, these vehicles are given unique IDs using the open-source SORT library which is implemented in the sort folder. Once each vehicle has been assigned an ID our next step is to detect the license plate from the vehicle. For this, we use a pretrained model called wpod -net which can detect license plates of 8 different countries' number plates including India. Once the number plate is detected and cropped, the text present in it is read using the python's easy-ocr library with some confidence score. Finally, these results are stored back in a CSV file.

  # Wpod-net Pre-trained Model
     WPOD-NET, or Warped Planar Object Detection Network, is a pre-trained model that can detect and extract license plates from vehicle images. It was developed using insights from YOLO, SSD, and Spatial Transformer Networks (STN). 
      To know more about this model read this paper: https://arxiv.org/abs/2212.07066
