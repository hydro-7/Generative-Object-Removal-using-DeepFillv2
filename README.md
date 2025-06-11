# Generative-Object-Removal-using-DeepFillv2

## **Introduction**
This is a basic Image inpainting app. When you upload an image, you get a prompt to draw a bounding box around the object you wish to remove. It then takes inspiration from the surrounding regions and removes the image very accurately. This project is powered by MaskRCNN for image segmentation and DeepFillv2 for the inpainting

## **Models**

Mask R-CNN is an extension of Faster RCNN that adds a segmentation branch to generate high quality pixel masks for detected objects. In this project, it’s used to automatically identify and isolate objects in the image that the user wants to remove.
- Output: Bounding boxes, class labels, and binary masks
- Pretrained on COCO dataset (80 common object categories)
- Used here for: Detecting and segmenting objects to remove

DeepFill v2 is an advanced deep generative model for image inpainting it fills in missing or masked regions of an image with realistic textures and structures.
- Uses gated convolutions and contextual attention
- Learns to hallucinate missing content based on surrounding pixels
- Used here for: Filling in the area where the object was removed

NOTE : As DeepFill isn’t available as a pip installable library, the code is copied from the original implementation with slight changes. Further, to run the model you would need its [Weights](https://drive.usercontent.google.com/download?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download&authuser=0). These need to be placed inplace of the placeholder for the project to work.

## **Working**

1) Upload an image (Any format)
2) In the GUI use your mouse to draw a bounding box around the object to be removed.
3) Click enter
4) The result should be visible on your screen, click on download to save the image.

@ working video @

## **Requirements**
Within your environment install the required libraries by the command  ```pip install -r rquirements.txt```

## **File Structure**

```
project_root/
├── .venv/                        # Virtual environment
├── requirements.txt              # Dependencies
├── src/
│   ├── app.py                    # Flask main app
│   ├── objRemove.py              # Object removal logic
│   ├── models/
│   │   ├── deepFill.py           # DeepFill v2 model
│   │   └── weights.pth           # Pretrained DeepFill weights
│   ├── static/
│   │   ├── favicon.png           # Site favicon
│   │   ├── style.css             # Styling
│   │   └── outputs/              # Final images after inpainting
│   └── templates/
│       ├── index.html            # Upload page
│       └── result.html           # Result page
└── static/                       # Temporary uploaded images
```

## **Installation**

```
git clone https://github.com/hydro-7/Generative-Object-Removal-using-DeepFillv2.git
cd src

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Download DeepFill model weights and place them in models/
# e.g., models/weights.pth

# Run the app
python app.py
```

## **Notes**

- For best results, select the tightest bounding box around the object.
- The app automatically converts uploaded images to JPEG format for consistency.
- All processing is done locally — no external API calls.


## **To Do List**
- Add references and sources to the readme
- Improve the GUI for bounding boxes with tkinter
- Incoporate better UI features for the website
- Deploy on HuggingFace Spaces / Docker
