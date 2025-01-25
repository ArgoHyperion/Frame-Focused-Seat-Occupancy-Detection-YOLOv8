import matplotlib.pyplot as plt
import cv2
import pandas as pd

class Visualizer:
    def __init__(self):
        pass

    # Visualizing image with boxes of chairs
    def displayRoiBoxesOfImage(self, img, frameIndex, rois, dataframe):
        for _, row in dataframe.iterrows():
            for idx, roi in enumerate(rois):
                if str(idx+1) == str(row['Chair Number']):
                    topLeft = (roi[2], roi[3])
                    bottomRight = (roi[0], roi[1])
                    color = (0, 255, 0) if row['Status'] == 0 else (0, 0, 255)
                    label = "empty" if row['Status'] == 0 else "occupied"

                    # Add rectangle
                    cv2.rectangle(img, topLeft, bottomRight, color, 2)

                    # Add the label
                    text_position = (roi[2] + 2, roi[3] - 5)
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    fontScale = 0.6
                    thickness = 1
                    textColor = color
                    cv2.putText(img, label, text_position, font, fontScale, textColor, thickness)

        # Display the image
        dpi = plt.rcParams['figure.dpi']
        height, width, _ = img.shape
        plt.figure(figsize=(width / dpi, height / dpi))  # Set figure size to match the image size
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Remove axes
        plt.tight_layout(pad=0)  # Remove any padding
        plt.show()

