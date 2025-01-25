import cv2
import pandas as pd
import os
from ObjectsDetection import Detector
from DetectionSettings import *
from Visualizing import Visualizer

class Processor:
    # Initialize property
    def __init__(self, info):
        self._model, self._ROIs, self._smallerROIs, self._unimportantObjects = info.getProperty()
        self._detector = Detector(self._model)
        self._visualizer = Visualizer()
        self._room = None

    # Visualize Image
    def _visualizeImage(self, image, frameIndex, roi, frameDF):
        return self._visualizer.displayRoiBoxesOfImage(image, frameIndex, roi, frameDF)

    # Process function
    def process(self, room, imageFolder):
        self._room = room
        for filename in os.listdir(imageFolder):
            frameDF, frameIndex = self._processImage(imageFolder, filename)
            # Ensure Frame Number as String, Chair Number as Int
            frameDF[['Chair Number', 'Status']] = frameDF[['Chair Number', 'Status']].astype(int)
            frameDF['Frame Number'] = frameDF['Frame Number'].astype(str)
            frameDF = frameDF.sort_values(by=['Frame Number', 'Chair Number'], ascending=True)
            # Save Frame Dataframe into a csv file
            frameDF.to_csv(f'../csvFolder/status_{frameIndex}.csv', index=False)
        return 0

    # Check in the smaller ROI (only called when no object is detected or only chair is detected in the big ROI)
    def _processSmallerRoi(self, idx, img):
        zone = self._smallerROIs[self._room][idx]
        df = self._detector.detectObjects(
            img[zone[3]:zone[1], zone[2]:zone[0], :],
            confThreshold=0.3,
            nmsThreshold=0.5)
        status = 'empty'
        for item in df['ClassIds'].unique():
            if item not in self._unimportantObjects:
                status = 'occupied'
                break

        return status


    # Detect objects in the image
    def _processImage(self,folder, filename):
        frameDF = pd.DataFrame(columns=['Frame Number',
                                        'Chair Number',
                                        'Status'])
        frameDF = frameDF.astype({'Frame Number': 'str',
                                  'Chair Number': 'int',
                                  'Status': 'int'})
        frameIndex = filename.split('_')[1]
        frameIndex = frameIndex.split('.')[0]
        img = cv2.imread(os.path.join(folder, filename))

        # Resize image - ROIs are defined on this size
        img = cv2.resize(img, IMAGE_SIZE)

        # Check if image is empty
        if img is not None:
            # Select correct ROIs of chairs
            roi = self._ROIs[f'{self._room}']
            print(f'Frame Number: {frameIndex}')
            for idx, chair in enumerate(roi):
                print(f"ROI number : {idx + 1} is being processed.", end='\t')

                # Initialize status to default values for each chair
                status = 'empty'

                # Proceed detecting on selected ROI
                df = self._detector.detectObjects(img[chair[3]:chair[1],
                                                  chair[2]:chair[0], :],
                                                  confThreshold=0.35,
                                                  nmsThreshold=0.5)

                # Check if result from detecting is empty
                if df.empty:
                    status = self._processSmallerRoi(idx, img)
                else:
                    if 0 in df['ClassIds'].values:
                        status = 'occupied'
                    else:
                        uniqueVals = df['ClassIds'].unique()
                        for item in uniqueVals:
                            if item not in self._unimportantObjects:# Ignore chairs, and some others ( all these items have been mapped to chairs, manually by us )
                                status = 'occupied'
                                break

                # Add new row to Frame csv
                print(f"Status : {status}")
                newRow = pd.DataFrame([{'Frame Number': frameIndex,
                           'Chair Number': idx + 1,
                           'Status': STATUS_DICT[status]
                          }])
                frameDF = pd.concat([frameDF, newRow], ignore_index=True)  # Use pd.concat to append

        print(f"Completed {frameIndex}\n")
        self._visualizeImage(img, frameIndex, roi, frameDF) # For visualizing, not required
        return frameDF, frameIndex

def main():
    info = InfoContainer()
    processor = Processor(info)
    processor.process(ROOM, IMAGE_FOLDER_PATH)

if __name__ == '__main__':
    main()
