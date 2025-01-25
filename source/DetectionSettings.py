STATUS_DICT = {'empty': 0, 'occupied': 1}
IMAGE_FOLDER_PATH = '../data/testData'
ROOM = 'DiningRoom'
IMAGE_SIZE = (856,480)

class InfoContainer:
    def __init__(self):
        # Excluded Objects due to being same kind of chair, unimportant or impossible to appear in general
        self.unimportantObjects = {1, 13, 27, 39, 41, 56, 57, 59, 60, 68, 69, 70, 71, 72, 78, 79}

        # Model path
        self.modelPath = "../models/yolov8-human.pt"

        # Dictionary of ROIs of chairs in rooms
        self.ROIs = {
            'DiningRoom' : [ [268,288,108,90], [247,386,88,150], [512,291,377,102], [565,391,406,173] ],
        }

        # ROIs in case model predicted empty
        self.smallerROIs = {
            'DiningRoom' : [ [223,272,132,194], [227,368,102,279], [496,281,404,206], [537,375,440,300] ],
        }


    def getProperty(self):
        return self.modelPath, self.ROIs, self.smallerROIs, self.unimportantObjects