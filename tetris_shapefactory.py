import numpy as np
from configs.config import Config, TestConfig

# a factory that makes several "basic" shapes for a Tetris games
# aka the the building blocks of the Tetris game

class shapeFactory2D():
    def __init__(self, test=False, scope = 'all'):
        if test is True:
            configuration = TestConfig
        else :
            configuration = Config
        #we'll use square matrices to 'draw' shapes on; these set the max dimensions
        self.shape_max_width = configuration.shape_max_width
        #to simplify, as we'll need these a lot
        self.allZerosSquare = np.zeros((self.shape_max_width, self.shape_max_width))
        self.allOnesSquare = np.ones((self.shape_max_width, self.shape_max_width))
        self.scope = scope

    def Filter(self, string, substr):
        return [k for k in string if substr in k]

    def supportedShapeMethods(self ):
        if self.scope == "all":
            allMethods = dir(self)
            substr = 'generate'
            return self.Filter(allMethods, substr)
        elif self.scope == "halfSquare":
            return ["generateHalfSquareShape", "generateLShape"]
        else: raise NotImplementedError

    def generateLShape(self):
        dimension = self.shape_max_width
        shapeMatrix = self.allOnesSquare
        allZeros = self.allZerosSquare
        halfWay = dimension // 2
        shapeMatrix[:halfWay, halfWay:] = allZeros[:halfWay, halfWay:]
        return shapeMatrix

    def generateSmallLShape(self):
        dimension = self.shape_max_width
        shapeMatrix = self.allZerosSquare
        allOnes = self.allOnesSquare
        shapeMatrix[0:dimension-1, 0] = allOnes[0:dimension-1, 0]
        shapeMatrix[0, 1] = allOnes[0, 1]
        return shapeMatrix

    def generateZShape(self):
        dimension = self.shape_max_width
        shapeMatrix = self.allZerosSquare
        allOnes = self.allOnesSquare
        halfWay = dimension // 2
        shapeMatrix[halfWay:, 0] = allOnes[halfWay:, 0]
        shapeMatrix[halfWay-1:dimension-1, 1] = allOnes[halfWay-1:dimension-1, 1]
        return shapeMatrix

    def generateHalfSquareShape(self):
        dimension = self.shape_max_width
        shapeMatrix = self.allOnesSquare
        allZeros = self.allZerosSquare
        halfWay = dimension // 2
        shapeMatrix[:, halfWay:] = allZeros[:, halfWay:]
        return shapeMatrix
