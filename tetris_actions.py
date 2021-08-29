import numpy as np
from configs.config import Config, TestConfig
import tetris_shapefactory as shapefactory


# droppedShape is the Shape (after you've made your choices to rotate / shift)
# we assume the following order:
# 1. start with shape top left on top of Tetris board
# 2. perform any rotations you like
# 3. shift the shape
# 4. Initialize the shape for dropping (adding 0 columns until it has same dimension)
# 5. "Go" on the dropping

class tetris_actions():
    def __init__(self, test=False, scope = 'all'):
        if test is True:
            self.configuration = TestConfig
        else:
            self.configuration  = Config
        self.scope = scope
        self.factory = shapefactory.shapeFactory2D(test, scope)


    def doAction(self, action, state):
        if isinstance(action, tuple) == False:
            tuple_action = self.getActionTuple(action)

        shapeAction = self.factory.supportedShapeMethods()[tuple_action[0]]
        rotateAction = tuple_action[1]
        shiftAction = tuple_action[2]
        initialShape = getattr(self.factory, shapeAction)()
        rotatedShape = self.rotate(initialShape,rotateAction)
        rotatedandshiftedShape = self.shift(rotatedShape,shiftAction)
        return self.shapeDrop(rotatedandshiftedShape, state)

    def randomShift(self, shape):
        possibleShifts = self.actions.possibleShiftActions()
        randomPick = np.random.choice(possibleShifts)
        return self.actions.shift(shape, randomPick)

    def randomDroppedShape(self):
        shape = self.randomShape()
        rotatedShape = self.randomRotation(shape)
        shiftedandRotatedShape = self.randomShift(rotatedShape)
        return shiftedandRotatedShape

    # map the (flattened, 1D) action index in A to the indices of the 3 action spaces (A=A1xA2xA3)
    def getActionTuple(self, action):
        ranges = self.possibleActionsMatrixShape()
        actionIndex = np.unravel_index(action,ranges)
        shapeSelectionAction = self.possibleShapeSelectionActions()[actionIndex[0]]
        rotationAction =  self.possibleRotationActions()[actionIndex[1]]
        shiftAction =  self.possibleShiftActions()[actionIndex[2]]
        return (shapeSelectionAction,rotationAction,shiftAction)

    # total number of actions
    def possibleActionsDiscreet(self):
        possibleActions = self.possibleActionsMatrixShape()
        return possibleActions[0] * possibleActions[1] * possibleActions[2]

    # Shape of matrix of all possible actions A = A1 x A2 x A3, assuming starting position top left,
    # and including 0 (no action). The index of this matrix maps directly to actions.
    def possibleActionsMatrixShape(self ):
        shapeSelectionActions = self.possibleShapeSelectionActions()
        rotationActions =  self.possibleRotationActions()
        shiftActions =  self.possibleShiftActions()
        return (shapeSelectionActions[-1]+1,rotationActions[-1]+1,shiftActions[-1]+1)

    # all possible shapes that can be selected
    def possibleShapeSelectionActions(self):
        return range(len(self.factory.supportedShapeMethods()))

    # all possible shifts to the right (width-1), assuming starting position top left,
    # and including 0 (no shift). We assume at min a first column contains 'shape'.
    def possibleShiftActions(self):
        return range(self.configuration.board_width-1+1)

    # all possible rotations, assuming starting position top left,
    # and including 0 (no rotation)
    def possibleRotationActions(self):
        if self.scope == 'halfSquare':
            return [0]
        elif self.scope == 'all':
            return range(self.configuration.numberOf90DegRotations+1)
        else: raise NotImplementedError


    #if you rotate your Tetris shape clockwise
    def rotate( self, shape, numberofRotations=1):
        if numberofRotations == 0:
            return shape

        rotatedShape = np.rot90( shape, -numberofRotations)
        # push real shape to the left (so you can cover the left side of board)
        numberOfRows = np.shape(rotatedShape)[0]
        numberOfColumns = np.shape(rotatedShape)[1]
        zeroColumn = np.zeros((numberOfRows, 1))
        for i in range(numberOfColumns):
            if sum(abs(rotatedShape[:,0] )) == 0.:
                rotatedShape = np.append(rotatedShape,zeroColumn, axis=1)
                rotatedShape = np.delete(rotatedShape,0,1)
            else:
                break

        for i in range(numberOfRows):
            if sum(abs(rotatedShape[numberOfRows-i-1,:])) == 0.:
                rotatedShape = np.delete(rotatedShape,numberOfRows-i-1,0)
            else:
                break
        return rotatedShape

    def shift( self, shape, stepsToRight = 0):
        numberOfRows = np.shape(shape)[0]
        numberOfColumns = self.configuration.shape_max_width
        zeroColumn = np.zeros((numberOfRows, 1 ))
        numberOfColumnsOnBoard = self.configuration.board_width

        #add stepsToRight columns of zeros to right (if shifting left) and vice versa
        for i in range(abs(stepsToRight )):
            if np.shape(shape)[1] >= numberOfColumnsOnBoard:
                    if sum(shape[:,-1]) !=0.:
                        break
                    else:
                        newshape = np.append(zeroColumn, shape, axis=1)
                        shape = np.delete(newshape,-1,1)
            else:
                shape = np.append(zeroColumn,shape, axis=1)

        # now fill up the matrix with columns so that you reach the shape
        # (Config.numberOfMatrixAxis2 * Config.numberOfMatrixAxis2)
        numberOfColumnsToFitBoard = numberOfColumnsOnBoard - abs(stepsToRight) - self.configuration.shape_max_width
        for i in range(numberOfColumnsToFitBoard):
            if stepsToRight >= 0:
                shape = np.append(shape, zeroColumn, axis=1)
            else:
                shape = np.append(zeroColumn, shape, axis=1)

        return shape

    # returns new state of TetrisBoard after you drop droppedShape on currentTetrisBoard

    def shapeDrop( self, droppedShape, currentTetrisBoard ):
        newTetrisBoard = np.zeros( np.shape(currentTetrisBoard))
        newTetrisBoard = np.add(newTetrisBoard,currentTetrisBoard)
        height = np.shape(currentTetrisBoard)[0]
        rangeOfRowsCurrent = range(height)
        numberOfRowsShape = np.shape(droppedShape)[0]
        summedSlice = oldSummedSlice = 0
        for i in rangeOfRowsCurrent:
            # here we add the dropped shape to the currentTetrisBoard by adding the rows
            # from the top down. Where "1" elements are not added, we know the shape can drop
            # further. Where there is a "2" after sum, we stop and leave the dropped shape in
            # previous place
            startRowInBoard = max(i - numberOfRowsShape+1,0)
            endRowInBoard = min( i + 1, height)
            startRowInShape = max(numberOfRowsShape - 1 - i, 0)
            endRowInShape   = min(numberOfRowsShape, numberOfRowsShape-(i-height)-1 )
            tetrisBoardSlice = newTetrisBoard[startRowInBoard:endRowInBoard, :]
            droppedShapeSlice = droppedShape[startRowInShape:endRowInShape, :]
            oldSummedSlice = tetrisBoardSlice + droppedShapeSlice
            if np.size(oldSummedSlice) is 0: break

            if np.amax(oldSummedSlice) > 1.:
                newTetrisBoard[max(i - numberOfRowsShape,0):i, :] = summedSlice
                break
            elif i == rangeOfRowsCurrent[-1]:
                newTetrisBoard[startRowInBoard:i + 1, :] = oldSummedSlice
                break
            else:
                summedSlice = oldSummedSlice

        return newTetrisBoard

