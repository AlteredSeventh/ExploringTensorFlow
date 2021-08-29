import numpy as np
from configs.config import Config, TestConfig
import tetris_actions as tetrisActions
import tetris_shapefactory as shapeFactory
import tetrisRewardCalculator as rewardCalculator
import matplotlib.pyplot as plt

class tetrisActionSimulator():
    def __init__(self, test=False):
        if test is True:
            self.test = True
            np.random.seed(0)
            self.configuration = TestConfig
        else:
            self.test = False
            self.configuration = Config

        self.initialTetrisBoard = np.zeros((
            self.configuration.board_height,
            self.configuration.board_width
        ))

        self.actions = tetrisActions.tetris_actions(test)
        self.factory = shapeFactory.shapeFactory2D(self.test)
        self.runningTetrisBoard = np.zeros((
            self.configuration.board_height,
            self.configuration.board_width
        ))

    def randomShape(self):
        randomPick = np.random.choice(self.factory.supportedShapeMethods())
        result = getattr(self.factory, randomPick)()
        return result

    def randomRotation(self, shape):
        possibleClockwiseTurns = self.configuration.numberOf90DegRotations
        randomPick = np.random.choice(possibleClockwiseTurns)
        return self.actions.rotate(shape,randomPick)

    def randomShift(self, shape):
        possibleShifts = self.actions.possibleShiftActions()
        randomPick = np.random.choice(possibleShifts)
        return self.actions.shift(shape, randomPick)

    def randomDroppedShape(self):
        shape = self.randomShape()
        rotatedShape = self.randomRotation(shape)
        shiftedandRotatedShape = self.randomShift(rotatedShape)
        return shiftedandRotatedShape

    def dropSimulatedShape(self):
        randomShape = self.randomDroppedShape()
        actions = tetrisActions.tetris_actions(self.test)
        newTetrisBoard = actions.shapeDrop(randomShape, self.runningTetrisBoard )
        self.runningTetrisBoard = newTetrisBoard
        return newTetrisBoard

    def runSimulation(self, numberOfDrops = 100, pauseBetweenImg = 0.5):
        for i in range(1,numberOfDrops):
            self.dropSimulatedShape()
            plt.imshow(self.runningTetrisBoard)
            plt.pause(pauseBetweenImg)
            plt.draw()
            print(i)

        plt.show()
        plt.clf()


    def runSimulation2(self, numberOfDrops = 100, pauseBetweenImg = 0.5):
        actions = tetrisActions.tetris_actions(test = True)
        for i in range(1,numberOfDrops):
            action = np.random.choice(192)
            self.runningTetrisBoard = actions.doAction(action, self.runningTetrisBoard)
            plt.imshow(self.runningTetrisBoard)
            plt.pause(pauseBetweenImg)
            plt.draw()
            print(i)

        plt.show()
        plt.clf()



def test1():
    simu = tetrisActionSimulator(test=True)
    simu.runSimulation(numberOfDrops = 60)
    calc = rewardCalculator.tetrisRewardCalculator()
    print( "And the score is: ", calc.basicCoverageScore(simu.runningTetrisBoard))

def test2():
    simu = tetrisActionSimulator(test=True)
    simu.runSimulation2(numberOfDrops = 60)
    calc = rewardCalculator.tetrisRewardCalculator()
    print( "And the score is: ", calc.basicCoverageScore(simu.runningTetrisBoard))


if __name__ == '__main__':
    test1()
    test2()
