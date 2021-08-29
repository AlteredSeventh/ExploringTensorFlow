import unittest
import numpy as np
import proteinFun.tetris_actions as tetrisActions
import proteinFun.tetris_shapefactory as factory
class MyTestCase(unittest.TestCase):
    def test1(self):
        actions = tetrisActions.tetris_actions(test = True)
        lShapeDim2 = np.array([[1.,1.,0.,0.],[1.,1.,0.,0.],[1.,1.,1.,1.],[1.,1.,1.,1.]])
        rotatedShape = actions.rotate(lShapeDim2)
        rotatedShapeTest = np.array([[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,0.,0.],[1.,1.,0.,0.]])
        np.testing.assert_array_equal(rotatedShapeTest, rotatedShape)

        lShapeDim2 = np.array([[1.,1.,0.,0.],[0.,1.,0.,0.],[0., 0., 0., 0],[0., 0., 0., 0]])
        rotatedShape2 = actions.rotate(lShapeDim2)
        rotatedShapeTest2 = np.array([[0.,1.,0.,0.],[1.,1.,0.,0.]])
        np.testing.assert_array_equal(rotatedShapeTest2, rotatedShape2)
        print("test 1: OK")

    def test2(self):
        actions = tetrisActions.tetris_actions(test = True)
        lShapeDim2 = np.array([[1., 1., 0., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
        rotatedShape = actions.rotate(lShapeDim2, 2)
        rotatedShapeTest = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.], [0., 0., 1., 1.], [0., 0., 1., 1.]])
        np.testing.assert_array_equal(rotatedShapeTest, rotatedShape)
        print("test 2: OK")

    def test3(self):
        actions = tetrisActions.tetris_actions(test = True)
        lShapeDim2 = np.array([[1., 1., 0., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
        shiftedShape = actions.shift(lShapeDim2, 1 )
        shiftedShapeTest = np.array([[0.,1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0.,1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0.,1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                     [0.,1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])
        np.testing.assert_array_equal(shiftedShape,shiftedShapeTest)
        print("test 3: OK")

    def test4(self):
        actions = tetrisActions.tetris_actions(test = True)
        currentTetrisBoard = np.array([
                                     [0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 0., 0., 0.]])

        droppedShape = np.array([
                                     [ 0., 0., 0., 0., 0., 1., 1., 1.],
                                     [ 0., 0., 0., 0., 0., 1., 1., 1.]])

        newTetrisBoard = actions.shapeDrop(droppedShape,currentTetrisBoard)
        newTetrisBoardTest = np.array([
                                     [0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 1., 1., 1.],
                                     [0., 1., 1., 1., 1., 1., 1., 1.]])
        np.testing.assert_array_equal(newTetrisBoard,newTetrisBoardTest)
        print("test 4: OK")

    def test5(self):
        actions = tetrisActions.tetris_actions(test = True)
        currentTetrisBoard = np.array([
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])

        droppedShape = np.array([
                                     [ 1., 1., 1., 1., 1., 1., 1., 1. ,0., 0., 0., 0.]])
        newTetrisBoard = actions.shapeDrop(droppedShape,currentTetrisBoard[:])
        newTetrisBoardTest = np.array([
                                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])
        np.testing.assert_array_equal(newTetrisBoard,newTetrisBoardTest)
        shapeFactory = factory.shapeFactory2D(test = True)
        droppedLShape = shapeFactory.generateLShape()
        newTetrisBoardTest2 = np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 0.]])

        newTetrisBoard2 = actions.doAction((0,0,8), currentTetrisBoard)
        np.testing.assert_array_equal(newTetrisBoard2, newTetrisBoardTest2)
        print("test 5: OK")

    def test6(self):
        actions = tetrisActions.tetris_actions(test = True)
        lShapeDim2 = np.array([[1., 1., 0., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]])
        shiftedShape = actions.shift(lShapeDim2, 1 )

        shiftedShapeTest = np.array([[0.,1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0.,1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0.,1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                                     [0.,1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]])
        np.testing.assert_array_equal(shiftedShape,shiftedShapeTest)
        print("test 6: OK")

    def test7(self):
        actions = tetrisActions.tetris_actions(test=True)
        np.testing.assert_array_equal(actions.getActionTuple(10), (0,0,10))
        np.testing.assert_array_equal(actions.getActionTuple(20), (0, 1, 8))
        np.testing.assert_array_equal(actions.getActionTuple(24), (0, 2, 0))
        np.testing.assert_equal( 12*4*4, actions.possibleActionsDiscreet())
        print("test 7: OK")

if __name__ == '__main__':
    unittest.main()
