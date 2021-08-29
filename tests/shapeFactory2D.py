import unittest
import numpy as np
import proteinFun.tetris_shapefactory as shapeFactory

class MyTestCase(unittest.TestCase):

    def test1(self):
        lShapeDim2 = np.array([[1., 1., 0., 0.],
                               [1., 1., 0., 0.],
                               [1., 1., 1., 1.],
                               [1., 1., 1., 1.]])
        factory = shapeFactory.shapeFactory2D(test=True)
        lShapeDim2FromFactory = factory.generateLShape()
        np.testing.assert_array_equal(lShapeDim2, lShapeDim2FromFactory)
        print("test 1: OK")

    def test2(self):
        halfSquareDim2 = np.array([[1., 1., 0., 0.],
                                   [1., 1., 0., 0.],
                                   [1., 1., 0., 0.],
                                   [1., 1., 0., 0.]])
        factory = shapeFactory.shapeFactory2D(test=True)
        halfSquareShapeDim2FromFactory = factory.generateHalfSquareShape()
        np.testing.assert_array_equal(halfSquareDim2, halfSquareShapeDim2FromFactory)
        print("test 2: OK")

    def test3(self):
        smallLShapeDim2 = np.array([[1., 1., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [0., 0., 0., 0.]])
        factory = shapeFactory.shapeFactory2D(test=True)
        smallLShapeDim2FromFactory = factory.generateSmallLShape()
        np.testing.assert_array_equal(smallLShapeDim2, smallLShapeDim2FromFactory)
        print("test 3: OK")

    def test4(self):
        zShapeDim2 = np.array([[0., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [1., 1., 0., 0.],
                               [1., 0., 0., 0.]])
        factory = shapeFactory.shapeFactory2D(test=True)
        zShapeDim2FromFactory = factory.generateZShape()
        np.testing.assert_array_equal(zShapeDim2, zShapeDim2FromFactory)
        print("test 4: OK")

if __name__ == '__main__':
    unittest.main()
