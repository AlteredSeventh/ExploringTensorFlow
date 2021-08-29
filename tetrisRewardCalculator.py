import numpy as np

class tetrisRewardCalculator():
    def __init__(self):
        {}

    # number of cells on the board that are covered ("filled"), as a measure of success
    def basicCoverageScore(self, tetrisBoard):
        numberOfCells = np.product(np.shape(tetrisBoard))
        numberOfCoveredCells = np.sum(tetrisBoard)
        return numberOfCoveredCells/numberOfCells

    #todo: incentivize lowest row alignment as well
    def wellFitScore(self, oldtetrisboard, newtetrisboard):
        diff = np.subtract(newtetrisboard, oldtetrisboard)
        x,y = np.nonzero(diff)
        if len(x) * len(y) == 0. : return 0.
        height = max(x)-min(x)+1.
        width = max(y) - min(y)+1.
        totalPossibleFits = 0.

        bottomShiftScore = rightShiftScore = leftShiftScore = 0.
        #check maxes on 3 sides
        if max(y)+1 < newtetrisboard.shape[1]:
            rightShiftScore = (oldtetrisboard[(x,y+1)] >= 1.).sum()
            totalPossibleFits += height

        if min(y)> 0:
            leftShiftScore = ((oldtetrisboard[(x,y-1)]) >= 1.).sum()
            totalPossibleFits += height

        if max(x)+1 < newtetrisboard.shape[0]:
            bottomShiftScore = (oldtetrisboard[(x+1,y)] >= 1.).sum()
            totalPossibleFits += width


        if totalPossibleFits == 0.:
            return 0.
        else:
            score = (rightShiftScore + leftShiftScore + bottomShiftScore) / totalPossibleFits
            return score

def test1():
    tetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0., 0.]])
    calculator = tetrisRewardCalculator()
    calculatedCoverage = calculator.basicCoverageScore(tetrisBoard)
    np.testing.assert_equal( calculatedCoverage, 12./(6.*8.))
    print("test1: OK")

def test2():
    tetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0., 0.]])
    newTetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.]])
    calculator = tetrisRewardCalculator()
    calculatedFit = calculator.wellFitScore(tetrisBoard,newTetrisBoard)
    np.testing.assert_equal(calculatedFit, 0.5)
    print("test2: OK")

def test3():
    tetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 1., 1., 0.],
        [0., 1., 1., 0., 0., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.]])
    newTetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.]])
    calculator = tetrisRewardCalculator()
    calculatedFit = calculator.wellFitScore(tetrisBoard,newTetrisBoard)
    np.testing.assert_equal(calculatedFit, 1.)
    print("test3: OK")

def test4():
    tetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 1., 1., 0.],
        [0., 1., 1., 0., 0., 1., 1., 0.],
        [0., 1., 0., 0., 0., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.]])
    newTetrisBoard = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 1., 0., 1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1., 1., 1., 0.]])
    calculator = tetrisRewardCalculator()
    calculatedFit = calculator.wellFitScore(tetrisBoard,newTetrisBoard)
    np.testing.assert_equal(calculatedFit, 5./6.)
    print("test4: OK")


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()