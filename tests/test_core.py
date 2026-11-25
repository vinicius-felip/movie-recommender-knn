import unittest
import numpy as np
import pandas as pd
from src.metrics import proximity
from src.knn import MovieKNN
class CoreTests(unittest.TestCase):
    def test_metrics(self):
        a=[4,4,2,2]; b=[2,2,1,1]
        for metric, expected in [('euclidiana',np.sqrt(10)),('manhattan',6),('hamming',1),('cosseno',1),('pearson',1)]:
            self.assertAlmostEqual(proximity(a,b,metric)[0],expected)
    def test_missing(self):
        self.assertEqual(proximity([5,np.nan],[5,4])[1],0)
        self.assertEqual(proximity([3,3],[4,5],'pearson')[1],0)
    def test_prediction(self):
        matrix=pd.DataFrame([[5,4,5],[4,4,np.nan],[1,5,1]],index=[1,2,3],columns=[10,20,30])
        model=MovieKNN(matrix)
        neighbors=model.neighbors(pd.Series({10:5,20:4}))
        score,details=model.predict(30,neighbors,2)
        self.assertEqual([r[0] for r in details],[1,3])
        self.assertTrue(1<score<5)
        self.assertEqual(model.predict(30,neighbors,1)[0],5)
if __name__=='__main__': unittest.main()
