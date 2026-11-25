import unittest
import numpy as np
import pandas as pd
from src.hybrid import HybridKNN
class HybridTests(unittest.TestCase):
    def setUp(self):
        self.m=pd.DataFrame([[5,4,np.nan,5],[np.nan,np.nan,5,4],[2,1,2,1]],index=[1,2,3],columns=[10,20,30,40])
        self.g=pd.DataFrame([[1,0],[0,1],[1,0],[0,1]],index=self.m.columns)
        self.p=pd.Series({10:5,20:4})
    def test_no_overlap(self):
        model=HybridKNN(self.m,genres=self.g)
        self.assertIn(2,[r[1] for r in model.neighbors(self.p)])
        self.assertNotIn(2,[r[1] for r in HybridKNN(self.m,mode='tradicional').neighbors(self.p)])
    def test_support(self):
        model=HybridKNN(self.m,genres=self.g)
        recs,_=model.recommend(self.p.to_dict(),k=2,min_support=2)
        self.assertTrue(recs)
        self.assertTrue(all(n==2 for _,_,n in recs))
    def test_profile(self):
        model=HybridKNN(self.m,genres=self.g)
        np.testing.assert_allclose(model.genre_profile(self.p.reindex(self.m.columns).to_numpy()),[5,4])
        self.assertEqual(model.demographic_similarity({'occupation':'student'},{'occupation':'student'}),1)
        self.assertIsNone(model.demographic_similarity({},{}))
    def test_no_demographic_only(self):
        users=pd.DataFrame({'age':[24,24,24]},index=self.m.index)
        model=HybridKNN(self.m,genres=self.g,users=users,mode='perfil')
        self.assertEqual(model.neighbors(pd.Series(dtype=float),demographics={'age':24}),[])
