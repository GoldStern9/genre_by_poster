import unittest
import pandas as pd
import prep_utilities as pu

class TestSum(unittest.TestCase):

    def test_get_feat_keys(self):
        df_t = pd.read_csv('../data/train.csv')
        features = list(set(df_t.columns)- {"Id","Genre",})
        f_dict = pu.get_features_and_count(features, df_t)
        self.assertEqual(len(f_dict.keys()), 25, "Should be 25")
    
    def test_get_feat_count(self):
        df_t = pd.read_csv('../data/train.csv')
        features = list(set(df_t.columns)- {"Id","Genre",})
        f_dict = pu.get_features_and_count(features, df_t)
        self.assertEqual(sum(f_dict.values()), 16874, "Should be 16874")
    
    def test_drop_genre_count(self):
        df_t = pd.read_csv('../data/train.csv')
        features = list(set(df_t.columns)- {"Id","Genre",})
        pu.drop_genre_by_count(pu.get_features_and_count(features, df_t), df_t, 10, True)
        self.assertEqual(len(df_t.columns), 26, "Should be 26")

if __name__ == '__main__':
    unittest.main()