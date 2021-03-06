import unittest

import pandas as pd

from tree.utils import *


class MyTestCase(unittest.TestCase):
    watermellon = {
        'hardness': ["hard", "hard", "hard", "hard", "hard", "slimy", "slimy", "hard", "hard", "slimy", "hard", "slimy",
                     "hard", "hard", "slimy", "hard", "hard"]
        ,
        "good": ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "no", "no", "no", "no", "no", "no", "no",
                 "no"]}
    df = pd.DataFrame(data=watermellon)

    def test_entropy(self):
        self.assertAlmostEqual(entropy(self.df['hardness']), 0.874, 3)

    def test_infoGain(self):
        self.assertAlmostEqual(infoGain(self.df['hardness'], self.df['good']), 0.006, 3)

    def test_gainRatio(self):
        self.assertAlmostEqual(gainRatio(self.df['hardness'], self.df['good']), 0.006 / 0.874, 3)

if __name__ == '__main__':
    unittest.main()
