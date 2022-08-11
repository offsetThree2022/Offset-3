import unittest
import pickle

class MyTestCase(unittest.TestCase):
    def test_something(self):
        pickname = '/media/user/C14D581BDA18EBFA/nuScenesGenData/val_list.pkl'
        with open(pickname, 'rb+') as f:
            val_list = pickle.load(f)
        debug = 1


if __name__ == '__main__':
    unittest.main()
