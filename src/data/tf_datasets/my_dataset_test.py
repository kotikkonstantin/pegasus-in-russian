from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.summarization import my_dataset


class MyDatasetTest(testing.DatasetBuilderTestCase):
    #set your path to text corpora
    EXAMPLE_DIR = '/home/ma-user/work/origin_2500_300_clean_1'
    SKIP_REGISTERING = True
    IN_DEVELOPMENT=False
    DATASET_CLASS = my_dataset.MyDataset
    SPLITS = {
          "train": 5,  # Number of fake train example
          "validation": 5,  # Number of fake validation example
          "test": 5,  # Number of fake test example
          }
    DL_EXTRACT_RESULT = ""

if __name__ == "__main__":
    testing.test_main()
