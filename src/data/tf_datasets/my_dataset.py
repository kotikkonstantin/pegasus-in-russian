from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

#set your path to text corpora
DIR = '/home/ma-user/work/origin_2500_300_clean_1'

_DOCUMENT = "_DOCUMENT"
_SUMMARY = "_SUMMARY"

class MyDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.1.0')
    MANUAL_DOWNLOAD_INSTRUCTIONS = 'Here to be description of the dataset'
    SKIP_REGISTERING = True
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("This is the dataset for Huawei NLP courese. It contains Russian texts."),
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            _DOCUMENT: tfds.features.Text(),
            _SUMMARY: tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=(_DOCUMENT, _SUMMARY),
        )

    def _split_generators(self, dl_manager):
        # Download source data
     
        extracted_path = DIR

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "raw_texts": os.path.join(extracted_path, "train_src.broad.txt"),
                    "target_texts": os.path.join(extracted_path, "train_tgt.news.txt"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "raw_texts": os.path.join(extracted_path, "valid_src.broad.txt"),
                    "target_texts": os.path.join(extracted_path, "valid_tgt.news.txt"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "raw_texts": os.path.join(extracted_path, "test_src.broad.txt"),
                    "target_texts": os.path.join(extracted_path, "test_tgt.news.txt"),
                },
            ),
        ]
            
    def _generate_examples(self, raw_texts, target_texts):
        """Yields examples."""

        texts = None
        with tf.io.gfile.GFile(raw_texts) as f:
            texts = f.readlines()
        texts = [text.strip() for text in texts]

        targets = None
        with tf.io.gfile.GFile(target_texts) as f:
            targets = f.readlines()
        targets = [text.strip() for text in targets]

            
        for i, (text, target) in enumerate(zip(texts, targets)): 
            yield i, {_DOCUMENT: text, _SUMMARY: target}     