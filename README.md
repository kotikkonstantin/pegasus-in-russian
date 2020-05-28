# PEGASUS finetuning on Russian sport text broadcasts

This project is inspired by the result of [Google Research](https://github.com/google-research/pegasus) -- the state of the art model for abstractive summarization problem called PEGASUS

## How Does It finetune PEGASUS on the custom dataset?

### 1. Install [PEGASUS library and dependencies](https://github.com/google-research/pegasus) including model weights and put it to the repository root
### 2. Add new finetuning dataset in format of [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/add_dataset):
#### 2.1.  Copy `my_dataset.py` and `my_dataset_test.py` from `src/data/tf_datasets/` to `~/anaconda3/lib/python3.7/site-packages/tensorflow_datasets/summarization` (if you are using conda env)
#### 2.2. In `~/anaconda3/lib/python3.7/site-packages/tensorflow_datasets/summarization`  edit `__init__.py` adding next line: `from tensorflow_datasets.summarization.my_dataset import MyDataset`
#### 2.3. In the file `~/anaconda3/lib/python3.7/site-packages/tensorflow_datasets/core/registered.py` set **variable `_skip_registration = True`** 
### 3. Prepare PEGASUS finetuning:
#### 3.1. In the file `pegasus/pegasus/params/public_params.py` add your custom transformer similar to other in this file:
```
@registry.register("russian_sport_news_transformer")
def russian_sport_news_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:my_dataset-train",
          "dev_pattern": "tfds:my_dataset-validation",
          "test_pattern": "tfds:my_dataset-test",
          "max_input_len": 512,
          "max_output_len": 256,
          "train_steps": 250,
          "learning_rate": 0.0001,
          "batch_size": 16,
      }, param_overrides)
```  
### 4. Create Russian Vocabulary by BPE in implementation of [sentencepiece](https://github.com/google/sentencepiece). Just run `python src/utils/create_vocabulary.py`

### 5. Launch finetuning through bash - `sh run.sh`:
```
cd pegasus

export PYTHONPATH=.

python pegasus/bin/train.py --params=russian_sport_news_transformer \
--param_overrides=vocab_filename=ckpt/russian_sport_news_bpe.model \
--train_init_checkpoint=ckpt/model.ckpt-1500000 \
--model_dir=russian_sport_news
```
### 6. Results
| Metric name | 95% lower bound | Mean | 95% upper bound |
|-------------|-----------------|------|-----------------|
|**rouge1-R**|0.020335|0.024443|0.028950|
|**rouge1-P**|0.094834|0.111006|0.127912|
|**rouge1-F**|0.030708|0.035601|0.041000|
|**rouge2-R**|0.002816|0.004676|0.007199|
|**rouge2-P**|0.015092|0.023864|0.033897|
|**rouge2-F**|0.003743|0.005679|0.007902|
|**rougeL-R**|0.019067|0.022773|0.026978|
|**rougeL-P**|0.090116|0.106110|0.124162|
|**rougeL-F**|0.028705|0.033412|0.038519|
|**rougeLsum-R**|0.019225|0.022945|0.027161|
|**rougeLsum-P**|0.089901|0.105752|0.121960|
|**rougeLsum-F**|0.028633|0.033411|0.038445|
|**bleu**|0.725627|0.763095|0.804040|

**based on 760 test texts**. See more in **results/text_metrics-250-.test.full.txt**

The PEGASUS was finetuned on 6143 Russian sport text broadcasts. The text broadcasts are about different football and hockey matches in Russian Language. The origin of used data is [sports.ru](https://www.sports.ru/)
