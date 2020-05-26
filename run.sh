cd pegasus

export PYTHONPATH=.

python pegasus/bin/train.py --params=russian_sport_news_transformer \
--param_overrides=vocab_filename=ckpt/russian_sport_news_bpe.model \
--train_init_checkpoint=ckpt/model.ckpt-1500000 \
--model_dir=russian_sport_news