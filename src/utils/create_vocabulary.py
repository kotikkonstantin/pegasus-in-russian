import os
import sentencepiece as spm

#set YOUR path!
PATH_TO_TEXT_CORPORA = "/home/ma-user/work/origin_2500_300_clean_1/"
#set YOUR path!
PATH_TO_AUXILIARY_DATA = "data/auxiliary/"

CHUNK_TEXT_SIZE = 300


def main():

    texts = None
    with open(PATH_TO_TEXT_CORPORA + "train_src.broad.txt", "r") as f:
        texts = f.readlines()

    texts = [text.strip() for text in texts]
    texts = " ".join(texts)

    targets = None
    with open(PATH_TO_TEXT_CORPORA + "train_tgt.news.txt", "r") as f:
        targets = f.readlines()

    targets = [text.strip() for text in targets]
    targets = " ".join(targets)

    all_texts = texts + " " + targets

    os.makedirs(PATH_TO_AUXILIARY_DATA, exist_ok=True)
    with open(PATH_TO_AUXILIARY_DATA + "text_to_create_vocabulary.txt", "w") as f:
        for i in range(0, len(all_texts) - CHUNK_TEXT_SIZE, CHUNK_TEXT_SIZE):
            f.write("%s\n" % all_texts[i:i + CHUNK_TEXT_SIZE])

    spm.SentencePieceTrainer.train(
         f"--input={PATH_TO_AUXILIARY_DATA}text_to_create_vocabulary.txt --model_prefix=pegasus/ckpt/russian_sport_news_bpe --model_type=bpe --hard_vocab_limit=false --vocab_size=96000 --user_defined_symbols=<pad>,<n>")

if __name__ == "__main__":
    main()