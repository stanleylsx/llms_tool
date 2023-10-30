# -*- coding: utf-8 -*-
# @Time : 2023/10/24 22:29
# @Author : Mxoder
# @Email : mxode8@gmail.com
"""
基于 sentencepiece 实现
"""
import os
import shutil
import sentencepiece as sp
from transformers import AutoTokenizer, AutoModel


cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 训练词表
def train_vocab(logger,
                save_path,
                corpus,
                model_arch,
                vocab_size=8000,
                max_sentence_length=24000,
                character_coverage=0.9995
                ):
    logger.info('Start training the vocabulary.')
    sp.SentencePieceTrainer.train(
        # 只支持 txt 和 tsv 格式
        input=corpus,

        # 保存的模型前缀名
        model_prefix='bpe_expand',

        # 词表大小
        vocab_size=vocab_size,

        # 指定模型的字符覆盖率, 中文日文等推荐为 0.9995, 其余可以尝试 1.0
        character_coverage=character_coverage,

        # 分词算法
        model_type='bpe',

        # 是否将数字划分为单个 token, 在 llama 中是这么做的
        split_digits=True if model_arch == 'llama' else False,

        # 指定在遇到未知或很少的字符时将其分解为 UTF-8 字节, 开启后等效于 bbpe
        byte_fallback=True,

        # 指定输入句子的最大长度，以字节为单位
        max_sentence_length=max_sentence_length
    )
    bpe_model_path = os.path.join(os.path.dirname(par_dir), 'bpe_expand.model')
    bpe_vocab_path = os.path.join(os.path.dirname(par_dir), 'bpe_expand.vocab')
    shutil.move(bpe_model_path, save_path)
    shutil.move(bpe_vocab_path, save_path)
    logger.info(f'The vocabulary training is complete, saved to {save_path}.')


# 添加新词
def add_new_tokens(logger, tokenizer, save_path):
    logger.info('Start adding new tokens.')
    bpe_model = os.path.join(save_path, 'bpe_expand.model')
    sp_bpe = sp.SentencePieceProcessor()
    sp_bpe.load(bpe_model)

    raw_vocab = [sp_bpe.id_to_piece(id) for id in range(sp_bpe.get_piece_size())]
    clean_vocab = list(set(filter(is_chinese, raw_vocab)))

    tokenizer.add_tokens(clean_vocab)
    tokenizer.save_pretrained(save_path)
    logger.info(f'New tokens added, new tokenizer is saved to {save_path}.')

    return len(tokenizer)


# 初始化 embedding 层
# todo: 暂时只支持随机扩充，均值扩充在部分模型上没调好
def resize_embedding(logger, model, tokenizer_length, save_path):
    logger.info('Start resizing embedding.')
    new_length = int(tokenizer_length // 64 + 1) * 64
    model.resize_token_embeddings(new_length)

    model.save_pretrained(save_path)
    logger.info(f'New model: {model}')
    logger.info(f'Embedding resized, new model is saved to {save_path}.')


# 直接注入新词表
def inject_vocab(logger, tokenizer, save_path, corpus_list):
    logger.info('Start injecting new vocabulary.')

    all_words = []
    for file in corpus_list:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        words = [line.strip() for line in lines]
        all_words.extend(words)

    tokenizer.add_tokens(all_words)
    tokenizer.save_pretrained(save_path)
    logger.info(f'New vocabulary injected, new tokenizer is saved to {save_path}.')

    return len(tokenizer)


# 入口函数
def expand_vocab(logger,
                 model_path,
                 corpus_path,
                 model_arch,
                 save_path,
                 torch_dtype,
                 args
                 ):
    logger.info(f'Load base tokenizer from {model_path}.')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        # LLaMA 不用 TokenizerFast，表现有差异
        use_fast=False if model_arch == 'llama' else True
    )

    logger.info(f'Load base model from {model_path}'.capitalize)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )

    save_path = os.path.join(model_path, 'new_model') if save_path == 'auto' else save_path
    os.makedirs(save_path, exist_ok=True)
    logger.info(f'After expanding the vocabulary, the new model will be saved to {save_path}.')

    corpus_list = process_corpus(corpus_path)
    if args.expand_mode == 'inject':
        tokenizer_length = inject_vocab(logger, tokenizer, save_path, corpus_list)
    else:
        train_vocab(
            logger,
            save_path,
            corpus_list,
            model_arch,
            args.vocab_size,
            args.max_sentence_length
        )
        tokenizer_length = add_new_tokens(logger, tokenizer, save_path)
    resize_embedding(logger, model, tokenizer_length, save_path)
    logger.info('The vocabulary was successfully expanded.')


def process_corpus(corpus_path):
    ret_list = []
    if not os.path.isdir(corpus_path):
        if not corpus_path.endswith('.txt') and not corpus_path.endswith('.tsv'):
            raise ValueError('Only .txt or .tsv files are supported.')
        else:
            ret_list.append(corpus_path)
    else:
        file_list = os.listdir(corpus_path)
        for file in file_list:
            if not file.endswith('.txt') and not corpus_path.endswith('.tsv'):
                raise ValueError('Only .txt or .tsv files are supported.')
            else:
                ret_list.append(os.path.join(corpus_path, file))
    return ret_list


def is_chinese_char(cp):
    if ((
            cp >= 0x4E00 and cp <= 0x9FFF) or (
            cp >= 0x3400 and cp <= 0x4DBF) or (
            cp >= 0x20000 and cp <= 0x2A6DF) or (
            cp >= 0x2A700 and cp <= 0x2B73F) or (
            cp >= 0x2B740 and cp <= 0x2B81F) or (
            cp >= 0x2B820 and cp <= 0x2CEAF) or (
            cp >= 0xF900 and cp <= 0xFAFF) or (
            cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True
    return False


def is_chinese(word: str):
    for char in word:
        char = ord(char)
        if not is_chinese_char(char):
            return False
    return True
