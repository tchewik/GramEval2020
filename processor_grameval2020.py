# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import torch

from tqdm import tqdm

from allennlp.data.vocabulary import Vocabulary

from solution.train.main import Config, _build_model, _get_reader
from solution.train.lemmatize_helper import LemmatizeHelper
from solution.train.morpho_vectorizer import MorphoVectorizer

from typing import Iterable, Iterator, Optional, List, Any, Callable, Union

from isanlp.converter_conll_ud_v1 import ConverterConllUDV1
from isanlp.annotation_repr import CSentence

from isanlp.annotation import WordSynt

logger = logging.getLogger(__name__)

BERT_MAX_LENGTH = 512


class ProcessorGramEval2020:
    def __init__(self, 
                 model_path,
                 model_name="ru_bert_final_model",
                 batch_size=8,
                 pretrained_models_dir="pretrained_models/",
                 delay_init=False):
            
        self._model_path = model_path
        self._model_name = model_name
        self._batch_size = batch_size
        self._pretrained_models_dir = pretrained_models_dir
        self._checkpoint_name = "best.th"

        self.model = None
        if not delay_init:
            self.init()

    def init(self):
        
        if self.model is None:
            config = Config.load(os.path.join(self._model_path, 'config.json'))

        logger.info('Config: %s', config)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

        vocab = Vocabulary.from_files(os.path.join(self._model_path, 'vocab'))
        lemmatize_helper = LemmatizeHelper.load(self._model_path)
        morpho_vectorizer = MorphoVectorizer() if config.embedder.use_pymorphy else None
 
        self.model = _build_model(config, vocab, lemmatize_helper, morpho_vectorizer, bert_max_length=BERT_MAX_LENGTH)
        self.model.to(device)

        self.model.load_state_dict(torch.load(os.path.join(self._model_path, self._checkpoint_name), map_location=device))
        self.model.eval()
        
        self.reader = _get_reader(config, skip_labels=True, bert_max_length=BERT_MAX_LENGTH, reader_max_length=None)
        
        
    def __call__(self, tokens, sentences):
        """Performs tokenization, tagging, lemmatizing and parsing.
        
        Args:
            tokens(list): List of Token objects.
            sentences(list): List of Sentence objects.
            
        Returns:
            Dictionary that contains:
            
            1. lemma - list of lists of strings that represent lemmas of words.
            2. postag - list of lists of strings that represent postags of words.
            3. morph - list of lists of strings that represent morphological features.
            4. syntax_dep_tree - list of lists of objects WordSynt that represent a dependency tree.
        """
        
        result = {
            "lemma": [],
            "postag": [],
            "morph": [],
            "syntax_dep_tree": [],
        }
        
        for sentence in sentences:
            annotation = self._process_sentence(tokens[sentence.begin:sentence.end])
            result["lemma"].append(annotation[0])
            result["postag"].append(annotation[1])
            result["morph"].append(annotation[2])
            result["syntax_dep_tree"].append(annotation[3])        
        
        return self._process_text(tokens, sentences)
        
        
    def _process_sentence(self, tokens):
        
#         text_to_instance(self, words: List[str], upos_tags: List[str], dependencies: List[Tuple[str, int]] = None) → allennlp.data.instance.Instance[source]
        
        instances = self.reader.text_to_instance(words=tokens, 
                                                 upos_tags=['_',] * len(tokens),
                                                 dependencies=None)
            
        prediction = model.forward_on_instance(instance)       
        lemma = predictions['predicted_lemmas']
        postag = [predictions['predicted_gram_vals'][token_index].split('|', 1)[0] for token_index in range(len(tokens))]
        morph = [predictions['predicted_gram_vals'][token_index].split('|', 1)[1] for token_index in range(len(tokens))]
        syntax_dep_tree = [WordSynt(
            parent=int(predictions['predicted_heads'][token_index]) - 1, 
            link_name=predictions['predicted_dependencies'][token_index]
        ) for token_index in range(len(tokens))]
        
        return lemma, postag, morph, syntax_dep_tree
