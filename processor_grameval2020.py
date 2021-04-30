# -*- coding: utf-8 -*-

import logging
import os

import torch
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from isanlp.annotation import WordSynt
from solution.train.lemmatize_helper import LemmatizeHelper
from solution.train.main import Config, _build_model, _get_reader
from solution.train.morpho_vectorizer import MorphoVectorizer

logger = logging.getLogger(__name__)

BERT_MAX_LENGTH = 512


class ProcessorGramEval2020:
    def __init__(self,
                 model_path,
                 model_name="ru_bert_final_model",
                 batch_size=8,
                 max_sentences=100,
                 pretrained_models_dir="pretrained_models/",
                 delay_init=False):

        self._model_path = model_path
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_sentences = max_sentences
        self._pretrained_models_dir = pretrained_models_dir
        self._checkpoint_name = "best.th"

        self.model = None
        if not delay_init:
            self.init()

    def init(self):

        config = Config.load(os.path.join(self._model_path, 'config.json'))

        logger.info('Config: %s', config)

        print("torch.cuda.is_available()", torch.cuda.is_available())
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu:0')

        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._skip_labels = False

        vocab = Vocabulary.from_files(os.path.join(self._model_path, 'vocab'))
        lemmatize_helper = LemmatizeHelper.load(self._model_path)
        self.morpho_vectorizer = MorphoVectorizer()

        self.model = _build_model(config, vocab, lemmatize_helper, self.morpho_vectorizer,
                                  bert_max_length=BERT_MAX_LENGTH)
        self.model.to(device)

        self.model.load_state_dict(
            torch.load(os.path.join(self._model_path, self._checkpoint_name), map_location=lambda storage, loc: storage))
        self.model.eval()

        self.reader = _get_reader(config, skip_labels=True, bert_max_length=BERT_MAX_LENGTH, reader_max_length=None)
        self.reader.text_to_instance = self.sentence_to_instance

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

        batches = len(sentences) // self._max_sentences + 1
        for batch_number in range(batches):
            instances = [self.sentence_to_instance([token.text for token in tokens[sentence.begin:sentence.end]]) for
                         sentence in sentences[batch_number * 100: min(len(sentences), (batch_number + 1) * 100)]]

            self.morpho_vectorizer.apply_to_instances(instances)
            predictions = self.model.forward_on_instances(instances)

            for annotation in predictions:
                result["lemma"].append(annotation["predicted_lemmas"])
                result["postag"].append([morphofeats.split('|', 1)[0] for morphofeats in annotation["predicted_gram_vals"]])

                morph = [morphofeats.split('|', 1)[1] for morphofeats in annotation["predicted_gram_vals"]]
                result["morph"].append([self._convert_morphology(word) for word in morph])

                syntax_dep_tree = [WordSynt(
                    parent=int(annotation["predicted_heads"][token_index]) - 1,
                    link_name=annotation["predicted_dependencies"][token_index]
                ) for token_index in range(len(annotation["predicted_heads"]))]

                result["syntax_dep_tree"].append(syntax_dep_tree)

        return result

    def _convert_morphology(self, word):
        if not '|' in word:
            return {}

        return {feature.split('=')[0]: feature.split('=')[1] for feature in word.split('|')}

    def sentence_to_instance(self, words, upos_tags=None) -> Instance:
        fields: Dict[str, Field] = {}
        metadata = {}

        text_field = TextField(list(map(Token, words)), self.reader._token_indexers)
        fields['words'] = text_field
        metadata['words'] = words

        if not upos_tags:
            upos_tags = ["X", ] * len(words)

        fields['pos_tags'] = SequenceLabelField(upos_tags, text_field, 'pos')
        metadata['pos'] = upos_tags

        metadata['lemmas'] = [0, ] * len(words)

        fields['grammar_values'] = SequenceLabelField(["X|_", ] * len(words), text_field, 'grammar_value_tags')
        fields['head_indices'] = SequenceLabelField([0, ] * len(words), text_field, 'head_index_tags')
        fields['head_tags'] = SequenceLabelField(["punct", ] * len(words), text_field, 'head_tags')

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
