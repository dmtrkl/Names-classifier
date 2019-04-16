from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
from overrides import overrides
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, masked_log_softmax,  masked_softmax
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss

from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


torch.manual_seed(1)





@Predictor.register('names-classifier')
class NamesTaggerPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        
        name = json_dict['name']
        
        return self._dataset_reader.text_to_instance(tokens=[Token(name)])