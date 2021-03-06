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


@DatasetReader.register('names-reader')
class NamesDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": TokenCharactersIndexer()}
    

    def name_to_instance(self, token: List[Token], label: str = None) -> Instance:
        name_field = TextField(token, self.token_indexers)
        fields = {"name": name_field}

        if label:
            label_field = LabelField(label=label)
            fields["label"] = label_field

        return Instance(fields)
        
    def _read(self, file_path: str ) -> Iterator[Instance]:

        all_letters = string.ascii_letters + " .,;'"
        n_letters = len(all_letters)
        names = []
        countries = []
        
        # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
        def unicodeToAscii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )
        
        # Read a file and split into lines
        def readLines(file_path):
            lines = open(file_path, encoding='utf-8').read().strip().split('\n')
            return [unicodeToAscii(line) for line in lines]     

        lines = readLines(file_path)

        for pair in lines:  
            yield self.name_to_instance([Token(pair.strip().split()[0])], pair.strip().split()[1])
        
        
        
            
@Model.register('names-model')
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        
        
        self.hidden2lang = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                name: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        
        mask = get_text_field_mask(name)
        embedding = self.word_embeddings(name)
        encoder_out = self.encoder(embedding, mask)
        lang_probs = self.hidden2lang(encoder_out)
        output = {"lang_probs": lang_probs}

        if label is not None:
            self.accuracy(lang_probs, label, mask)
            output["loss"] = sequence_cross_entropy_with_logits(lang_probs, label, mask)
            #SampledSoftmaxLoss(vocab.get_vocab_size('labels'),6,50)
            #masked_log_softmax(lang_probs, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
    


@Predictor.register('names-classifier')
class NamesTaggerPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        
        name = json_dict['name']

        
        return self._dataset_reader.name_to_instance(token=[Token(name)])

