import seqio
from typing import Optional
from pydantic import BaseModel

from seqio.vocabularies import Vocabulary
from tensorflow.python.framework.dtypes import (
    int32,
    DType
)
from tensorflow.python.platform.gfile import GFile
from sentencepiece import sentencepiece_model_pb2 as sentencepiece_model
from tfdsio.utils import logger

class ModelVocab(BaseModel):
    """
    Base Model for SPE Model
    """
    path: str
    size: int
    extra_ids: int = 100


_T5Vocabs = {
    'default': ModelVocab(
        path = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model', 
        size = 32000, 
        extra_ids = 100
    ),
    'mt5': ModelVocab(
        path = 'gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model',
        size = 250000,
        extra_ids = 100
    )
}

class Vocabs:
    
    @classmethod
    def get_vocabulary(
        cls, 
        name: Optional[str] = 'mt5',
        path: Optional[str] = None,
        size: Optional[int] = None,
        extra_ids: Optional[int] = None,
    ) -> Vocabulary:
        """
        Get a T5 Vocabulary
        """
        return _T5Vocabs.get(name, ModelVocab(path = path, size = size, extra_ids = extra_ids))

    @classmethod
    def load_sentencepiece_model(cls, vocab_path: str):
        """
        Load a SentencePiece Model
        """
        sp_model = sentencepiece_model.ModelProto()
        sp_model.ParseFromString(GFile(vocab_path).read())
        return sp_model

    @classmethod
    def calculate_sentencepiece_extra_ids(cls, vocab_path: str, show_extra: bool = False):
        """
        Calculate the number of extra IDs for a SentencePiece Model
        """
        sp_model = cls.load_sentencepiece_model(vocab_path)
        vocab_size = len(sp_model.pieces)
        extra_ids = 0 if vocab_size % 1000 == 0 else 1000 - (vocab_size % 1000)
        base_size = vocab_size - extra_ids
        if extra_ids and show_extra:
            logger.info(f'Extra IDs: {extra_ids}')
            for i, piece in enumerate(sp_model.pieces[base_size:]):
                logger.info(f'[{i+base_size}] -> {piece.piece}')

        logger.info(f'Extra IDs: {extra_ids}. SPE Model Size: {vocab_size}')
        return extra_ids


    @classmethod
    def get_output_features(
        cls,
        input_key: str = 'inputs', 
        target_key: str= 'targets',
        vocab: Optional[ModelVocab] = None,
        vocab_path: Optional[str] = None,
        vocab_name: Optional[str] = None,
        inputs_add_eos: bool = True,
        targets_add_eos: bool = True,
        inputs_required: bool = False, # For LM, Inputs are not required
        targets_required: bool = True,
        input_dtype: Optional[DType] = int32,
        target_dtype: Optional[DType] = int32,

        extra_ids: Optional[int] = None,
        reverse_extra_ids: Optional[bool] = False,

    ):
        if vocab: 
            vocab_path = vocab.path
            extra_ids = vocab.extra_ids
        
        elif vocab_name:
            vocab = cls.get_vocabulary(vocab_name)
            vocab_path = vocab.path
            extra_ids = vocab.extra_ids
        
        elif vocab_path and extra_ids is None:
            extra_ids = cls.calculate_sentencepiece_extra_ids(vocab_path)

        return {
            input_key: seqio.Feature(
                seqio.SentencePieceVocabulary(
                    vocab_path, 
                    extra_ids = extra_ids,
                    reverse_extra_ids = reverse_extra_ids,
                ),
                add_eos = inputs_add_eos, 
                required = inputs_required, 
                dtype = input_dtype
            ),
            target_key: seqio.Feature(
                seqio.SentencePieceVocabulary(
                    vocab_path, 
                    extra_ids = extra_ids,
                    reverse_extra_ids = reverse_extra_ids,
                ),
                add_eos = targets_add_eos, 
                required = targets_required,
                dtype = target_dtype
            ),
        }
