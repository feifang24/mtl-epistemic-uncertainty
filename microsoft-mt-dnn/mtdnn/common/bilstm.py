import torch
import torch.nn as nn
import torch.nn.functional as F
from mtdnn.configuration_mtdnn import MTDNNConfig
from mtdnn.tasks.config import MTDNNTaskDefs
from transformers import BertTokenizer
from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp_mods.numeric_field import NumericField
from allennlp.modules.similarity_functions import LinearSimilarity, DotProductSimilarity
from mtdnn.tasks.glue_tasks import CoLATask, MRPCTask, MultiNLITask, QQPTask, RTETask, \
                  QNLITask, QNLIv2Task, SNLITask, SSTTask, STSBTask, WNLITask

PATH_PREFIX = '../../glue_data/' #PATH_PREFIX + 'processed_data/mtl-sentence-representations/'

#ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnli', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnliv2', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
NAME2INFO = {'sst': (SSTTask, 'SST-2/'),
             'cola': (CoLATask, 'CoLA/'),
             'mrpc': (MRPCTask, 'MRPC/'),
             'qqp': (QQPTask, 'QQP'),
             'sts-b': (STSBTask, 'STS-B/'),
             'mnli': (MultiNLITask, 'MNLI/'),
             'qnli': (QNLITask, 'QNLI/'),
             'qnliv2': (QNLIv2Task, 'QNLIv2/'),
             'rte': (RTETask, 'RTE/'),
             'snli': (SNLITask, 'SNLI/'),
             'wnli': (WNLITask, 'WNLI/')
            }
for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], PATH_PREFIX + v[1])

ELMO_OPT_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
ELMO_WEIGHTS_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long

class BiLSTMElmoAttn(nn.Module):
    def __init__(self, config: MTDNNConfig):
        self.config = config
        elmo = Elmo(options_file=ELMO_OPT_PATH, weight_file=ELMO_WEIGHTS_PATH,
                    num_output_representations=1)
        d_inp_phrase = 1024
        d_hid_phrase = config.hidden_size
        phrase_layer = nn.LSTM(input_size=d_inp_phrase, 
                                    hidden_size=d_hid_phrase, 
                                    num_layers=config.num_hidden_layers,
                                    batch_first=True,
                                    bidirectional=True)

        d_inp_model = 4 * d_hid_phrase
        d_hid_model = d_hid_phrase
        modeling_layer = nn.LSTM(input_size=d_inp_model, 
                                      hidden_size=d_hid_model, 
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True)
        
        _, _, vocab, _ = self._build_tasks(list(self.tasks.keys()))

        sent_encoder = HeadlessSentEncoder(vocab, None, config.num_highway_layers,
                                           phrase_layer, dropout=config.dropout_p,
                                           cove_layer=None, elmo_layer=elmo)
        pair_encoder = HeadlessPairAttnEncoder(vocab, 
                                                    None,
                                                    config.num_highway_layers,
                                                    phrase_layer, 
                                                    DotProductSimilarity(),
                                                    cove_layer=None,
                                                    elmo_layer=elmo,
                                                    dropout=config.dropout_p)
        
        d_single = 2 * d_hid_phrase
        d_pair = 2 * d_hid_phrase
        self.model = MultiTaskModel(config, sent_encoder, pair_encoder)

        self.tasks = config.tasks
        self.task_types = config.task_types
        self.task_dropout_p = config.tasks_dropout_p
        self.tasks_nclass_list = config.tasks_nclass_list

        for task, task_id in self.tasks.items():
            d_task =  d_pair * 4 if task.lower() == 'sst' else d_single
            self.model.build_classifier(task_id, self.tasks_nclass_list[task_id], d_task)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        


    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        premise_mask=None,
        hyp_mask=None,
        task_id=0,
    ):
        decodings = [self.tokenizer.decode(ids[1:]) for ids in input_ids] # slice to get rid of [CLS]
        sents = [sent[:sent.rfind('[SEP]')] for sent in decodings] # strip [PAD]
        inputs = [sent.split('[SEP]') for sent in sents]
        input1 = [inp[0] for inp in inputs]
        input2 = None if len(inputs[0]) == 1 else [inp[1] for inp in inputs]
        return self.model(task_id=task_id, input1=input1, input2=input2)

    def _build_tasks(all_task_names):
        '''Prepare tasks'''
        train_task_names = all_task_names
        eval_task_names = all_task_names
        tasks = get_tasks(all_task_names, self.config.max_seq_len)

        max_v_sizes = {'word': args.max_word_v_size}
        token_indexer = {"elmo": ELMoTokenCharactersIndexer("elmo")}

        word2freq = get_words(tasks)
        vocab = get_vocab(word2freq, max_v_sizes)
        word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)
        preproc = {'word_embs': word_embs}
        for task in tasks:
            train, val, test = process_task(task, token_indexer, vocab)
            task.train_data = train
            task.val_data = val
            task.test_data = test
            del_field_tokens(task)
            preproc[task.name] = (train, val, test)
        log.info("\tFinished indexing tasks")
        pkl.dump(preproc, open(preproc_file, 'wb'))
        vocab.save_to_files(vocab_path)
        log.info("\tSaved data to %s", preproc_file)
        del word2freq

        train_tasks = [task for task in tasks if task.name in train_task_names]
        eval_tasks = [task for task in tasks if task.name in eval_task_names]
        log.info('\t  Training on %s', ', '.join([task.name for task in train_tasks]))
        log.info('\t  Evaluating on %s', ', '.join([task.name for task in eval_tasks]))
        return train_tasks, eval_tasks, vocab, word_embs


class MultiTaskModel(nn.Module):
    '''
    Playing around designing a class
    '''

    def __init__(self, config: MTDNNConfig, sent_encoder, pair_encoder):
        '''

        Args:
        '''
        super(MultiTaskModel, self).__init__()
        self.sent_encoder = sent_encoder
        self.pair_encoder = pair_encoder
        self.pair_enc_type = 'attn'

        self.dropout_cls = config.classifier_dropout
        self.d_hid_cls = config.classifier_hid_dim

    def build_classifier(self, task_id, n_class, d_inp):
        ''' Build a task specific prediction layer and register it '''
        dropout, d_hid = self.dropout_cls, self.d_hid_cls

        layer = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid), nn.Tanh(),
                                nn.Dropout(p=dropout), nn.Linear(d_hid, n_class))

        setattr(self, f'{task_id}_pred_layer', layer)

    def forward(self, task_id, input1=None, input2=None):
        '''
        Predict through model and task-specific prediction layer

        Args:
            - inputs (tuple(TODO))
            - pred_layer (nn.Module)
            - pair_input (int)

        Returns:
            - logits (TODO)
        '''
        pair_input = input2 is not None
        pred_layer = getattr(self, '%s_pred_layer' % task.name)
        if pair_input:
            if self.pair_enc_type == 'bow':
                sent1 = self.sent_encoder(input1)
                sent2 = self.sent_encoder(input2) # causes a bug with BiDAF
                logits = pred_layer(torch.cat([sent1, sent2, torch.abs(sent1 - sent2),
                                               sent1 * sent2], 1))
            else:
                pair_emb = self.pair_encoder(input1, input2)
                logits = pred_layer(pair_emb)

        else:
            sent_emb = self.sent_encoder(input1)
            logits = pred_layer(sent_emb)
        return logits

class HeadlessSentEncoder(Model):
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 cove_layer=None, elmo_layer=None, dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator(), regularizer= None):
        super(HeadlessSentEncoder, self).__init__(vocab)#, regularizer)

        if text_field_embedder is None:
            self._text_field_embedder = lambda x: x
            d_emb = 0
            self._highway_layer = lambda x: x
        else:
            self._text_field_embedder = text_field_embedder
            d_emb = text_field_embedder.get_output_dim()
            self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))
        self._phrase_layer = phrase_layer
        d_inp_phrase = phrase_layer.get_input_dim()
        self._cove = cove_layer
        self._elmo = elmo_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        #if d_emb != d_inp_phrase:
        if (cove_layer is None and elmo_layer is None and d_emb != d_inp_phrase) \
                or (cove_layer is not None and d_emb + 600 != d_inp_phrase) \
                or (elmo_layer is not None and d_emb + 1024 != d_inp_phrase):
            raise ConfigurationError("The output dimension of the text_field_embedder "
                                     "must match the input dimension of "
                                     "the phrase_encoder. Found {} and {} respectively." \
                                     .format(d_emb, d_inp_phrase))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, sent):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sent : Dict[str, torch.LongTensor]
            From a ``TextField``.

        Returns
        -------
        """
        sent_embs = self._highway_layer(self._text_field_embedder(sent))
        if self._cove is not None:
            sent_lens = torch.ne(sent['words'], self.pad_idx).long().sum(dim=-1).data
            sent_cove_embs = self._cove(sent['words'], sent_lens)
            sent_embs = torch.cat([sent_embs, sent_cove_embs], dim=-1)
        if self._elmo is not None:
            elmo_embs = self._elmo(sent['elmo'])
            if "words" in sent:
                sent_embs = torch.cat([sent_embs, elmo_embs['elmo_representations'][0]], dim=-1)
            else:
                sent_embs = elmo_embs['elmo_representations'][0]
        sent_embs = self._dropout(sent_embs)

        sent_mask = util.get_text_field_mask(sent).float()
        sent_lstm_mask = sent_mask if self._mask_lstms else None

        sent_enc = self._phrase_layer(sent_embs, sent_lstm_mask)
        if self._elmo is not None and len(elmo_embs['elmo_representations']) > 1:
            sent_enc = torch.cat([sent_enc, elmo_embs['elmo_representations'][1]], dim=-1)
        sent_enc = self._dropout(sent_enc)

        sent_mask = sent_mask.unsqueeze(dim=-1)
        sent_enc.data.masked_fill_(1 - sent_mask.byte().data, -float('inf'))
        return sent_enc.max(dim=1)[0]

class HeadlessPairAttnEncoder(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in after the bidirectional
        attention.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 attention_similarity_function, modeling_layer,
                 cove_layer=None, elmo_layer=None, deep_elmo=False,
                 dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator(), regularizer=None):
        super(HeadlessPairAttnEncoder, self).__init__(vocab)#, regularizer)

        if text_field_embedder is None: # just using ELMo embeddings
            self._text_field_embedder = lambda x: x
            d_emb = 0
            self._highway_layer = lambda x: x
        else:
            self._text_field_embedder = text_field_embedder
            d_emb = text_field_embedder.get_output_dim()
            self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))

        self._phrase_layer = phrase_layer
        self._matrix_attention = MatrixAttention()
        self._modeling_layer = modeling_layer
        self._cove = cove_layer
        self._elmo = elmo_layer
        self._deep_elmo = deep_elmo
        self.pad_idx = vocab.get_token_index(vocab._padding_token)

        d_inp_phrase = phrase_layer.get_input_dim()
        d_out_phrase = phrase_layer.get_output_dim()
        d_out_model = modeling_layer.get_output_dim()
        d_inp_model = modeling_layer.get_input_dim()
        self.output_dim = d_out_model

        if (elmo_layer is None and d_inp_model != 2 * d_out_phrase) or \
            (elmo_layer is not None and not deep_elmo and d_inp_model != 2 * d_out_phrase) or \
            (elmo_layer is not None and deep_elmo and d_inp_model != 2 * d_out_phrase + 1024):
            raise ConfigurationError("The input dimension to the modeling_layer must be "
                                     "equal to 4 times the encoding dimension of the phrase_layer. "
                                     "Found {} and 4 * {} respectively.".format(d_inp_model, d_out_phrase))
        if (cove_layer is None and elmo_layer is None and d_emb != d_inp_phrase) \
            or (cove_layer is not None and d_emb + 600 != d_inp_phrase) \
            or (elmo_layer is not None and d_emb + 1024 != d_inp_phrase):
            raise ConfigurationError("The output dimension of the text_field_embedder "
                                     "must match the input "
                                     "dimension of the phrase_encoder. Found {} and {} "
                                     "respectively.".format(d_emb, d_inp_phrase))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        s1 : Dict[str, torch.LongTensor]
            From a ``TextField``.
        s2 : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this s2 contains the answer to the
            s1, and predicts the beginning and ending positions of the answer within the
            s2.

        Returns
        -------
        pair_rep : torch.FloatTensor?
            Tensor representing the final output of the BiDAF model
            to be plugged into the next module

        """
        s1_embs = self._highway_layer(self._text_field_embedder(s1))
        s2_embs = self._highway_layer(self._text_field_embedder(s2))
        if self._elmo is not None:
            s1_elmo_embs = self._elmo(s1['elmo'])
            s2_elmo_embs = self._elmo(s2['elmo'])
            if "words" in s1:
                s1_embs = torch.cat([s1_embs, s1_elmo_embs['elmo_representations'][0]], dim=-1)
                s2_embs = torch.cat([s2_embs, s2_elmo_embs['elmo_representations'][0]], dim=-1)
            else:
                s1_embs = s1_elmo_embs['elmo_representations'][0]
                s2_embs = s2_elmo_embs['elmo_representations'][0]
        if self._cove is not None:
            s1_lens = torch.ne(s1['words'], self.pad_idx).long().sum(dim=-1).data
            s2_lens = torch.ne(s2['words'], self.pad_idx).long().sum(dim=-1).data
            s1_cove_embs = self._cove(s1['words'], s1_lens)
            s1_embs = torch.cat([s1_embs, s1_cove_embs], dim=-1)
            s2_cove_embs = self._cove(s2['words'], s2_lens)
            s2_embs = torch.cat([s2_embs, s2_cove_embs], dim=-1)
        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        if self._mask_lstms:
            s1_mask = s1_lstm_mask = util.get_text_field_mask(s1).float()
            s2_mask = s2_lstm_mask = util.get_text_field_mask(s2).float()
            s1_mask_2 = util.get_text_field_mask(s1).float()
            s2_mask_2 = util.get_text_field_mask(s2).float()
        else:
            s1_lstm_mask, s2_lstm_mask, s2_lstm_mask_2 = None, None, None

        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask)
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)

        # Similarity matrix
        # Shape: (batch_size, s2_length, s1_length)
        similarity_mat = self._matrix_attention(s2_enc, s1_enc)

        # s2 representation
        # Shape: (batch_size, s2_length, s1_length)
        s2_s1_attention = util.last_dim_softmax(similarity_mat, s1_mask)
        # Shape: (batch_size, s2_length, encoding_dim)
        s2_s1_vectors = util.weighted_sum(s1_enc, s2_s1_attention)
        # batch_size, seq_len, 4*enc_dim
        s2_w_context = torch.cat([s2_enc, s2_s1_vectors], 2)
        # s1 representation, using same attn method as for the s2 representation
        s1_s2_attention = util.last_dim_softmax(similarity_mat.transpose(1, 2).contiguous(), s2_mask)
        # Shape: (batch_size, s1_length, encoding_dim)
        s1_s2_vectors = util.weighted_sum(s2_enc, s1_s2_attention)
        s1_w_context = torch.cat([s1_enc, s1_s2_vectors], 2)
        if self._elmo is not None and self._deep_elmo:
            s1_w_context = torch.cat([s1_w_context, s1_elmo_embs['elmo_representations'][1]], dim=-1)
            s2_w_context = torch.cat([s2_w_context, s2_elmo_embs['elmo_representations'][1]], dim=-1)
        s1_w_context = self._dropout(s1_w_context)
        s2_w_context = self._dropout(s2_w_context)

        modeled_s2 = self._dropout(self._modeling_layer(s2_w_context, s2_lstm_mask))
        s2_mask_2 = s2_mask_2.unsqueeze(dim=-1)
        modeled_s2.data.masked_fill_(1 - s2_mask_2.byte().data, -float('inf'))
        s2_enc_attn = modeled_s2.max(dim=1)[0]
        modeled_s1 = self._dropout(self._modeling_layer(s1_w_context, s1_lstm_mask))
        s1_mask_2 = s1_mask_2.unsqueeze(dim=-1)
        modeled_s1.data.masked_fill_(1 - s1_mask_2.byte().data, -float('inf'))
        s1_enc_attn = modeled_s1.max(dim=1)[0]

        return torch.cat([s1_enc_attn, s2_enc_attn, torch.abs(s1_enc_attn - s2_enc_attn),
                          s1_enc_attn * s2_enc_attn], 1)


    @classmethod
    def from_params(cls, vocab, params):
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        num_highway_layers = params.pop("num_highway_layers")
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        dropout = params.pop('dropout', 0.2)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab, text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers, phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function, modeling_layer=modeling_layer,
                   dropout=dropout, mask_lstms=mask_lstms,
                   initializer=initializer, regularizer=regularizer)


def del_field_tokens(task):
    ''' Save memory by deleting the tokens that will no longer be used '''
    #all_instances = task.train_data.instances + task.val_data.instances + task.test_data.instances
    all_instances = task.train_data + task.val_data + task.test_data
    for instance in all_instances:
        if 'input1' in instance.fields:
            field = instance.fields['input1']
            del field.tokens
        if 'input2' in instance.fields:
            field = instance.fields['input2']
            del field.tokens

def get_tasks(task_names, max_seq_len):
    '''
    Load tasks
    '''
    tasks = []
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        pkl_path = NAME2INFO[name][1] + "%s_task.pkl" % name
        if os.path.isfile(pkl_path):
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            task = NAME2INFO[name][0](NAME2INFO[name][1], max_seq_len, name)
            pkl.dump(task, open(pkl_path, 'wb'))
        tasks.append(task)
    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks

def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq = defaultdict(int)

    def count_sentence(sentence):
        '''Update counts for words in the sentence'''
        for word in sentence:
            word2freq[word] += 1
        return

    for task in tasks:
        splits = [task.train_data_text, task.val_data_text, task.test_data_text]
        for split in [split for split in splits if split is not None]:
            for sentence in split[0]:
                count_sentence(sentence)
            if task.pair_input:
                for sentence in split[1]:
                    count_sentence(sentence)
    log.info("\tFinished counting words")
    return word2freq

def get_vocab(word2freq, max_v_sizes):
    '''Build vocabulary'''
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes['word'])
    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    for word, _ in words_by_freq[:max_v_sizes['word']]:
        vocab.add_token_to_namespace(word, 'tokens')
    log.info("\tFinished building vocab. Using %d words", vocab.get_vocab_size('tokens'))
    return vocab

def get_embeddings(vocab, vec_file, d_word):
    '''Get embeddings for the words in vocab'''
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, d_word) #np.zeros((word_v_size, d_word))
    with open(vec_file) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                idx = vocab.get_token_index(word)
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index('@@PADDING@@')] = 0.
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading embeddings")
    return embeddings

def process_task(task, token_indexer, vocab):
    '''
    Convert a task's splits into AllenNLP fields then
    Index the splits using the given vocab (experiment dependent)
    '''
    if hasattr(task, 'train_data_text') and task.train_data_text is not None:
        train = process_split(task.train_data_text, token_indexer, task.pair_input, task.categorical)
        #train.index_instances(vocab)
    else:
        train = None
    if hasattr(task, 'val_data_text') and task.val_data_text is not None:
        val = process_split(task.val_data_text, token_indexer, task.pair_input, task.categorical)
        #val.index_instances(vocab)
    else:
        val = None
    if hasattr(task, 'test_data_text') and task.test_data_text is not None:
        test = process_split(task.test_data_text, token_indexer, task.pair_input, task.categorical)
        #test.index_instances(vocab)
    else:
        test = None
    for instance in train + val + test:
        instance.index_fields(vocab)
    return train, val, test

def process_split(split, indexers, pair_input, categorical):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    if pair_input:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        inputs2 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[1]]
        if categorical:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[2]]
        else:
            labels = [NumericField(l) for l in split[-1]]

        if len(split) == 4: # numbered test examples
            idxs = [LabelField(l, label_namespace="idxs", skip_indexing=True) for l in split[3]]
            instances = [Instance({"input1": input1, "input2": input2, "label": label, "idx": idx})\
                          for (input1, input2, label, idx) in zip(inputs1, inputs2, labels, idxs)]

        else:
            instances = [Instance({"input1": input1, "input2": input2, "label": label}) for \
                            (input1, input2, label) in zip(inputs1, inputs2, labels)]

    else:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        if categorical:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[2]]
        else:
            labels = [NumericField(l) for l in split[2]]

        if len(split) == 4:
            idxs = [LabelField(l, label_namespace="idxs", skip_indexing=True) for l in split[3]]
            instances = [Instance({"input1": input1, "label": label, "idx": idx}) for \
                         (input1, label, idx) in zip(inputs1, labels, idxs)]
        else:
            instances = [Instance({"input1": input1, "label": label}) for (input1, label) in
                         zip(inputs1, labels)]
    return instances #DatasetReader(instances) #Batch(instances) #Dataset(instances)