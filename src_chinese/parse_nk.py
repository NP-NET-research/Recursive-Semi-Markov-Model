from my_transformer import *

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import chart_helper
from my_chart_helper import *
import nkutil
import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

TAG_UNK = "UNK"

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
    }


class NKChartParser(nn.Module):
    # We never actually call forward() end-to-end as is typical for pytorch
    # modules, but this inheritance brings in good stuff like state dict
    # management.
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            hparams,
    ):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['hparams'] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab

        self.course_label_ix = []
        for i in range(0, Paras.COURSE_LABEL_SIZE):
            self.course_label_ix.append([])
        for i in range(0, label_vocab.size):
            label_str = label_vocab.value(i)
            if label_str != ():
                label_str = (label_str[0], )
            label_course = Paras.FINE_2_COURSE.get(label_str)
            self.course_label_ix[label_course].append(i)

        self.d_model = hparams.d_model
        self.partitioned = hparams.partitioned
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positional = (hparams.d_model // 2) if self.partitioned else None

        num_embeddings_map = {
            'tags': tag_vocab.size,
            'words': word_vocab.size,
            'chars': char_vocab.size,
        }
        emb_dropouts_map = {
            'tags': hparams.tag_emb_dropout,
            'words': hparams.word_emb_dropout,
        }

        self.emb_types = []
        if hparams.use_tags:
            self.emb_types.append('tags')
        if hparams.use_words:
            self.emb_types.append('words')

        self.use_tags = hparams.use_tags

        self.morpho_emb_dropout = None
        if hparams.use_chars_lstm or hparams.use_bert or hparams.use_bert_only:
            self.morpho_emb_dropout = hparams.morpho_emb_dropout
        else:
            assert self.emb_types, "Need at least one of: use_tags, use_words, use_chars_lstm, use_bert, use_bert_only"

        self.char_encoder = None
        self.bert = None

        self.bert_tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case)
        if hparams.bert_transliterate:
            from transliterate import TRANSLITERATIONS
            self.bert_transliterate = TRANSLITERATIONS[hparams.bert_transliterate]
        else:
            self.bert_transliterate = None

        d_bert_annotations = self.bert.pooler.dense.in_features
        self.bert_max_len = self.bert.embeddings.position_embeddings.num_embeddings

        self.project_bert = nn.Linear(d_bert_annotations, self.d_content, bias=False)

        self.embedding = MultiLevelEmbedding(
            [num_embeddings_map[emb_type] for emb_type in self.emb_types],
            hparams.d_model,
            d_positional=self.d_positional,
            dropout=hparams.embedding_dropout,
            timing_dropout=hparams.timing_dropout,
            emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
            extra_content_dropout=self.morpho_emb_dropout,
            max_len=hparams.sentence_max_len,
        )

        self.encoder = Encoder(
            self.embedding,
            num_layers=hparams.num_layers,
            num_heads=hparams.num_heads,
            d_kv=hparams.d_kv,
            d_ff=hparams.d_ff,
            d_positional=self.d_positional,
            num_layers_position_only=hparams.num_layers_position_only,
            relu_dropout=hparams.relu_dropout,
            residual_dropout=hparams.residual_dropout,
            attention_dropout=hparams.attention_dropout,
        )

        self.f_label = nn.Sequential(
            nn.Linear(hparams.d_model, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, label_vocab.size-1),
            )

        self.my_f_score = nn.Sequential(
            nn.Linear(hparams.d_model*2, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, Paras.COURSE_LABEL_SIZE*Paras.COURSE_LABEL_SIZE),
        )
        self.my_f_label = nn.Sequential(
            nn.Linear(hparams.d_model, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, label_vocab.size),
        )

        if use_cuda:
            self.cuda()


    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        hparams = spec['hparams']
        if 'use_chars_concat' in hparams and hparams['use_chars_concat']:
            raise NotImplementedError("Support for use_chars_concat has been removed")
        if 'sentence_max_len' not in hparams:
            hparams['sentence_max_len'] = 300
        if 'use_bert' not in hparams:
            hparams['use_bert'] = False
        if 'bert_transliterate' not in hparams:
            hparams['bert_transliterate'] = ""

        spec['hparams'] = nkutil.HParams(**hparams)
        res = cls(**spec)
        if use_cuda:
            res.cpu()
        state = {k: v for k, v in res.state_dict().items() if k not in model}
        state.update(model)
        res.load_state_dict(state)
        if use_cuda:
            res.cuda()
        return res

    def split_batch(self, sentences, golds, subbatch_max_tokens=3000):

        lens = [
            len(self.bert_tokenizer.tokenize(' '.join([word for (_, word) in sentence]))) + 2
            for sentence in sentences
        ]
        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()

        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def parse(self, sentence, gold=None):
        tree_list, loss_list = self.parse_batch([sentence], [gold] if gold is not None else None)
        return tree_list[0], loss_list[0]

    def parse_batch(self, sentences, golds=None):
        is_train = golds is not None
        self.train(is_train)
        torch.set_grad_enabled(is_train)

        if golds is None:
            golds = [None] * len(sentences)

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        word_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
                tag_idxs[i] = 0 if (not self.use_tags) else self.tag_vocab.index_or_unk(tag, TAG_UNK)
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_idxs[i] = self.word_vocab.index(word)
                batch_idxs[i] = snum
                i += 1
        assert i == packed_len

        batch_idxs = BatchIndices(batch_idxs)

        emb_idxs_map = {
            'tags': tag_idxs,
            'words': word_idxs,
        }
        emb_idxs = [
            from_numpy(emb_idxs_map[emb_type])
            for emb_type in self.emb_types
            ]

        all_input_ids = np.zeros((len(sentences), self.bert_max_len), dtype=int)
        all_input_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
        all_word_start_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
        all_word_end_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)

        subword_max_len = 0
        for snum, sentence in enumerate(sentences):
            tokens = []
            word_start_mask = []
            word_end_mask = []

            tokens.append("[CLS]")
            word_start_mask.append(1)
            word_end_mask.append(1)

            if self.bert_transliterate is None:
                cleaned_words = []
                for _, word in sentence:
                    word = BERT_TOKEN_MAPPING.get(word, word)
                    # This un-escaping for / and * was not yet added for the
                    # parser version in https://arxiv.org/abs/1812.11760v1
                    # and related model releases (e.g. benepar_en2)
                    word = word.replace('\\/', '/').replace('\\*', '*')
                    # Mid-token punctuation occurs in biomedical text
                    word = word.replace('-LSB-', '[').replace('-RSB-', ']')
                    word = word.replace('-LRB-', '(').replace('-RRB-', ')')
                    if word == "n't" and cleaned_words:
                        cleaned_words[-1] = cleaned_words[-1] + "n"
                        word = "'t"
                    cleaned_words.append(word)
            else:
                # When transliterating, assume that the token mapping is
                # taken care of elsewhere
                cleaned_words = [self.bert_transliterate(word) for _, word in sentence]

            for word in cleaned_words:
                word_tokens = self.bert_tokenizer.tokenize(word)
                for _ in range(len(word_tokens)):
                    word_start_mask.append(0)
                    word_end_mask.append(0)
                word_start_mask[len(tokens)] = 1
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("[SEP]")
            word_start_mask.append(1)
            word_end_mask.append(1)

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            subword_max_len = max(subword_max_len, len(input_ids))

            all_input_ids[snum, :len(input_ids)] = input_ids
            all_input_mask[snum, :len(input_mask)] = input_mask
            all_word_start_mask[snum, :len(word_start_mask)] = word_start_mask
            all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
        all_word_start_mask = from_numpy(np.ascontiguousarray(all_word_start_mask[:, :subword_max_len]))
        all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
        # all_encoder_layers, _ = self.bert(all_input_ids, attention_mask=all_input_mask)
        #
        # del _
        # features = all_encoder_layers[-1]
        features = self.bert(all_input_ids, attention_mask=all_input_mask)[0]

        features_packed = \
            features.masked_select(all_word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(-1, features.shape[-1])

        # For now, just project the features from the last word piece in each word
        extra_content_annotations = self.project_bert(features_packed)

        annotations, _ = self.encoder(emb_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)

        if self.partitioned:
            # Rearrange the annotations to ensure that the transition to
            # fenceposts captures an even split between position and content.
            # TODO(nikita): try alternatives, such as omitting position entirely
            annotations = torch.cat([
                annotations[:, 0::2],
                annotations[:, 1::2],
            ], 1)

        fencepost_annotations = torch.cat([
            annotations[:-1, :self.d_model // 2],
            annotations[1:, self.d_model // 2:],
        ], 1)
        fencepost_annotations_start = fencepost_annotations
        fencepost_annotations_end = fencepost_annotations

        # Note that the subtraction above creates fenceposts at sentence
        # boundaries, which are not used by our parser. Hence subtract 1
        # when creating fp_endpoints
        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if not is_train:
            trees = []
            scores = []

            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                sentence = sentences[i]
                tree, score = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentence, golds[i])
                trees.append(tree)
                scores.append(score)
            return trees, scores
        else:
            loss_total = None
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                tree, loss = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentences[i], golds[i])
                if loss_total is None:
                    loss_total = loss
                else:
                    loss_total = loss_total+loss
            return None, loss_total

    def parse_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end, sentence, gold=None):
        is_train = gold is not None

        seq_features = fencepost_annotations_end
        if is_train:
            gold_list = []
            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length
                    label = self.label_vocab.index(gold.oracle_label(left, right))
                    if label > 0 or length == 1:
                        gold_list.append((left, right, label))

            decoder_args = dict(
                seq_feature=seq_features,
                gold=gold_list,
                label_vocab=self.label_vocab,
                course_label_ix=self.course_label_ix,
                is_train=is_train,
                score_model=self.my_f_score,
                label_model=self.my_f_label,
                sent=sentence
            )

            tree, loss = my_decode(**decoder_args)
            return None, loss
        else:
            decoder_args = dict(
                seq_feature=seq_features,
                gold=None,
                label_vocab=self.label_vocab,
                course_label_ix=self.course_label_ix,
                is_train=is_train,
                score_model=self.my_f_score,
                label_model=self.my_f_label,
                sent=sentence
            )
            tree, loss = my_decode(**decoder_args)
            return tree, None

