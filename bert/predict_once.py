import os

import tensorflow as tf

import modeling
import tokenization

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

# Required parameters
tf.flags.DEFINE_string(
    "data_dir", 'data',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

tf.flags.DEFINE_string(
    "bert_config_file", 'chinese_L-12_H-768_A-12/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

tf.flags.DEFINE_string("vocab_file", 'chinese_L-12_H-768_A-12/vocab.txt',
                       "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string(
    "init_checkpoint", 'output',
    "Initial checkpoint (usually from a pre-trained BERT model).")

tf.flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

tf.flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

tf.flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

tf.flags.DEFINE_integer("iterations_per_loop", 1000,
                        "How many steps to make in each estimator call.")
tf.flags.DEFINE_string(
    "output_dir", 'output',
    "The output directory where the model checkpoints will be written.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = 0
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, input_ids, input_mask, segment_ids, labels, num_labels):
    model = modeling.BertModel(bert_config, False, input_ids, input_mask, segment_ids,
                               False)

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)

        return probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint):
    def model_fn(features, labels, mode, params):
        probabilities = create_model(
            bert_config, features["input_ids"], features["input_mask"],
            features["segment_ids"], features["label_ids"], num_labels)

        tvars = tf.trainable_variables()

        if init_checkpoint:
            assignment_map, names = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"probabilities": probabilities},
            scaffold_fn=None)
        return output_spec

    return model_fn


def one_epoch_input_fn(sentences, label_list, batch_size=8, seq_len=64):
    examples = [InputExample(guid=i, text_a=sent, text_b=None, label=None)
                for i, sent in enumerate(sentences)]

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    features = [convert_single_example(i, example, seq_len, tokenizer)
                for i, example in enumerate(examples)]

    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = zip(*((
        f.input_ids, f.input_mask, f.segment_ids, f.label_id
    ) for f in features))

    def input_fn(params):
        num_examples = len(examples)
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(
                all_input_ids, shape=[num_examples, seq_len], dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask, shape=[num_examples, seq_len], dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids, shape=[num_examples, seq_len], dtype=tf.int32),
            "label_ids":
                tf.constant(
                    all_label_ids, shape=[num_examples], dtype=tf.int32)
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def main(_):
    label_list = ['0', '1', '2']
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    run_config = tf.contrib.tpu.RunConfig(model_dir=FLAGS.output_dir)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=tf.contrib.tpu.RunConfig(),
        predict_batch_size=FLAGS.predict_batch_size)

    sentences = ['衣服太好看了', '这件衣服真好看']
    predict_input_fn = one_epoch_input_fn(sentences, label_list)
    result = estimator.predict(input_fn=predict_input_fn)

    for i, r in enumerate(result):
        if i >= 5:
            break
        print(r)

    return result


if __name__ == "__main__":
    main('')
