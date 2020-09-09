import tensorflow as tf
import evaluation_fns_np
import nn_utils


def create_metric_variable(name, shape, dtype):
  return tf.get_variable(name=name, shape=shape, dtype=dtype, trainable=False,
                         collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES])


def accuracy_tf(predictions, targets, mask):
  with tf.name_scope('accuracy'):
    return tf.metrics.accuracy(labels=targets, predictions=predictions, weights=mask)


def conll_srl_eval_tf(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                      gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets):

  with tf.name_scope('conll_srl_eval'):

    # create accumulator variables
    correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.int64)
    excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.int64)
    missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.int64)

    # indexes = tf.where(tf.equal(-1, tf.cast(predictions, tf.int32)))
    # predictions = tf.Print(predictions, [tf.gather_nd(predictions, indexes)], message='conll_srl_eval predictions srl', summarize=10)
    str_predictions = nn_utils.int_to_str_lookup_table(predictions, reverse_maps['srl'])

    # indexes = tf.where(tf.equal(-1, tf.cast(words, tf.int32)))
    # words = tf.Print(words, [tf.gather_nd(words, indexes)], message='conll_srl_eval words', summarize=10)
    str_words = nn_utils.int_to_str_lookup_table(words, reverse_maps['word'])

    # indexes = tf.where(tf.equal(-1, tf.cast(targets, tf.int32)))
    # targets = tf.Print(targets, [tf.gather_nd(targets, indexes)], message='conll_srl_eval targets srl', summarize=10)
    str_targets = nn_utils.int_to_str_lookup_table(targets, reverse_maps['srl'])

    # indexes = tf.where(tf.equal(-1, tf.cast(pos_predictions, tf.int32)))
    # pos_predictions = tf.Print(pos_predictions, [tf.gather_nd(pos_predictions, indexes)], message='conll_srl_eval targets gold-pos', summarize=10)
    str_pos_predictions = nn_utils.int_to_str_lookup_table(pos_predictions, reverse_maps['gold_pos'])

    # indexes = tf.where(tf.equal(-1, tf.cast(pos_targets, tf.int32)))
    # pos_targets = tf.Print(pos_targets, [tf.gather_nd(pos_targets, indexes)], message='conll_srl_eval pos targets', summarize=10)
    str_pos_targets = nn_utils.int_to_str_lookup_table(pos_targets, reverse_maps['gold_pos'])

    # need to pass through the stuff for pyfunc
    # pyfunc is necessary here since we need to write to disk
    py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, str_targets, predicate_targets,
                      pred_srl_eval_file, gold_srl_eval_file, str_pos_predictions, str_pos_targets]
    out_types = [tf.int64, tf.int64, tf.int64]
    correct, excess, missed = tf.py_func(evaluation_fns_np.conll_srl_eval, py_eval_inputs, out_types, stateful=False)

    update_correct_op = tf.assign_add(correct_count, correct)
    update_excess_op = tf.assign_add(excess_count, excess)
    update_missed_op = tf.assign_add(missed_count, missed)

    precision_update_op = update_correct_op / (update_correct_op + update_excess_op)
    recall_update_op = update_correct_op / (update_correct_op + update_missed_op)
    f1_update_op = 2 * precision_update_op * recall_update_op / (precision_update_op + recall_update_op)

    precision = correct_count / (correct_count + excess_count)
    recall = correct_count / (correct_count + missed_count)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, f1_update_op


def conll09_srl_eval_tf(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                      gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets, parse_head_targets,
                        parse_head_predictions, parse_label_targets, parse_label_predictions):

  with tf.name_scope('conll_srl_eval'):

    # create accumulator variables
    correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.int64)
    excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.int64)
    missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.int64)

    # first, use reverse maps to convert ints to strings

    # indexes = tf.where(tf.equal(-1, tf.cast(predictions, tf.int32)))
    # predictions = tf.Print(predictions, [tf.gather_nd(predictions, indexes)], message='conll09_srl_eval predictions',  summarize=10)
    str_predictions = nn_utils.int_to_str_lookup_table(predictions, reverse_maps['srl'])

    # indexes = tf.where(tf.equal(-1, tf.cast(words, tf.int32)))
    # words = tf.Print(words, [tf.gather_nd(words, indexes)], message='conll09_srl_eval words',  summarize=10)
    str_words = nn_utils.int_to_str_lookup_table(words, reverse_maps['word'])

    # indexes = tf.where(tf.equal(-1, tf.cast(targets, tf.int32)))
    # targets = tf.Print(targets, [tf.gather_nd(targets, indexes)], message='conll09_srl_eval targets',  summarize=10)
    str_srl_targets = nn_utils.int_to_str_lookup_table(targets, reverse_maps['srl'])

    # indexes = tf.where(tf.equal(-1, tf.cast(parse_label_targets, tf.int32)))
    # parse_label_targets = tf.Print(parse_label_targets, [tf.gather_nd(parse_label_targets, indexes)], message='conll09_srl_eval parse target',  summarize=10)
    str_parse_label_targets = nn_utils.int_to_str_lookup_table(parse_label_targets, reverse_maps['parse_label'])

    # indexes = tf.where(tf.equal(-1, tf.cast(parse_label_predictions, tf.int32)))
    # parse_label_predictions = tf.Print(parse_label_predictions, [tf.gather_nd(parse_label_predictions, indexes)], message='conll09_srl_eval parse pred',  summarize=10)
    str_parse_label_predictions = nn_utils.int_to_str_lookup_table(parse_label_predictions, reverse_maps['parse_label'])

    # indexes = tf.where(tf.equal(-1, tf.cast(pos_predictions, tf.int32)))
    # pos_predictions = tf.Print(pos_predictions, [tf.gather_nd(pos_predictions, indexes)], message='conll09_srl_eval pos pred',  summarize=10)
    str_pos_predictions = nn_utils.int_to_str_lookup_table(pos_predictions, reverse_maps['gold_pos'])

    # indexes = tf.where(tf.equal(-1, tf.cast(pos_targets, tf.int32)))
    # pos_targets = tf.Print(pos_targets, [tf.gather_nd(pos_targets, indexes)], message='conll09_srl_eval pos target',  summarize=10)
    str_pos_targets = nn_utils.int_to_str_lookup_table(pos_targets, reverse_maps['gold_pos'])

    # indexes = tf.where(tf.equal(-1, tf.cast(predicate_predictions, tf.int32)))
    # predicate_predictions = tf.Print(predicate_predictions, [tf.gather_nd(predicate_predictions, indexes)], message='conll09_srl_eval predicate pred',  summarize=10)
    str_predicate_predictions = nn_utils.int_to_str_lookup_table(predicate_predictions, reverse_maps['predicate'])

    # indexes = tf.where(tf.equal(-1, tf.cast(predicate_targets, tf.int32)))
    # predicate_targets = tf.Print(predicate_targets, [tf.gather_nd(predicate_targets, indexes)], message='conll09_srl_eval predicate target',  summarize=10)
    str_predicate_targets = nn_utils.int_to_str_lookup_table(predicate_targets, reverse_maps['predicate'])

    # need to pass through the stuff for pyfunc
    # pyfunc is necessary here since we need to write to disk
    py_eval_inputs = [str_predictions, str_predicate_predictions, str_words, mask, str_srl_targets, str_predicate_targets,
                      str_parse_label_predictions, parse_head_predictions, str_parse_label_targets, parse_head_targets,
                      str_pos_targets, str_pos_predictions, pred_srl_eval_file, gold_srl_eval_file]
    out_types = [tf.int64, tf.int64, tf.int64]
    correct, excess, missed = tf.py_func(evaluation_fns_np.conll09_srl_eval, py_eval_inputs, out_types, stateful=False)

    update_correct_op = tf.assign_add(correct_count, correct)
    update_excess_op = tf.assign_add(excess_count, excess)
    update_missed_op = tf.assign_add(missed_count, missed)

    precision_update_op = update_correct_op / (update_correct_op + update_excess_op)
    recall_update_op = update_correct_op / (update_correct_op + update_missed_op)
    f1_update_op = 2 * precision_update_op * recall_update_op / (precision_update_op + recall_update_op)

    precision = correct_count / (correct_count + excess_count)
    recall = correct_count / (correct_count + missed_count)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, f1_update_op


# todo share computation with srl eval
def conll_parse_eval_tf(predictions, targets, parse_head_predictions, words, mask, parse_head_targets, reverse_maps,
                   gold_parse_eval_file, pred_parse_eval_file, pos_targets):

  with tf.name_scope('conll_parse_eval'):

    # create accumulator variables
    total_count = create_metric_variable("total_count", shape=[], dtype=tf.int64)
    correct_count = create_metric_variable("correct_count", shape=[3], dtype=tf.int64)

    # indexes = tf.where(tf.equal(-1, tf.cast(words, tf.int32)))
    # words = tf.Print(words, [tf.gather_nd(words, indexes)],  message='conll_parse_eval words', summarize=10)
    str_words = nn_utils.int_to_str_lookup_table(words, reverse_maps['word'])

    # indexes = tf.where(tf.equal(-1, tf.cast(predictions, tf.int32)))
    # predictions = tf.Print(predictions, [tf.gather_nd(predictions, indexes)], message='conll_parse_eval parse pred', summarize=10)
    str_predictions = nn_utils.int_to_str_lookup_table(predictions, reverse_maps['parse_label'])
    # str_predictions = tf.Print(str_predictions, [str_predictions], message='str predictions')
    # print(str_predictions.dtype,'str predictions dtype')

    # indexes = tf.where(tf.equal(-1, tf.cast(targets, tf.int32)))
    # targets = tf.Print(targets, [tf.gather_nd(targets, indexes)], message='conll_parse_eval parse target', summarize=10)
    # print('Reverse maps for parse label:', {x: reverse_maps['parse_label'][x] for x in list(reverse_maps['parse_label'].keys())})
    str_targets = nn_utils.int_to_str_lookup_table(targets, reverse_maps['parse_label'])

    # indexes = tf.where(tf.equal(-1, tf.cast(pos_targets, tf.int32)))
    # pos_targets = tf.Print(pos_targets, [tf.gather_nd(pos_targets, indexes)], message='conll_parse_eval pos target', summarize=10)
    str_pos_targets = nn_utils.int_to_str_lookup_table(pos_targets, reverse_maps['gold_pos'])

    # need to pass through the stuff for pyfunc
    # pyfunc is necessary here since we need to write to disk
    py_eval_inputs = [str_predictions, parse_head_predictions, str_words, mask, str_targets, parse_head_targets,
                      pred_parse_eval_file, gold_parse_eval_file, str_pos_targets]
    out_types = [tf.int64, tf.int64]
    total, corrects = tf.py_func(evaluation_fns_np.conll_parse_eval, py_eval_inputs, out_types, stateful=False)

    update_total_count_op = tf.assign_add(total_count, total)
    update_correct_op = tf.assign_add(correct_count, corrects)

    update_op = update_correct_op / update_total_count_op

    accuracies = correct_count / total_count

    return accuracies, update_op


dispatcher = {
  'accuracy': accuracy_tf,
  'conll_srl_eval': conll_srl_eval_tf,
  'conll_parse_eval': conll_parse_eval_tf,
  'conll09_srl_eval': conll09_srl_eval_tf,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined evaluation function `%s' % fn_name)
    exit(1)


def get_params(task_outputs, task_map, train_outputs, features, labels, task_labels, reverse_maps, tokens_to_keep):

  # always pass through predictions, targets and mask
  params = {'predictions': task_outputs['predictions'], 'targets': task_labels, 'mask': tokens_to_keep}
  if 'params' in task_map:
    params_map = task_map['params']
    for param_name, param_values in params_map.items():
      if 'reverse_maps' in param_values:
        params[param_name] = {map_name: reverse_maps[map_name] for map_name in param_values['reverse_maps']}
      elif 'label' in param_values:
        params[param_name] = labels[param_values['label']]
      elif 'feature' in param_values:
        params[param_name] = features[param_values['feature']]
      elif 'layer' in param_values:
        outputs_layer = train_outputs[param_values['layer']]
        params[param_name] = outputs_layer[param_values['output']]
      else:
        params[param_name] = param_values['value']
  return params

