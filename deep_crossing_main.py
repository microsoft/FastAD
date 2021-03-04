import math
import os
import sys
import logging
import numpy as np

from param import FLAGS
import tensorflow as tf
import tensorflow.contrib.microsoft as mstf
from DeepCrossing import DeepCrossing
import pandas as pd
import AUCBoot

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.set_printoptions(threshold=np.nan, linewidth=np.nan)


def lpMap2rating(x):
    x = str(x).lower()
    if x in ['disjoint', 'aggregator', 'spam', 'very bad', 'verybad', 'detrimental']:
        return 'disjoint'
    elif x in ['overlap', 'fair', 'bad']:
        return 'overlap'
    elif x in ['subset', 'good']:
        return 'subset'
    elif x in ['superset', 'excellent']:
        return 'superset'
    elif x in ['same', 'perfect']:
        return 'same'
    return 'null'


def get_df(qid, labels, mtyp, scores, rat):
    df = pd.DataFrame(
        {'m:QueryId': qid, 'm:AdCopyJudgment': labels, 'm:MatchType': mtyp, 'm:Predicted': scores, 'm:Rating': rat})
    df = df[['m:QueryId', 'm:AdCopyJudgment', 'm:MatchType', 'm:Predicted', 'm:Rating']]
    df['m:Rating'] = df['m:Rating'].apply(lpMap2rating)
    df = df[df['m:Rating'].isin(['disjoint', 'overlap', 'subset', 'superset', 'same'])]
    return df


if __name__ == '__main__':

    scriptdir = os.path.dirname(os.path.realpath(__file__))

    training_files = mstf.DataSource.get_files_to_read(FLAGS.input_training_data_path)
    validation_files = mstf.DataSource.get_files_to_read(FLAGS.input_validation_data_path)
    dict_file = os.path.join(scriptdir, FLAGS.dict_path)

    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info('Training.....')

    logging.info('training_files: {}'.format(training_files))
    logging.info('validation_files: {}'.format(validation_files))
    logging.info('training_data_schema: {}'.format(FLAGS.training_data_schema))
    logging.info('validation_data_schema: {}'.format(FLAGS.validation_data_schema))
    logging.info('training_data_shuffle: {}'.format(FLAGS.training_data_shuffle))
    logging.info('dict: {}'.format(dict_file))
    logging.info('model_dir: {}'.format(FLAGS.output_model_path))
    logging.info('epochs: {}'.format(FLAGS.epochs))
    logging.info('learning_rate: {}'.format(FLAGS.learning_rate))
    logging.info('batch_size: {}'.format(FLAGS.batch_size))
    logging.info('dims: {}'.format(FLAGS.dims))
    logging.info('batch_num_to_print_loss: {}'.format(FLAGS.batch_num_to_print_loss))

    setting = mstf.RunSetting(sys.argv[1:])

    model = DeepCrossing(FLAGS.types.split(","), list(map(int, FLAGS.embedding.split(","))), FLAGS.names.split(","),
                         list(map(int, FLAGS.dims.split(","))), FLAGS.att_dim, FLAGS.embedding_dim,
                         list(map(int, FLAGS.res_dims.split(","))), FLAGS.dict_path,
                         FLAGS.summaries_histogram, FLAGS.pos_weight, list(map(int, FLAGS.deep_dims.split(","))),
                         FLAGS.cross_layers, FLAGS.double_cross, FLAGS.negative_count)


    def process_label_test(label):
        return label


    def share_info(x):
        return x


    def to_int(x):
        return int(x)


    def train(x):
        return True


    def not_train(x):
        return False


    training_ds = mstf.TextDataSource(training_files, FLAGS.batch_size, FLAGS.training_data_schema,
                                      shuffle=FLAGS.training_data_shuffle, post_process=dict(data=to_int,
                                                                                             label=process_label_test,
                                                                                             flabel=process_label_test,
                                                                                             QueryId=share_info,
                                                                                             MatchType=share_info,
                                                                                             Rating=share_info,
                                                                                             qice=share_info,
                                                                                             kice=share_info,
                                                                                             is_training=train))
    validation_ds = mstf.TextDataSource(validation_files, FLAGS.batch_size, FLAGS.validation_data_schema,
                                        post_process=dict(data=to_int, label=process_label_test,
                                                          flabel=process_label_test, QueryId=share_info,
                                                          MatchType=share_info, Rating=share_info, qice=share_info,
                                                          kice=share_info, is_training=not_train))

    trainer = mstf.Trainer.create_trainer(setting)
    trainer.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)

    trainer.setup(model)
    trainer.start()

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                         trainer._sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    if setting.job_name == 'worker':
        step = 0
        for epoch in range(FLAGS.epochs):

            if FLAGS.mode == 'train':
                total_loss = 0.0
                total_samples = 0
                total_loss_avg = 0
                i = 0

                for _, result, batch, samples in trainer.run(training_ds, 'train'):

                    # print(result['debug'])

                    loss = result['loss']
                    recall = result['recall']
                    precision = result['precision']

                    total_loss += loss
                    total_samples += samples
                    partial_loss_avg = loss / samples
                    total_loss_avg = total_loss / total_samples

                    step += 1
                    current_lr = FLAGS.learning_rate * math.sqrt(1 - 0.999 ** step) / (1 - FLAGS.beta1 ** step)

                    if i % FLAGS.batch_num_to_write_tensorboard == 0:
                        train_writer.add_summary(result['summary'], i)

                    if i % FLAGS.batch_num_to_print_loss == 0:
                        logging.info('Batch: {:f}, partial average loss: {:f}, total average loss: {:f}'
                                     .format(i, partial_loss_avg, total_loss_avg))
                        logging.info('recall score: {:f}, precision score: {:f}'
                                     .format(recall, precision))

                        if FLAGS.has_validation:
                            scores = []
                            labels = []
                            qid = []
                            mtyp = []
                            lp = []
                            total_recall = 0
                            total_precision = 0
                            test_loss = 0
                            j = 0

                            if i == 0:
                                test_res_path = FLAGS.output_model_path + str(epoch) + "epoch" + str(i) + ".csv"
                                print('Running in test mode, save results to ' + test_res_path)
                                outputter = tf.gfile.GFile(test_res_path, mode='w')

                            test_summary = None
                            for _, test_result, test_batch, _ in trainer.run(validation_ds, 'predict'):
                                scores.extend(test_result['score'])
                                labels.extend(test_batch['aclabel'])
                                qid.extend(test_batch['QueryId'])
                                mtyp.extend(test_batch['MatchType'])
                                lp.extend(test_batch['lplabel'])
                                total_recall += test_result['recall']
                                total_precision += test_result['precision']
                                test_loss += test_result['loss']
                                test_summary = test_result['summary']
                                if i == 0:
                                    for m in range(len(test_batch['label'])):
                                        out_str = ''
                                        for name in FLAGS.names.split(","):
                                            out_str += str(test_batch[name][m]) + "\t"
                                        out_str += str(test_result['score'][m][0]) + "\n"
                                        outputter.write(out_str)
                                j += 1
                            test_writer.add_summary(test_summary, i)

                            scores = np.concatenate(scores, axis=0).tolist()

                            if i == 0:
                                outputter.close()
                            print('test_loss:' + str(test_loss / len(labels)))
                            print('learning_rate:' + str(current_lr))
                            df = get_df(qid, labels, mtyp, scores, lp)
                            prauc, rocauc = AUCBoot.cal_AUC(df)
                            logging.info('test PR AUC:' + str(prauc))
                            logging.info('test ROC AUC:' + str(rocauc))

                            print('recall_score:' + str(total_recall / j))
                            print('precision_score:' + str(total_precision / j))

                    i += 1

                logging.info('Epoch {} finished, total average loss: {:f}'.format(epoch, total_loss_avg))

            elif FLAGS.mode == 'test':
                test_res_paths = [FLAGS.output_model_path + "/pred" + str(i) + ".txt"
                                  for i in range(FLAGS.score_file_part)]
                print('Running in test mode, save results to ' + str(test_res_paths))

                outputters = [tf.gfile.GFile(test_res_paths[i], mode='w') for i in range(FLAGS.score_file_part)]

                cnt = 0
                j = 0
                # columns = ['QueryId', 'label', 'MatchType', 'flabel', 'query','keyword','title',]
                for _, result, batch, _ in trainer.run(training_ds, 'predict'):
                    num = len(batch['data'])
                    for i in range(num):
                        line = ''
                        for pair in FLAGS.training_data_schema.split(","):
                            name = pair.split(":")[0]
                            line += str(batch[name][i]) + '\t'
                        line += str(result['score'][i][0]) + "\n"

                        outputters[j].write(line)
                    cnt += 1
                    j = (j + 1) % FLAGS.score_file_part
                    if cnt % 2000 == 0:
                        print('Batch %d finished.' % cnt)

                for i in range(FLAGS.score_file_part):
                    outputters[i].close()
                print('Test Done.')
                break

            print('PROGRESS: {:.2f}%'.format(100 * epoch / FLAGS.epochs))

            if FLAGS.output_model_path is not None and FLAGS.save_model_every_epoch:
                trainer.save_model(os.path.join(FLAGS.output_model_path, 'model_{}'.format(epoch)))

        # if FLAGS.output_model_path is not None:
        #     trainer.save_model(os.path.join(FLAGS.output_model_path, 'model_final'))

    trainer.stop()

    print('Done')
