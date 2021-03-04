import tensorflow as tf

tf.app.flags.DEFINE_string('input-previous-model-path', 'model/', 'initial model dir')
tf.app.flags.DEFINE_string('input-previous-model-name', 'model_final', 'initial model name')
tf.app.flags.DEFINE_string('log-dir', 'summary/', 'initial summary dir')

tf.app.flags.DEFINE_string('input-training-data-path', 'data/train', 'training data dir')
tf.app.flags.DEFINE_string('input-validation-data-path', 'data/test', 'validation data dir')
tf.app.flags.DEFINE_string('output-model-path', 'model/', 'output model dir')
tf.app.flags.DEFINE_bool('save-model-every-epoch', True, 'whether save model every epoch')
tf.app.flags.DEFINE_string('dict-path', 'l3g.txt', 'path of x-letter dict')
tf.app.flags.DEFINE_integer('epochs', 4, 'epochs')
tf.app.flags.DEFINE_float('learning-rate', 0.0008, 'learning rate')
tf.app.flags.DEFINE_integer('batch-size', 128, 'batch size')
tf.app.flags.DEFINE_integer('batch-num-to-print-loss', 1000, 'number of batch number to print loss')
tf.app.flags.DEFINE_integer('batch-num-to-write-tensorboard', 200, 'number of batch number to write tensorboard')
tf.app.flags.DEFINE_float('beta1', 0.9, 'beta1 of adam')
tf.app.flags.DEFINE_float('pos_weight', 0.4, 'pos_weight of weighted cross entry')
tf.app.flags.DEFINE_integer('negative_count', 0, 'number of negative samples')

# tf.app.flags.DEFINE_string("types", "float,string,string,string,ice,ice,string,string,string",
#                            "types of input, including label")
# tf.app.flags.DEFINE_string("embedding", "1,1,1,1,1,1,1,1", "embedding of data(1 for True, 0 for False), excluding label")
# tf.app.flags.DEFINE_string("names", "label,query,keyword,title,qice,kice,url,desc,lptitle",
#                            "names of model input, including label")
# tf.app.flags.DEFINE_string("dims", "1,49292,49292,49292,3181,3181,49292,49292,49292",
#                            "dims of model input, including label")

# tf.app.flags.DEFINE_string("types", "float,string,string,string,ice,ice,string,string",
#                            "types of input, including label")
# tf.app.flags.DEFINE_string("embedding", "1,1,1,1,1,1,1", "embedding of data(1 for True, 0 for False), excluding label")
# tf.app.flags.DEFINE_string("names", "label,query,keyword,title,qice,kice,url,desc",
#                            "names of model input, including label")
# tf.app.flags.DEFINE_string("dims", "1,49292,49292,49292,3181,3181,49292,49292",
#                            "dims of model input, including label")

tf.app.flags.DEFINE_string("types", "float,string,string,string,ice,ice,ice,ice,string,string,web,web,string",
                           "types of input, including label")
tf.app.flags.DEFINE_string("embedding", "2,2,2,1,1,1,1,2,2,1,1,2",
                           "embedding of data(1 for True, 0 for False), excluding label")
tf.app.flags.DEFINE_string("names", "label,query,keyword,title,qice,kice,qice2,kice2,url,desc,mergeids,tids,lptitle",
                           "names of model input, including label")
tf.app.flags.DEFINE_string("dims", "1,49292,49292,49292,3181,3181,3181,3181,49292,49292,371839,371839,49292",
                           "dims of model input, including label")

tf.app.flags.DEFINE_integer("embedding_dim", 128, "embedding layer dim")
tf.app.flags.DEFINE_integer("att_dim", 1024, "attention dim")
tf.app.flags.DEFINE_string("res_dims", "256,256,256", "resnet layer dims")
tf.app.flags.DEFINE_string("deep_dims", "0", "deep layer dims")
tf.app.flags.DEFINE_integer("cross_layers", 0, "layers of cross")
tf.app.flags.DEFINE_bool("double_cross", True, "double cross")

# tf.app.flags.DEFINE_string('training-data-schema', 'data:0,label:1,query:2,keyword:3,title:4,desc:5,url:6,flabel:7,'
#                                                    'QueryId:10,MatchType:11,qice:12,kice:13,lptitle:16,is_training:16',
#                            'schema of training data')
# tf.app.flags.DEFINE_string('validation-data-schema', 'data:0,label:1,query:2,keyword:3,title:4,desc:5,url:6,flabel:7,'
#                                                      'QueryId:10,MatchType:11,qice:12,kice:13,'
#                                                      'lptitle:16,is_training:16',
#                            'schema of validation data')

# tf.app.flags.DEFINE_string('training-data-schema', 'data:0,label:1,query:2,keyword:3,title:4,desc:5,url:6,flabel:7,'
#                                                    'QueryId:10,MatchType:11,qice:12,kice:13,is_training:13',
#                            'schema of training data')
# tf.app.flags.DEFINE_string('validation-data-schema', 'data:0,label:1,query:2,keyword:3,title:4,desc:5,url:6,flabel:7,'
#                                                      'QueryId:10,MatchType:11,qice:12,kice:13,'
#                                                      'is_training:13',
#                            'schema of validation data')

tf.app.flags.DEFINE_string('training-data-schema', 'data:0,label:1,query:2,keyword:3,title:4,desc:5,url:6,'
                                                   'aclabel:7,lplabel:8,rating:9,QueryId:10,MatchType:11,'
                                                   'qice:12,kice:13,qice2:14,kice2:15,mergeids:16,tids:17,'
                                                   'lptitle:18,is_training:18',
                           'schema of training data')
tf.app.flags.DEFINE_string('validation-data-schema', 'data:0,label:1,query:2,keyword:3,title:4,desc:5,url:6,'
                                                     'aclabel:7,lplabel:8,rating:9,QueryId:10,MatchType:11,'
                                                     'qice:12,kice:13,qice2:14,kice2:15,mergeids:16,tids:17,'
                                                     'lptitle:18,is_training:18',
                           'schema of validation data')

tf.app.flags.DEFINE_bool('training-data-shuffle', True, 'if shuffle training data')
tf.app.flags.DEFINE_bool('has_validation', True, 'has validation')
tf.app.flags.DEFINE_bool('summaries_histogram', False, 'record histogram or not, will be slow if record')
tf.app.flags.DEFINE_string('mode', 'train', 'training or prediction mode')
tf.app.flags.DEFINE_integer('score_file_part', 1, 'num of score files')

FLAGS = tf.app.flags.FLAGS
