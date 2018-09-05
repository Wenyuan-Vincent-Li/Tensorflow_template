'''
This is a python file that used for evaluate the model.
'''
import tensorflow as tf
from Training.Saver import Saver
from Testing.eval_base import Evaler_base
from Testing.utils import save_dict_as_txt, initialize_uninitialized_vars, \
    convert_list_2_nparray


class Evaler(Evaler_base):
    def __init__(self, config, save_dir):
        super(Evaler, self).__init__()
        self.config = config
        self.save_dir = save_dir

    def evaler(self, model, dir_names=None, epoch=None):
        # Reset the tensorflow graph
        tf.reset_default_graph()
        # Input node
        init_val_op, val_input, val_lab = self._input_fn_eval()
        val_lab = tf.argmax(val_lab, axis=-1)
        # Build up the graph
        with tf.device('/gpu:0'):
            output_list, main_graph = self._build_test_graph(val_input, val_lab, model)

            # Add saver
        saver = Saver(self.save_dir)
        # List to store the results
        Out =  []

        # Create a session
        with tf.Session() as sess:
            # restore the weights
            _ = saver.restore(sess, dir_names=dir_names, epoch=epoch)
            # initialize the unitialized variables
            initialize_uninitialized_vars(sess)
            # initialize the dataset iterator
            sess.run(init_val_op)
            # start evaluation
            count = 1
            while True:
                try:
                    out = \
                        sess.run(output_list,\
                                 feed_dict={main_graph.is_training: False})
                    # store results
                    Out.append(out)
                    tf.logging.debug("The current validation sample batch num is {}." \
                                     .format(count))
                    count += 1
                except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
                    # print out the evaluation results
                    tf.logging.info("The validation results are: accuracy {:.2f}; roc_auc {:.2f}; pr_auc {:.2f}." \
                                    .format(out))
                    break
        return Out

    def _input_fn_eval(self):
        '''
        Function used to create the input node.
        '''
        pass

    def _build_test_graph(self, val_input, val_lab, model):
        '''
        Function used to create the eval graph
        :return outputnode and main_graph
        '''
        pass

    def _metric(self, real_lab, prediction, probs):
        '''
        Function used to create evaluation metric
        '''
        pass


if __name__ == "__main__":
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs
    tf.logging.set_verbosity(tf.logging.INFO)

    from config import Config
    from Model.GAN_00 import GAN_00 as Model
    from Testing.Analysis_00 import Analysis


    class TempConfig(Config):
        DATA_DIR = "Data/Data_File"
        FILE_MID_NAME = "_AFT_FS_All_MinMax"
        BATCH_SIZE = 10240
        SUMMARY = False
        SAVE = False


    # Create a global configuration object
    tmp_config = TempConfig()

    ## Specify the trained weights localtion
    save_dir = "Training/Weight_MinMax_GAN_Val_Unl"  # Folder that saves the trained weights
    # Specify the Run, choose either Run = None (find the latest run)
    # or use Run = ['Run_2018-08-03_15_10_34']
    Run = ['Run_2018-08-03_15_10_34']
    # Run = None
    epoch = 4  # Specify the epoch

    # Create a evaler object
    Eval = Evaler(tmp_config, save_dir)
    # Run evaluation
    val_lab, preds, logits, \
    output = Eval.evaler(Model, dir_names=Run, epoch=epoch)
