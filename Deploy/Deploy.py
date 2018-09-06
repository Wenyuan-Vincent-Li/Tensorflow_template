import os, sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from Deploy.deploy_base import Deploy_base


# import the dataset module

class Deploy(Deploy_base):
    def __init__(self, model_path, save_path, **kwargs):
        super(Develop, self).__init__(model_path, save_path)

    def extend_meta_graph(self, clear_devices=True):
        """
        Extend the meta graph
        """
        # import the meta graph
        self.tf_graph = self._import_meta_graph(clear_devices)
        # add nodes to the metagraph
        output_logits = tf.get_collection('output_logits')[0]
        prediction = tf.argmax(output_logits, axis=-1, name='prediction')
        probs = tf.nn.softmax(output_logits, name='probs')

        collections_keys = self.tf_graph.get_all_collection_keys()
        tf.logging.debug(collections_keys)
        tf.logging.debug(self.tf_graph.get_tensor_by_name('probs:0'))

    def freeze_model(self):
        tf.logging.debug([n.name for n in \
                          tf.get_default_graph().as_graph_def().node])
        output_node_names = "prediction,probs"
        self._freeze_model(self.save_path, output_node_names)

    def use_frozen_model(self, config):
        self.tf_graph = self._load_frozen_model()
        iterator, val_input, val_lab = self._input_fn(config)
        tf.logging.debug([op.name for op in self.tf_graph.get_operations()])
        # We access the input and output nodes
        input_iterator = self.tf_graph.get_tensor_by_name('prefix/Input_Data/Iterator:0')
        input_getnext = self.tf_graph.get_tensor_by_name('prefix/Input_Data/IteratorGetNext:0')
        input_istraining = self.tf_graph.get_tensor_by_name('prefix/Is_training:0')
        predictions = self.tf_graph.get_tensor_by_name('prefix/prediction:0')
        probs = self.tf_graph.get_tensor_by_name('prefix/probs:0')

        ## TODO: Figure out how to fed the input data to frozen model
        with tf.Session(graph=self.tf_graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            predictions_out = sess.run(predictions, feed_dict={
                input_iterator: iterator,
                input_getnext: [val_input, val_lab],
                input_istraining: False
            })

    def _input_fn(self, config):
        """
        create a input function to replace the element in the frozen model
        :param config: the configuration file
        :return: input iterator, input data
        """
        pass

    def _severing_API(self):
        """
        TODO: severing the api using the framework
        :return:
        """
        pass

def _main_freeze_model():
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs
    model_path = "Deploy/FinalModel/model"
    save_path = "Deploy/ModelWrapper"
    develop = Develop(model_path, save_path)
    develop.extend_meta_graph()
    develop.freeze_model()


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs
    tf.logging.set_verbosity(tf.logging.INFO)
    from config import Config


    class TempConfig(Config):
        DATA_DIR = "Data/Data_File"
        FILE_MID_NAME = "_AFT_FS_All_MinMax"
        BATCH_SIZE = 10240


    # Create a global configuration object
    tmp_config = TempConfig()

    model_path = "Deploy/FinalModel/model"
    save_path = "Deploy/ModelWrapper"
    develop = Develop(model_path, save_path)
    develop.use_frozen_model(tmp_config)
