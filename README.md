# Tensorflow Template
This github project serves as a template that people can
use for fast prototyping machine learning models in
Tensorflow. If you find it useful, don't forget to star
the repo and let more people to know.

The principle behind this design is trying to isolate each
stage in machine learning modeling so that modifying
each module will note affect others. In other words, people
can easily use the same developed model in their own dataset,
or use the same dataset in different models.

This project is more a proposal than a definitive guide.
However I feel that it should cover most of the cases in 
machine learning when doing my own coding.

## The overall folder architecture
![WHole process for machine learning](/Plots/Overflow.png)

1. __Dataset:__ used for store and explore the new dataset for your own problem.
You can convert your dataset to tfrecord in it.

2. __Inputpipeline:__ used for read in the data from tfrecord or other sources and parser
the data to feed into the NN. The return arguments usually are: iterator, input_data,
target(if it is for supervised learning).

3. __Model:__ used for create the NN model.

4. __Training:__ used for training the NN model.

5. __Testing:__ used for testing and post-processing the results.

6. __Deploy:__ used for freeze the model and serve the tensorflow API.

Other folders and files are:
* __Plots:__ folder that stores pics
* __README file:__  Github markdown.
* __.gitignore:__ used to customize the content for git synchronization.

## More Details
1. __Dataset:__
    * __utils.py:__ provides a variety of functions that can be used for
    general data processing, including those convert image data and csv files
    to tfrecord.
    * __utils_dataset_spec.py:__ should be used to store the pre-processing functions
    that specific for the dataset.
2. __Inputpipeline:__
    * __ProstateDataSet.py:__ creates a dataset object. It should be modified to your
    own dataset. A typical data input pipeline includes: read in the data, parser the data,
     preprocessing the data, shuffle and repeat the data, batch the data up, 
     make the data iterator.
    * __inputpipeline.py:__ provides some functions that can be used in the 
    data input pipeline.
    * __input_source.py:__ shows several example that tensorflow can use for data
    input, such as input from numpy, input from numpy as placeholder, input from 
    tfrecord, etc.

    ```python
    def input_from_numpy(image, label):
        image = tf.convert_to_tensor(image, dtype=tf.int32, name="image")
        label = tf.convert_to_tensor(label, dtype = tf.int32, name="label")
        dataset = tf.data.Dataset.from_tensor_slices(
                {"input": image,
                 "target": label})
        return dataset
    
    def input_from_numpy_as_placeholder(image, label):
        input_placeholder = tf.placeholder(image.dtype, image.shape)
        target_placeholder = tf.placeholder(label.dtype, label.shape)
        dataset = tf.data.Dataset.from_tensor_slices((input_placeholder, \
                                                      target_placeholder))
        return dataset
    
    def input_from_tfrecord():
        filenames = tf.placeholder(tf.string, shape=[None])
        # make filenames as placeholder for training and validating purpose
        dataset = tf.data.TFRecordDataset(filenames)
        return dataset
    ```

3. __Model:__
    * __model_base.py:__ provides a series of building blocks that you might
    use in your NN, such as relu, leakyrelu, fully_connected layer etc. The
    model_base object will be inherited by the main NN model.
    ```python
    def _relu(self, x):
        return tf.nn.relu(x)
    
    def _leakyrelu(self, x, leak=0.2, name="lrelu"):
        with tf.name_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
    ```
    * __VGG_16.py:__ constructs the main model. A forward_pass method should be
    implemented within this object.
    ```python
    def forward_pass(self, x):
        with tf.name_scope('Conv_Block_0'):
            x = self._conv_batch_relu(x, filters = self._filters[0], \
                                  kernel_size = 3, strides = (1,1))
            x = self._max_pool(x, pool_size = 2)
        
        with tf.name_scope('Conv_Block_1'):
            x = self._conv_batch_relu(x, filters = self._filters[1], \
                                  kernel_size = 3, strides = (1,1))
                
        with tf.name_scope('Fully_Connected'):
            with tf.name_scope('Tensor_Flatten'):
                x = tf.reshape(x, shape = [self._batch_size, -1])
            x = self._fully_connected(x, self._num_classes)
        return x
    ``` 
4. __Training:__
    * __Saver.py:__ creates a saver object that save and restore training weights
    in tensorflow.
    * __Summary.py:__ creates a summary object that store the data in the training
    process. Data including scalar, image, histogram, graph, etc. can be utilized
    by tenorboard.
    * __train_base.py:__ a base function that can be inherited by Train.py.
    This base function includes different optimizer, metrics, etc. 
    * __Train.py:__ a main function that trains the model. 
    * __utils.py:__ utility function that being used by other training functions.

5. __Testing:__
    * __eval_base.py:__ a base function that can be inherited by Evaler.py.
    This base function includes different metrics, etc.
    * __Evaler.py:__ a main function that evaluates the model.
    * __utils.py:__ utility functions that being used by other evaluation 
    functions.

6. __Deploy:__
    * __deploy_base.py:__ a base function that can be inherited by Deploy.py.
    This base function includes import_meta_graph, extend_meta_graph, freeze_mode,
    etc.
    * __Deploy.py:__ a main function that deploys the model.
    * __construct_deploy_model.py:__ construct the model for production use. Put placeholder
    as an input interface.
    * __model_inspect.py:__ functions that can be used to inspect your trained ckpt file. 

#### Examples of using this template:


#### Useful links:

[TensorFlow: A proposal of good practices for files, folders and models architecture](
https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
)

[Mastering markdown in GitHub](https://guides.github.com/features/mastering-markdown/)

###### TODO LIST:
+ Combine all the utils files together

 