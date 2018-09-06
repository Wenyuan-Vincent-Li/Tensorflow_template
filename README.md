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

2. __Inputpipeline:__

3. __Model:__

4. __Training:__

5. __Develop:__

Other folders and files are:
* __Plots:__ folder that stores pics
* __README file:__  Github markdown.
* __.gitignore:__ used for customize the content for git synchronization.

## More Details
3. __Model:__
    * __model_base.py:__ provides a series of building blocks that you might
use in your NN, such as relu, leakyrelu, fully_connected layer etc. 
```python
def _relu(self, x):
    return tf.nn.relu(x)

def _leakyrelu(self, x, leak=0.2, name="lrelu"):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)
```

#### Examples of using the template


#### Useful Links:
[TensorFlow: A proposal of good practices for files, folders and models architecture](
https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3
)
###### TODO LIST:
 