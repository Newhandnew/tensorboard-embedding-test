import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import numpy as np

slim = tf.contrib.slim


def model(inputs, is_training, dropout_rate, num_classes, scope='Net'):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm):
            net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool1')
            tf.summary.histogram("conv1", net)

            net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool2')
            tf.summary.histogram("conv2", net)

            net = slim.flatten(net, scope='flatten')
            fc1 = slim.fully_connected(net, 1024, scope='fc1')
            tf.summary.histogram("fc1", fc1)

            net = slim.dropout(fc1, dropout_rate, is_training=is_training, scope='fc1-dropout')
            net = slim.fully_connected(net, num_classes, scope='fc2')

            return net, fc1


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits


# Parameters
learning_rate = 1e-4
total_epoch = 5001
batch_size = 100
display_step = 200
save_step = 1000
load_checkpoint = False
checkpoint_dir = "checkpoint"
checkpoint_name = 'model.ckpt'
logs_path = "logs"
test_size = 2000
projector_path = 'projector'

# Network Parameters
n_input = 28 * 28 #784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout_rate = 0.5 # Dropout, probability to keep units

mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name='InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')
is_training = tf.placeholder(tf.bool, name='IsTraining')
keep_prob = dropout_rate #tf.placeholder(tf.float32)  # dropout (keep probability)

logits, fc1 = model(x, is_training, keep_prob, n_classes)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
tf.summary.scalar("loss", loss)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

projectorDir = os.path.join(logs_path, projector_path)
pathForMetadata = os.path.join(projectorDir,'metadata.tsv')
pathForSprites = os.path.join(projectorDir, 'mnistdigits.png')
# check project directory
if not os.path.exists(projectorDir):
    os.makedirs(projectorDir)

# embedding program
mnistTest = input_data.read_data_sets('MNIST-data', one_hot=False)
batchXTest = mnistTest.test.images[:test_size]
batchYTest = mnistTest.test.labels[:test_size]

embedding_var = tf.Variable(tf.zeros([test_size, 1024]), name='embedding')
assignment = embedding_var.assign(fc1)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.join(projector_path,'metadata.tsv')
embedding.sprite.image_path = os.path.join(projector_path, 'mnistdigits.png')
embedding.sprite.single_image_dim.extend([28,28])

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

# Launch the graph
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    # Restore model weights from previously saved model
    prevModel = tf.train.get_checkpoint_state(checkpoint_dir)
    if load_checkpoint:
        if prevModel:
            saver.restore(sess, prevModel.model_checkpoint_path)
            print('Checkpoint found, {}'.format(prevModel))
        else:
            print('No checkpoint found')

    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    projector.visualize_embeddings(summary_writer, config)
    startTime = time.time()
    # Keep training until reach max iterations
    for epoch in range(total_epoch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # reshapeX = np.reshape(batch_x, [-1, 28, 28, 1])
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       is_training: True})
        if epoch % display_step == 0:
            # Calculate batch loss and accuracy
            cost, acc, summary = sess.run([loss, accuracy, merged_summary_op],
                                          feed_dict={x: batch_x,
                                                     y: batch_y,
                                                     is_training: False})
            elapsedTime = time.time() - startTime
            startTime = time.time()
            print('epoch {}, training accuracy: {:.4f}, loss: {:.5f}, time: {}'.format(epoch, acc, cost, elapsedTime))
            summary_writer.add_summary(summary, epoch)
        if epoch % save_step == 0:
            # Save model weights to disk
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            save_path = saver.save(sess, checkpoint_path)
            print("Model saved in file: {}".format(save_path))

    # save to log path
    saver.save(sess, os.path.join(logs_path, "model.ckpt"), 1)
    # create sprite image file
    to_visualise = batchXTest
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)
    sprite_image = create_sprite_image(to_visualise)
    # save sprite image file
    plt.imsave(pathForSprites, sprite_image, cmap='gray')
    # create metadata file
    with open(pathForMetadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(batchYTest):
            f.write("%d\t%d\n" % (index, label))

    print("Optimization Finished!")
