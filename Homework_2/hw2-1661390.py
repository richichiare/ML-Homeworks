import tensorflow as tf
import os

#Iterating throw the directory in which there is a directory for each boat 
def iterate_dir(dir_path):
    return next(os.walk(dir_path))[1]
'''
We need to store the entire path in the list
of image names because when calling the decode_images() function
it needs the path of images
'''
def images_label(path):
	labels_name = iterate_dir(path) #Taking boat names to build labels
	filename_list = list()
	label_list = list()
	index = -1
	for class_name in labels_name:
		index += 1
		img_list = os.listdir(path + '/' + class_name)
		for img in img_list:
			filename_list.append(path + '/' + class_name + '/' + img) #Contains images name
			label_list.append(index) #Each label matches to the same position of the above list 
	return filename_list, label_list

'''
This function is used to create two lists, one containing image names,
the other contains the respective labels of the test set
'''
def parsing_groundtruth(path):
	boat_label = dict()
	labels_name = iterate_dir('sc5') #Taking boat names to build labels
	i = 0
	for e in labels_name:
		boat_label[e] = i
		i = i+1
	test_images = []
	test_labels = []
	file = open(path+'/ground_truth.txt', 'r')
	lines = file.readlines()
	for l in lines:
		l = l.replace(":", "")
		l = l.replace(" ", "").strip()
		element = l.split(';')
		if((element[1] != 'SnapshotBarcaParziale') and (element[1] != 'SnapshotBarcaMultipla')):
			if(element[1] == "SnapshotAcqua"):
				test_images.append(path + '/' + element[0])
				test_labels.append(boat_label['Water'])
			elif(element[1] == 'Mototopocorto'):
				test_images.append(path + '/' + element[0])
				test_labels.append(boat_label['Mototopo'])
			else:
				test_images.append(path + '/' + element[0])
				test_labels.append(boat_label[element[1]])
	return test_images, test_labels

'''
- Reading each image (passing a list of image paths) transforming it into a tensor string
- train-label could be a vector of the same size of train_images containing for each image 
	the membership class (i.e. [1, 23, 4, 14, 0, ...]). Making it one_hot means transforming each
	entry into a tensor having 1 in the position specified by the number (i.e. for the previous example [[0,1,..,0], [0,...,1],[0,0,0,1,..,0]])?
- Decode each img into a tensor in greyscale format
- Resize each image such that to be 24x80 
'''
def decode_images(train_images, train_labels):
	img_file = tf.read_file(train_images)
	img_decoded = tf.image.decode_jpeg(img_file, channels=1)
	resized_decoded_img = tf.image.resize_images(img_decoded, [24, 80])
	label_tensor = tf.one_hot(train_labels, 24)
	return resized_decoded_img, label_tensor

'''
- From lists to tensors by using tf.constant
- Creating a tensorflow Dataset object with tf.dataset
'''
def create_dataset(filename_list, label_list):
	train_images = tf.constant(filename_list)
	train_labels = tf.constant(label_list)
	labeled_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
	return labeled_dataset.map(decode_images)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Functions for setting up the convolutional and max-pooling layers
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#x is the input image: the shape is None meaning the batch size could be anyone while 24*80 is the size of an image
x = tf.placeholder(tf.float32, shape=[None, 24, 80, 1])
#t is the tensor containing the membership class of each image
t = tf.placeholder(tf.float32, shape=[None, 24])
#W are the weights of the model
W = tf.Variable(tf.zeros([24*80,24]))
#while b is the bias of the model
b = tf.Variable(tf.zeros([24]))

'''
From here we start defining the CNN:
- first convolutional layer (1 max-pool)
- second convolutional layer (1 max-pool)
- first dense layer
- second dense layer
'''
# Reshape data to original shape
x_image = tf.reshape(x, [-1, 24, 80, 1])
tf.summary.image('input_image', x_image)

# Define shape of the first layer parameters (Filter tensor: `[height, width, in_channels, out_channels]`)
# Define first convolutional layer
with tf.name_scope('conv1'):
	W_conv1 = weight_variable([5, 5, 1, 32]) # 
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #ReLU then sets all negative values in the matrix in output to the conv2d to zero
														  #while all other values are kept constant.
	h_pool1 = max_pool_2x2(h_conv1)

# Define shape of the second layer parameters
with tf.name_scope('conv2'):
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

'''
Define first fully connected layer ($1024$ units)
NOTE: 6x20 is the size of the pooled image at the end of the second
convolutional layer
'''
with tf.name_scope('fc1'):
	W_fc1 = weight_variable([6*20*64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 6*20*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
with tf.name_scope('dropout'):
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Define second fully connected layer
with tf.name_scope('fc2'):
	W_fc2 = weight_variable([1024,24])
	b_fc2 = bias_variable([24])
	y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

# Define loss function based on softmax
with tf.name_scope('boat'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y_conv))

	# Use ADAM Optimizer to minimize the loss
	optimizer = tf.train.AdamOptimizer(1e-4)
	#optimizer = tf.train.GradientDescentOptimizer(0.01)
	train_step = optimizer.minimize(loss)

# Collect summary data of `loss` for visualization in TensorBoard
tf.summary.scalar('loss', loss)
feat2 = tf.slice(h_conv1,[0,0,0,0],[-1,-1,-1,1])
tf.summary.image('features_h_conv1', feat2)
feat1 = tf.slice(h_conv2,[0,0,0,0],[-1,-1,-1,1])
tf.summary.image('features_h_conv2', feat1)

# Define function to compute classification accuracy 
with tf.name_scope('output'):
	correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

with tf.name_scope("Confusion_Matrix"):
	cm = tf.contrib.metrics.confusion_matrix(tf.argmax(t, 1),tf.argmax(y_conv, 1), num_classes = 15)

summary = tf.summary.merge_all()

# Start interatctive session (use `with tf.Session() as sess:` instead for non-interactive code).
# Collect summaries and initialize variables.
sess = tf.InteractiveSession()

filename_list, label_list = images_label('sc5')
training_dataset = create_dataset(filename_list, label_list)

testnames_list, testlabels_list = parsing_groundtruth('sc5-2013-Mar-Apr-Test-20130412')
test_dataset = create_dataset(testnames_list, testlabels_list)

summary_writer = tf.summary.FileWriter('/Users/riccardochiaretti/Desktop/HW2-ML/tmp', sess.graph)
sess.run(tf.global_variables_initializer())
 
#This is done for the training set
dataset_shuffled = training_dataset.shuffle(buffer_size=5000)
dataset_repeated = dataset_shuffled.repeat(21)
batched_dataset = dataset_repeated.batch(50)
iterator = batched_dataset.make_one_shot_iterator()
next_el = iterator.get_next()

#This instead, is done for the test set
batched_testset = test_dataset.batch(len(testnames_list))
test_iterator = batched_testset.make_one_shot_iterator()
test_next_el = test_iterator.get_next()

for i in range(1001):
    batch = next_el
    batch = sess.run(batch)
    feed_dict = {x: batch[0], t: batch[1], keep_prob: 0.5}
    if i % 10 == 0:
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
        
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict=feed_dict)
   
dataset = sess.run(training_dataset.batch(4774).make_one_shot_iterator().get_next())
print('train accuracy %g' % accuracy.eval(feed_dict={x: dataset[0], t: dataset[1], keep_prob: 1.0}))

print(cm.eval(feed_dict={x: dataset[0], t: dataset[1], keep_prob: 1.0}))

testset = sess.run(test_next_el)
print('test accuracy %g' % accuracy.eval(feed_dict={x: testset[0], t: testset[1], keep_prob: 1.0}))

print(cm.eval(feed_dict={x: testset[0], t: testset[1], keep_prob: 1.0}))

checkpoint_file = os.path.join('/Users/riccardochiaretti/Desktop/HW2-ML/tmp', 'model.ckpt')
saver = tf.train.Saver()
saver.save(sess, checkpoint_file, global_step=1001)