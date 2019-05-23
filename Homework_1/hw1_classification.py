import random
import os
import sys

#deciding the number of files (malware/benign) and how many of them
def select_files_dataset(num_malwares, num_non_malwares, num_applications, path_malware_csv, path_featuresvector):
	csv_file = open(path_malware_csv, 'r')
	csv_file.readline() #the first line is not useful
	malwares = []
	line = csv_file.readline()
	while(line): #reading all malwares
		malwares.append(line.split(",")[0]) #taking the hash of the malwares
		line = csv_file.readline()
	csv_file.close()
	app_list = os.listdir(path_featuresvector) #list of all the apps
	random.shuffle(app_list) #shuffling the list of all malware and benign files
	random.shuffle(malwares) #shuffling the list of all malwares files

	app_category = {} #each entry is like {filename : membership class}
	app_features = {} #each entry is like {filename : list of features}
	num_malwares_files = 0
	num_benign_files = 0

	if((num_malwares != 'x') & (num_non_malwares != 'x')): #it means that i'm choosing the specified num. of malwares and non-malwares
		#picking the specified number of malwares
		for count in range(0, num_malwares):
			f_malw = malwares[count]
			app_features[f_malw] = []
			app_category[f_malw] = 'malware'
		#picking the specified number of non-malwares
		j = 0
		counter =0
		while(counter < num_non_malwares):
			f_ben = app_list[j]
			if(f_ben not in malwares):
				app_features[f_ben] = []
				app_category[f_ben] = 'non-malware'
				counter += 1
				j += 1
			else:
				j += 1
	else: #i'm choosing at random from all the dataset
		for count in range(0, num_applications):
			f = app_list[count]
			app_features[f] = []
			if(f in malwares):
				num_malwares_files += 1
				app_category[f] = 'malware'
			else:
				num_benign_files += 1
				app_category[f] = 'non-malware'

	'''
	return -> [{filename : list of features}, {filename : membership class}]
	'''
	print('malwares: '+str(num_malwares_files), 'non-malwares: '+str(num_benign_files))
	return [app_features, app_category] 

'''
Opening each file in app_features:
	Scanning such files per line:
		Splitting the line
		Checking if the prefix is in the feature_list:
			Add that feature in the list corresponding to the file
		Otherwise pass to the next feature
'''
def collect_features(app_features, app_category, feature_list):
	num_malware_features = 0 #number of total malwares features
	num_benign_features = 0 #number of total benign features
	for application in app_features: #For each malw/benign file
		#could be optimized only reading the line
		file = open('drebin/feature_vectors/' + application, 'r')
		line = file.readline()
		while (line):
			splitted_line = line.split('::')
			if(splitted_line[0] in feature_list):
				x = splitted_line[1].strip()
				app_features[application].append(x) #appending each feature to the belonging file's list
				if(app_category[application] == 'malware'):
					num_malware_features += 1
				else:
					num_benign_features += 1
			line = file.readline()
		file.close()

'''
Searching through all files in the training set and selects only those whose belonging class is class_j
returns a dict containing only features:occurance of that feature in that file
'''
def obtain_class_files(training_set, app_features, app_category, class_j):
	features_occur = {}
	count = 0
	num_files = 0
	for file in training_set:
		if(app_category[file] == class_j):
			num_files += 1
			l = app_features.get(file)
			for f in l:
				features_occur[f] = features_occur.get(f, 0) + 1
				count += 1
	#[{features : occurrance}, total number of features, total number of files in that class]
	return [features_occur, count, num_files]

'''
The learn_naive_bayes() function applies the the naive Bayes approach computing
prior probabilities of each class and conditional probabilities of each feature 
'''
def learn_naive_bayes(training_set, app_features, app_category, bag_of_features):
	num_distinct_features = len(bag_of_features)
	num_all_training_set_files = len(training_set)
	prior_prob = {} #prior probability dict for classes prior prob
	cond_prob_malwares_features = {} #conditional probability dict for malwares features
	cond_prob_benign_features = {} #conditional probability dict for non-malwares features
	for class_j in ['malware', 'non-malware']:
		features_occur_counters = obtain_class_files(training_set, app_features, app_category ,class_j) #doc_j_num_features -> doc_j and num of features for that class
		features_occur = features_occur_counters[0] #{feature : occurrance} in the actual class
		t_j = features_occur_counters[2] #number of files in that class
		TF_j = features_occur_counters[1] #total number of features in class_j
		if(class_j == 'malware'):
			prior_prob['malware'] = t_j / num_all_training_set_files #prior probability of that malware class
			for feature in bag_of_features:
				TF_i_j = features_occur.get(feature, 0)
				cond_prob_malwares_features[feature] = (TF_i_j + 1)/(TF_j + num_distinct_features)
		else:
			prior_prob['non-malware'] = t_j / num_all_training_set_files #prior probability of non-malware class
			for feature in bag_of_features:
				TF_i_j = features_occur.get(feature, 0)
				cond_prob_benign_features[feature] = (TF_i_j + 1)/(TF_j + num_distinct_features)
	return (prior_prob, cond_prob_malwares_features, cond_prob_benign_features)

'''
The function below is the one whose task is to evaluate the work of 
the learning algorithm on a set distinct from the training one
'''
def evaluation(out, test_set, app_category, app_features, bag_of_features):
	prior_prob = out[0]
	cond_prob_malwares_features = out[1]
	cond_prob_benign_features = out[2]
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0
	error = 0
	for file in test_set:
		value_malw_benign = [prior_prob['malware'], prior_prob['non-malware']] #to store the output of evaluation
		list_features = app_features[file]
		for class_j in ['malware', 'non-malware']:
			if(class_j == 'malware'):
				internal_prod = 1.0 #it stores the internal product of each feature conditional probability
				for feature in list_features:
					if(feature in bag_of_features):
						internal_prod *= cond_prob_malwares_features.get(feature, 1)
				if(internal_prod != 0):
					value_malw_benign[0] *= internal_prod
			else:
				internal_prod = 1.0 #it stores the internal product of each feature conditional probability
				for feature in list_features:
					if(feature in bag_of_features):
						internal_prod *= cond_prob_malwares_features.get(feature, 1)
				if(internal_prod != 0):
					value_malw_benign[1] *= internal_prod
		if(app_category[file] == 'malware'): #if the file is a malware..
			if(value_malw_benign[0] >= value_malw_benign[1]): #..and i detected it
				true_positive += 1
			else: #if the file is a malware but i didn't detect it
				false_negative += 1
				error += 1
		else: #if the file is not a malware
			if(value_malw_benign[1] >= value_malw_benign[0]): #..and i've classified it rightly
				true_negative += 1
			else: #if the file is not a malware but i've classified it as a malware
				false_positive += 1
				error += 1
	error_si = error / len(test_set)
	return [error_si, true_positive, false_negative, true_negative, false_positive]

'''
Given the set of files in the training set, the compute_bag_of_features() function, computes all
distinct features contained in all lists in the app_features data structure
'''
def compute_bag_of_features(training_set, app_features):
	bag_of_features = {}
	for file in training_set:
		list_features = app_features.get(file)
		for feature in list_features:
			if(feature in bag_of_features):
				bag_of_features[feature] += 1
			else:
				bag_of_features[feature] = 1
	return bag_of_features

'''
The k_fold() function is doing at each iteration (for num_iterations times):
	derives the training set and the test set from the app_features
	computes the bag of features for elements in the training set
	calls the learning algorithm passing to it the training set and its bag of features
	it evaluates how many errors does the algorithm on the test set
At the end it computes the total accuracy
'''
def k_fold(app_features, app_category, num_iterations):
	k = num_iterations
	dataset_len_k = int(len(app_features) / k)
	list_files = list(app_features)
	random.shuffle(list_files)
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0
	total_error = 0

	for count in range(0, k):
		training_set = list_files.copy() #a copy for extracting the training set
		test_set = list_files[(count*dataset_len_k):((count*dataset_len_k)+dataset_len_k)] #this is the test set
		training_set[(count*dataset_len_k):((count*dataset_len_k)+dataset_len_k)] = [] #deleting from the training set files in the testset
		#compute bag of features for training set
		bag_of_features = compute_bag_of_features(training_set, app_features)
		out = learn_naive_bayes(training_set, app_features, app_category, bag_of_features)
		eval_out = evaluation(out, test_set, app_category, app_features, bag_of_features)
		total_error += eval_out[0]
		true_positive += eval_out[1]
		false_negative += eval_out[2]
		true_negative += eval_out[3]
		false_positive += eval_out[4]
	accuracy = 1 - (total_error / k)
	precision = true_positive/(true_positive+false_positive)
	recall = true_positive/(true_positive+false_negative)
	false_positive_rate = false_positive/(false_positive+true_negative)
	f_measure = (2*precision*recall)/(precision+recall)
	accuracy = (true_positive + true_negative)/(true_positive+false_negative+true_negative+false_positive)
	print('TP: ' + str(true_positive), 'FN: ' + str(false_negative))
	print('FP: '+ str(false_positive), 'TN: ' + str(true_negative))
	print('Precision: ' + str(precision))
	print('Recall: ' + str(recall))
	print('False positive rate: ' + str(false_positive_rate))
	print('F-measure: ' + str(f_measure))
	print(str(accuracy * 100) + '%')


'''
feature_list -> list of features to take into account
num_malwares, num_non_malwares -> if both 'x' taking randomly files from DREBIN otherwise taking num_malwares of malware and num_non_malwares of non malware
num_applications -> if the parameters above are 'x' than this is the number of files take in consideration for the learning problem
num_iterations -> number of iterations for the k-fold
'''
def main(parameters, feature_list, num_malwares, num_non_malwares, num_applications, num_iterations):

	if(len(parameters) != 3):
		print('To execute the program digit: python hw1_classification.py <your path for the sha256_family.csv file> <your path for feature_vectorsdirectory>')
		sys.exit()

	path_malware_csv = parameters[1]
	path_featuresvector = parameters[2]
	#selected_files returns -> [{filename : list of features}, {filename : membership class}]
	print('Considered features: '+ str(feature_list))
	selected_files = select_files_dataset(num_malwares, num_non_malwares, num_applications, path_malware_csv, path_featuresvector)
	app_features = selected_files[0]
	app_category = selected_files[1]
	collect_features(app_features, app_category, feature_list)
	k_fold(app_features, app_category, num_iterations)


#'drebin/sha256_family.csv', 'drebin/feature_vectors'
main(sys.argv, ['api_call', 'service_receiver', 'real_permission'], 'x', 'x', 5000, 10)






