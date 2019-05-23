import random

def select_files_dataset(num_malwares, num_benign):
	csv_file = open('drebin/sha256_family.csv', 'r')
	csv_file.readline() #the first line is not useful
	malwares = (csv_file.readlines().split(","))[0]
	app_list = os.listdir('drebin/feature_vectors') #all the apps
	malwares_dict = benign_dict = {}
	#searching for malwares
	for count in range(1, num_malwares):
		malwares_dict[malwares[random(0, len(malwares) - 1)]] #adding a malware app to the malwares I have to take into account
		print(malwares_dict)
	#searching for benign applications
	while(num_benign == 0):
		if(app_list[random(0, len(app_list) -1)] not in malwares): #not pick 
			benign_dict[app_list[random(0, len(app_list) -1)]]
			print(benign_dict)
			num_benign --
	
select_files_dataset(3, 2)