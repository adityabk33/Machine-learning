from utils import *
import pprint
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def naive_bayes():
	global pos_train, pos_test, neg_train, neg_test, vocab, counts_p, counts_n, prob_p, prob_n,prob_pos,prob_neg,total_neg,total_pos
	percentage_positive_instances_train = 0.1
	percentage_negative_instances_train = 0.5
	
	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))
	

	with open('vocab.txt','w', encoding="Utf-8") as f:
		for word in vocab:
			f.write("%s\n" % word)
	print("Vocabulary (training set):", len(vocab))
	
	
	prob_p = len(pos_train)/(len(pos_train) + len(neg_train))
	prob_n = len(neg_train)/(len(pos_train) + len(neg_train))
	
	counts_p={word: 0 for sublist in pos_train for word in sublist}
	for index, sublist in enumerate(pos_train):
		for word in sublist:
			counts_p[word] += 1
	total_pos = sum(counts_p.values())
			
	counts_n={word: 0 for sublist in neg_train for word in sublist}
	for index, sublist in enumerate(neg_train):
		for word in sublist:
			counts_n[word] += 1
	total_neg = sum(counts_n.values())
	print("Counting done!")
	
	prob_pos = {}
	prob_neg = {}
	for word in counts_p:
		prob_pos[word] = counts_p.get(word, 0)/total_pos
	for word in counts_n:
		prob_neg[word] = counts_n.get(word, 0)/total_neg
	print("probabilities found")
	
def prob_class(X_train,instance_test, k):
	if X_train == pos_train:
		prob = prob_pos
		total = total_pos
		counts = counts_p
	if X_train == neg_train:
		prob = prob_neg
		total = total_neg
		counts = counts_n
		
	condition = 0

	if any(word not in instance_test for word in counts_p.keys()):
		condition = 1
	else:
		condition = 2
	prob_test_instance = 0 
	p_t = {}
	prob_t = 0
	if condition == 2:
		for word_t in instance_test:
			p_t[word_t] = prob.get(word_t, 0)
		for p in p_t.values():
			prob_t += math.log(p)
			prob_test_instance = prob_t + math.log(len(X_train)/(len(pos_train) + len(neg_train)))
	else:
		for word_t in instance_test:
			if word_t in counts:
				p_t[word_t] = math.log((counts.get(word_t, 0) + k)/(total + k*len(vocab)))
			else:
				p_t[word_t] = math.log(k/(total + k*len(vocab)))
		for p in p_t.values():
			prob_t += p
			prob_test_instance = prob_t + math.log(len(X_train)/(len(pos_train) + len(neg_train)))
	return prob_test_instance

def prediction(instance_test, k):
	prob_test_p = prob_class(pos_train, instance_test, k)
	prob_test_n = prob_class(neg_train, instance_test, k)
	predict = 0
	if prob_test_p > prob_test_n:
		predict = 'pos'
	else:
		predict = 'neg'
		
	return predict

def accuracy(k):
	predict_pos_test = []
	predict_neg_test = []
	for index, instance_test in enumerate(pos_test):
		predict_pos_test.append(prediction(instance_test, k))
	for index, instance_test in enumerate(neg_test):
		predict_neg_test.append(prediction(instance_test, k))
	tp = 0
	tn = 0
	fn = 0
	fp = 0
	for i in predict_pos_test:
		if i == 'pos':
			tp += 1
		else:
			fn += 1
	for i in predict_neg_test:
		if i == 'neg':
			tn += 1
		else:
			fp += 1
	if k == 10:
		accuracy = (tp + tn)/(len(predict_pos_test) + len(predict_neg_test))
		precision = tp/(tp + fp)
		recall  = tp/(tp + fn)
		confusion_matrix = [[tn, fp],[fn, tp]]
		rownames = ['actual pos', 'actual neg']
		colnames = ['predicted pos', 'predicted neg']
		m = pd.DataFrame(confusion_matrix, index = rownames, columns = colnames)

		print("My log accuracy is: ",accuracy)
		print("The log precision is:", precision)
		print("The log recall is:",recall)
		print("The log confusion matrix is:\n", m)
	else:
		accuracy = (tp + tn)/(len(predict_pos_test) + len(predict_neg_test))
		print("My accuracy for k = " + str(k) + ", is: " + str(accuracy))
	return accuracy

if __name__=="__main__":
	naive_bayes()
	#a = [0.0001,0.001,0.01,0.1,1,10,100,1000]
	#cc = []
	#for i in a:
	#	acc.append(accuracy(i))
	#loga = [-4,-3,-2,-1,0,1,2,3]
	#plt.scatter(loga, acc)
	#plt.xlabel('log(a)')
	#plt.ylabel('accuracy')
	#plt.title('accuracy vs log(a)')
	#plt.show()
	accuracy(10)	
	
		


