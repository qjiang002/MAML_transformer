import os
'''
for f in os.listdir('./data'):
	if f.endswith('.train'):
		os.rename('./data/'+f, './data/train/'+f.replace('.train',''))
	if f.endswith('.test'):
		os.rename('./data/'+f, './data/test/'+f.replace('.test',''))
	if f.endswith('.dev'):
		os.rename('./data/'+f, './data/dev/'+f.replace('.dev',''))
'''
meta_train_list = []
meta_test_list = []
with open('./data/workspace.filtered.list','r') as f:
	for l in f:
		l=l.strip()
		meta_train_list.extend([l+'.t2', l+'.t4', l+'.t5'])
with open('./data/workspace.target.list','r') as f:
	for l in f:
		l=l.strip()
		meta_test_list.extend([l+'.t2', l+'.t4', l+'.t5'])

with open('./data/meta_train_tasks.list','w') as f:
	for t in meta_train_list:
		f.write(t+'\n') 

with open('./data/meta_test_tasks.list','w') as f:
	for t in meta_test_list:
		f.write(t+'\n') 