import torch
from options  import opts
from DeepPS2 import DeepPS2
from options import opts
import pdb
from data_loader import load_dataset
import time
import warnings


warnings.filterwarnings("ignore")
args = opts.TrainOpts().parse()
model = USLE(args).cuda()
print("------------ Model Loaded Successfully ---------")
train_set  = load_dataset(args)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, num_workers=args.workers, pin_memory=args.cuda, shuffle=True)
dataset_size = len(train_loader)
print('------------- Data loaded successfully: \t Dataset Size: %d -----------' % len(train_set))
total_steps = 0

data={}
epoch_start_time = time.time()


for epoch in range(args.start_epoch, args.epochs):
	model.train()
	print('Model training started for Epoch: %d / %d' %(epoch, args.epochs))
	epoch_iter = 0
	for i, sample in enumerate(train_loader):
		for key,vals in sample.items():
			data[key] = vals.cuda()	

		# pdb.set_trace()
		model.forward(data, epoch) 
		# print("----------One iteration is successful-----------")
		

		total_steps += args.batch
		epoch_iter += args.batch	

		
		# print('Forward and Backward pass successful')
		# if total_steps % args.display_freq == 0:
		# 	model.get_results(epoch, epoch_iter)

		if total_steps % args.save_latest_freq == 0:
			# print('saving the latest model (epoch %d, total_steps %d)' %(epoch, total_steps))
			model.save('latest')
			model.get_results(epoch, epoch_iter)

	if epoch % args.save_epoch_freq == 0:
		print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
		model.save('latest')
		model.save(epoch)
		
	model.get_results(epoch, epoch_iter)
	model.save('latest')


	print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.epochs, time.time() - epoch_start_time))
	model.update_learning_rate()


	
			







