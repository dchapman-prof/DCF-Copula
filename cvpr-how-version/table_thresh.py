import os
import sys


def mkdir(path):
	try:
		os.mkdir(path)
	except:
		print('cannot make'+path)


#---------------------------
# Read command arguments
#---------------------------
if (len(sys.argv)<4):
	print("Usage")
	print("  python3 table_thresh.py dataset model seed")
	sys.exit()
arg_dataset =     sys.argv[1]
arg_model   =     sys.argv[2]
arg_seed    = int(sys.argv[3])

attacks = ['none', 'fgsm', 'pgd', 'cw', 'deepfool']
defenses = ['random', 'weibull']
percentile_vals = list(range(0,105,5))


#
# Open the output files
#
mkdir('table_thresh')

outpre='table_thresh/%s_%s_%d' % (arg_dataset, arg_model, arg_seed)
outacc=outpre+'_acc.csv'
outloss=outpre+'_loss.csv'

facc=open(outacc,'w')
floss=open(outloss,'w')

#
# Write out the header
#
facc.write('percentile\t')
floss.write('percentile\t')
for attack in attacks:
	for defense in defenses:
		facc.write('%s_%s\t' % (attack, defense))
		floss.write('%s_%s\t' % (attack, defense))
facc.write('\n')
floss.write('\n')

#
# Write the accuracies and losses to the file
#
for percentile in percentile_vals:
	facc.write('%d\t' % percentile)
	floss.write('%d\t' % percentile)
	for attack in attacks:
		for defense in defenses:
			inpre = 'test_thresh_acc/%s_%s_%d_%s_%s_%02d' % (arg_dataset,arg_model,arg_seed,attack,defense,percentile)
			inacc = inpre + '_acc.txt'
			inloss = inpre + '_loss.txt'


			print('----------try acc')
			try:
				print('try to open', inacc)
				fin = open(inacc, 'r')
				print('fin', fin)
				val = float(fin.read())
				print('val', val)
				facc.write('%f\t' % val)
				print('close')
				fin.close()
				print('end try')
			except Exception as e:
				print('except', e)
				facc.write('\t')

			print('----------try loss')
			try:
				print('try to open', inloss)
				fin = open(inloss, 'r')
				print('fin', fin)
				val = float(fin.read())
				print('val', val)
				floss.write('%f\t' % val)
				print('close')
				fin.close()
				print('end try')
			except Exception as e:
				print('except', e)
				floss.write('\t')

	facc.write('\n')
	floss.write('\n')



#
# We're done
#
facc.close()
floss.close()

print('Done!')
