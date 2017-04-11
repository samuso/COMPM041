counter = 0
counter1 = 0
with open('train.csv', 'r') as input_file:
	for line in input_file:
		line = line[:-1]
		counter1+=1
		if line.split(',')[0] == '1':
			counter += 1
print counter 
print counter1 