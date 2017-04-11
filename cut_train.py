print 'sam'
counter = 0
buff = ''
with open('train.csv', 'r') as input_file:
	for line in input_file:
		buff += line
		counter += 1
		if counter == 250000:
			break

with open("train_10percent.csv", 'w') as output_file:
	output_file.write(buff)