def analyse_classes(file_name, col_index):
	points = []
	flag = True
	with open(file_name, 'r') as infile:
		for line in infile:
			if flag:
				flag = False
				print line.split(',')[col_index]
			else:
				line = line[:-1]
				points.append(line.split(',')[col_index])
	print set(points)
	print str(len(set(points))) + ' ' + str(col_index)
	print ''
	return len(set(points))

continuous = [15,16,19,21,22]
categorical = [1,2,0,6,8,9,10,17,18,20,23,24,25]
# for i in categorical+continuous:
# 	analyse_classes('train.csv', i)
print sum([analyse_classes('train.csv', i) for i in categorical])