from sklearn.linear_model import LinearRegression


categorical_data_mapping = {}

def update_category_to_value_mapping(values, name):
    if name in categorical_data_mapping:
        pass
    else:
        categorical_data_mapping[name] = {}
        length = len(values)
        mapped_to = 0
        for e in values:
            a = [0 for i in range(mapped_to)] + [1] + [0 for i in range(length-mapped_to-1)]
            categorical_data_mapping[name][e] = a
            mapped_to += 1

def get_value_for(key, name):
    if key not in categorical_data_mapping[name]:
        for a in categorical_data_mapping[name]:
            return [0 for i in range(len(a))]
    else:
        return categorical_data_mapping[name][key]

# loads data into containers data_points and Y
def load_data(file_path, continuous_indexes, categorical_indexes, label_index, bid_id_index, bids=-1):
    Xs = []
    Y = []
    data = []
    counter = 0
    with open(file_path,"r") as input_file:
        for line in input_file:
            if counter > 0:
                data.append(line[:-1].split(','))
            counter += 1

    for i in continuous_indexes:
        Xs.append([int(x[i]) for x in data])

    for i in categorical_indexes:
        update_category_to_value_mapping(set([x[i] for x in data]),str(i))
        encodings = [get_value_for(x[i], str(i)) for x in data]
        for j in range(len(encodings[0])):
            feature = [encodings[k][j] for k in range(len(encodings))]
            Xs.append(feature)
    Y = [int(x[label_index]) for x in data]

    order = [x[bid_id_index] for x in data]

    data_points = []
    if len(Xs) > 0:
        for j in range(len(Xs[0])):
            data_points.append([Xs[i][j] for i in range(len(Xs))])

    if bids != -1:
    	return data_points, Y, order, [int(x[bids]) for x in data]
    else:
    	return data_points, Y, order

def load_data1(file_path, continuous_indexes, categorical_indexes, bid_id_index):
    Xs = []
    Y = []
    data = []
    counter = 0
    with open(file_path,"r") as input_file:
        for line in input_file:
            if counter > 0:
                data.append(line[:-1].split(','))
            counter += 1

    for i in continuous_indexes:
        Xs.append([int(x[i]) for x in data])

    for i in categorical_indexes:
        update_category_to_value_mapping(set([x[i] for x in data]),str(i))
        encodings = [get_value_for(x[i], str(i)) for x in data]
        for j in range(len(encodings[0])):
            feature = [encodings[k][j] for k in range(len(encodings))]
            Xs.append(feature)

    order = [x[bid_id_index] for x in data]

    data_points = []
    if len(Xs) > 0:
        for j in range(len(Xs[0])):
            data_points.append([Xs[i][j] for i in range(len(Xs))])

    return data_points, order

continuous_indexes = [16]
categorical_indexes = [2,8,24]


# continuous_indexes = [15,16,19,22]
# categorical_indexes = [1,2,6,8,9,10,17,18,20,23,24,25]

label_index = 0
bid_id_index = 3
data_points_train, Y_train, order_train = load_data('train_10percent.csv', continuous_indexes, categorical_indexes, label_index, bid_id_index)
data_points_valid, order_valid = load_data1('test.csv', [15], [1,7,21], 2)

pctr_predictor = LinearRegression()
pctr_predictor.fit(data_points_train, Y_train)

Y_valid_prediction = pctr_predictor.predict(data_points_valid)
data_dict = {}
order_of_sorted_Y_valid = sorted(range(len(Y_valid_prediction)), key=lambda k: Y_valid_prediction[k])

spent = 0
num_of_clicks = 0
num_of_bids = 0
final_bids = [0 for i in range(len(Y_valid_prediction))]
for i in reversed(order_of_sorted_Y_valid):
	f = 200 * float(Y_valid_prediction[i]/0.00075)
	if f < 200:
		f = 0
	if f > 500:
		f = 500
	final_bids[i] = f
	spent+=f
	if spent > 12000000:
		break


with open('out.txt', 'w') as out_f:
	out_f.write('bid_id' + ',' + 'bid'+'\n')
	for i in range(len(final_bids)):
		out_f.write(str(order_valid[i]) + ',' + str(final_bids[i]) + '\n')
print final_bids














