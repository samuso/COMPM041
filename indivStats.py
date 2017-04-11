import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as sm
import scipy

# For 3d plots. This import is necessary to have 3D plotting below
from mpl_toolkits.mplot3d import Axes3D

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
start = time.time()

print('Start')

def get_variables_from_maps(variables, order, year):
    variable = []
    for msoa_code in msoa_order:
        variable.append(float(variables[year][msoa_code]))
    return variable

# df = pd.read_csv('train.csv')
# df = pd.read_csv('validation.csv')

# advertisers = list(set(df['advertiser']))

# n = len(df)

# print('\nStats:')
# print('Advertiser\tImps\t\tClicks\t\tCost\t\tCTR\t\tCPM\t\tCPC\t\teCPC')

# for advertiser in advertisers:
#     rows = df[df['advertiser']==advertiser]
#     imps = len(rows)  # imps = number of impressions
#     clicks = len(rows[rows['click']==1])  # number of bids that were clicked
#     totalCost = sum(rows['payprice'].values)  # total money spent
#     clickedCost = sum(rows[rows['click']==1]['payprice'].values)  # money spent on bids that were clicked
#     ctr = 100 * float(clicks) / float(imps)  # ctr = click through rate
#     cpm = float(totalCost) / float(imps) / float(1000)  # cpm = cost per mille = cost per 1000 impressions
#     cpc = float(totalCost) / float(clicks)  # cpc = cost per click
#     ecpc = float(clickedCost) / float(clicks)  # eCPC = enhanced/effective cost per click (http://cpm.wiki/define/eCPC)
#     bid_price = rows['bidprice'].values
#     print('{} \t\t{} \t\t{} \t\t{}  \t{:.3f}% \t\t{:.3f} \t\t{:.2f} \t{:.2f}'.format(advertiser, imps, clicks, totalCost, ctr, cpm, cpc, ecpc))


# end = time.time()
# print('\nTime elapsed: {} minutes {} seconds'.format(int((end-start)/60), int((end-start) % 60)))

# print('Finish')

data = []
counter = 0
with open("train_10percent.csv","r") as input_file:
    for line in input_file:
        if counter > 1:
            data.append(line[:-1].split(','))
        counter += 1
print len(data)

indexes = [15,16,19,0]

Xs = []
Y = []
for i in indexes:
    if i  == indexes[len(indexes) - 1]:
        Y = [int(x[i]) for x in data]
    else:
        Xs.append([x[i] for x in data])

list_len = 249998

X1 = [int(Xs[0][i]) for i in range(list_len)]
X2 = [int(Xs[1][i]) for i in range(list_len)]
X3 = [int(Xs[2][i]) for i in range(list_len)]
Y = Y[:list_len]

X1 = scipy.stats.zscore(X1)
X2 = scipy.stats.zscore(X2)
X3 = scipy.stats.zscore(X3)

# print X1
# print X2
# print X3
# print Y

data = pandas.DataFrame({'x1': X1, 'x2' : X2, 'x3' : X3, 'y': Y})
model = ols("y ~ x1 + x2 + x3", data).fit()
# else:
#     data = pandas.DataFrame({'x1': X1, 'x2' : X2, 'y': Y})
#     model = ols("y ~ x1 + x2", data).fit()

# # Print the summary
# print "model for year: " + str(i)
print model.summary()
print model.predict(data[:10])







