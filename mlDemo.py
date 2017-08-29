import numpy as np
import pandas as pd
import graphlab

loanData = graphlab.SFrame('lc-data.gl/')

loanData['safe_loans'] = loanData['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loanData = loanData.remove_column('bad_loans')
# setting target data
target = 'safe_loans'

allLoans = loanData.to_dataframe()

# custom one hot encoder
def oneHotEncoding(dframe):
    from tqdm import tqdm
    StringFeatures=[]
    vlength = len(dframe)
    hlength = len(dframe.columns)
    [StringFeatures.append(col) for col in dframe.columns if isinstance(dframe[col][0],str)==True]
    featuresDict = {}
    numFeatures=len(StringFeatures)
    print("The number of features are: ",numFeatures)
    for feature in dframe.columns:
        iVars = set(dframe[feature])
        print('created ivars set')
        iVarsLength = len(iVars)
        if feature in StringFeatures:
            print('The current feature transforming: ',feature) 
            if iVarsLength<=50:
                newVals = dict(list(enumerate(iVars)))
                ListNewVal=[]
                [ListNewVal.append(x) for x in newVals]
                print ('New Values: ',ListNewVal)
                reverse_dict = {v:k for k,v in newVals.iteritems()}

                for m in tqdm(range(vlength)):
                    #for k in range(iVarsLength):
                    for newVal in reverse_dict:
                        #print ('comparing: %s with: %s')%(newVal,dframe[feature].iloc[m])
                        if newVal == dframe[feature].iloc[m]:
                            dframe=dframe.set_value(m,feature,reverse_dict[newVal])
                            #print ('the new val set was: ',reverse_dict[newVal])
                featuresDict[feature]=reverse_dict
            else:
                print('more than 50 vars. deleting: ',feature)
                del dframe[feature]
        else:
            print ('skipping %s, not a string feature')%(feature)
    return (featuresDict,dframe)

featuresDict,dframe=oneHotEncoding(allLoans)
#checkpoint save dframe to csv
tmp = '/Users/home/Desktop/devProjects/MLdemo/dframeNew.csv'
dframe.to_csv(path_or_buf=tmp)