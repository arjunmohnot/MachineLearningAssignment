
# ID3 Algorithm

<img src="https://avatars0.githubusercontent.com/u/33459977?s=80&v=4" align="left"/>

---------------------------------------------------------------------------
About
=====

- **ARJUN MOHNOT**
- E17CSE102
- EB04
- Mobile: +91-7733993964
- [Website](https://arjun009.github.io)
- [WhatsApp](https://wa.me/917733993964?text=Hey%20Arjun%20Mohnot,%20I%27m%20contacting%20you%20from%20your%20Jupyter%20Notebook,%20A.I.-M.L.)
- [PlayStore](https://play.google.com/store/apps/developer?id=ARJUN+MOHNOT)

---------------------------------------------------------------------------

# Importing libraries


```python
import pandas as pd
import numpy as np
from collections import Counter
```

# Opening Dataframe


```python
openFile=pd.read_csv("DatasetTwo.csv")

```


```python
#Extract no. of features in the dataset through selected columns
colD=[i for i in openFile][2:] # do slicing here

#Printed list of column name will be considered for features
colD
```




    ['Branch', 'CGPA', 'Gamer', 'Movie_Fanatic', 'Committed?']




```python
openFile
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>S.No</th>
      <th>Branch</th>
      <th>CGPA</th>
      <th>Gamer</th>
      <th>Movie_Fanatic</th>
      <th>Committed?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>CSE</td>
      <td>High</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>CSE</td>
      <td>Low</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>CSE</td>
      <td>High</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>CSE</td>
      <td>High</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>CSE</td>
      <td>Low</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>6</td>
      <td>ECE</td>
      <td>Low</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
      <td>ECE</td>
      <td>High</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>8</td>
      <td>ECE</td>
      <td>Low</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>9</td>
      <td>ECE</td>
      <td>High</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>10</td>
      <td>ECE</td>
      <td>High</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>11</td>
      <td>MECH</td>
      <td>High</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>12</td>
      <td>MECH</td>
      <td>High</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>13</td>
      <td>MECH</td>
      <td>High</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>14</td>
      <td>MECH</td>
      <td>Low</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>15</td>
      <td>MECH</td>
      <td>Low</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
#openFile[openFile['Outlook']=='Sunny']
```

# Entropy (Entropy(S) = ∑ – p(I) . log2p(I))


```python
Yname=colD[-1]
def entropy(df,Yname=colD[-1]):
    
    dfDict=dict(Counter(df[Yname]))
    totalElements=sum(dfDict.values())
    entropy=0
    
    for i in dfDict:
        entropy-=(dfDict[i]/totalElements)*(np.log2(dfDict[i]/totalElements))
    if entropy==0:
        if len(dfDict)==1:
            entropy=str(list(dfDict.keys())[0])
    return entropy    
        


#entropyY=entropy(openFile[openFile['Outlook']=='Sunny'])
entropyY=entropy(openFile)

entropyY
```




    0.9967916319816366



# Information Gain (Gain(S, A) = Entropy(S) – ∑ [ p(S|A) . Entropy(S|A) ])


```python
def gain(decision,feature):
    
    df=decision
    Xname=feature
    uniqueFields=df[Xname].unique().tolist()
    totalElements=len(df)
    #print("length",len(df))
    gainScore=0


    for i in uniqueFields:
        currentDf=df[df[Xname]==i]
        calculate=entropy(currentDf)
       
        if type(calculate)==str:
            pass
        else:
            gainScore+=(len(currentDf)/totalElements)*entropy(currentDf)
        
        
  
       
    #print("gainScore",entropyY-gainScore)    
    return gainScore
        
        
```

# Helper Function


```python
def helperfunction(dataframe,root):
    
    Columns=colD[:-1] #ColY
    start=dataframe[root].unique().tolist()
    #print(start)
    d={}
    for i in start:
        df=dataframe[dataframe[root]==i]
        calculate=entropy(df)
        count=0
        for j in Columns:
            if type(calculate)==str:
                d[i]= "@"+calculate+"@"
                break
            else:
                value=calculate-gain(df,j)
                if count<value:
                    count=value
                    d[i]=j
                #d[str(i)+"-"+str(j)]= calculate-gain(df,j)
                
    return d     
    
```


```python
arrays=[]

def helperRoot(dataframe,Columns,checkKey=""):
    dStart={}
    d={}
    for i in Columns:
        d[i]=entropyY-gain(dataframe,i)
    

    zipper=dict(sorted(d.items(), key=lambda x: x[1],reverse=True))
    #print(zipper)

    for k,v in zipper.items():
        if v>0:
            if checkKey!=k:
                root=k
                break

            
    #Columns.remove(root)
    dStart[root]=helperfunction(dataframe,root)
    #print(dStart)
    arrays.append(dStart)
    return dStart

# Main functions

if __name__=="__main__":
    cols=colD[:-1]
    col=colD[:-1]

    a=helperRoot(openFile,col)
    key=str(list(a.keys())[0])
    rootKey=key
    print(rootKey)
    col.remove(key)
    arrayc=[]


    for i in a[key]:
        dataframe=openFile
        value=a[key][i]
        if value[0]!="@":
            dataframe=dataframe[dataframe[key]==i]
            fun=helperRoot(dataframe,col)
            arrayc.append(fun)
    #print(arrayc,"---------------------")    

    arrayq=arrayc
    print(a)
    print(arrayq)
    again=[]
    o=openFile[colD[-1]].unique().tolist()


    for j in range(len(arrayq)):

        key=str(list(arrayq[j].keys())[0])
        for i in arrayq[j][key]:
            try:
                
                dataframe=openFile[openFile[rootKey]==o[j]]
                value=arrayq[j][key][i]
                #print("value=---",value,"   !")
                if value[0]!="@":
                    print(value,key,i)
                    dataframe=dataframe[dataframe[key]==i]
                    #print(value)
                    fun=helperRoot(dataframe,col,key)
                    again.append(fun)
            except Exception as e:
                print(e)
                pass

    again        
arrays
```

    Gamer
    {'Gamer': {'Yes': 'Branch', 'No': 'Branch'}}
    [{'Branch': {'CSE': '@No@', 'ECE': 'CGPA', 'MECH': '@No@'}}, {'Branch': {'CSE': '@Yes@', 'ECE': '@Yes@', 'MECH': 'CGPA'}}]
    CGPA Branch ECE
    CGPA Branch MECH
    




    [{'Gamer': {'Yes': 'Branch', 'No': 'Branch'}},
     {'Branch': {'CSE': '@No@', 'ECE': 'CGPA', 'MECH': '@No@'}},
     {'Branch': {'CSE': '@Yes@', 'ECE': '@Yes@', 'MECH': 'CGPA'}},
     {'CGPA': {'High': '@Yes@'}},
     {'CGPA': {'High': '@No@'}}]



# Creating Dictionary


```python
age=arrays[::-1]


for i in range(len(age)):

    flag=0
    k=list(age[i].keys())[0]
    v=list(age[i].values())[0]
    for j in range(len(age)):
        if flag==1:
            break
        else:
            if j!=i:
                kk=list(age[j].keys())[0]
                vv=list(age[j].values())[0]
                
                
               
                try:
                    if len(set(list(vv.values())))!=len(list(vv.values())):
                        gk=[mj for mj in vv.items()][::-1]
                        vv=dict(gk)
                except:
                    pass
                
                
                if k==kk:
                    pass
                else:
                    for kkk,vvv in vv.items():

                        
                        if vvv==k:
                            
                            age[j][kk][kkk]=age[i]
                            flag=1
                            break
                          
                            

               
            else:
                pass
finalDict=age[-1]
print("Graph is\n",finalDict)
```

    Graph is
     {'Gamer': {'Yes': {'Branch': {'CSE': '@No@', 'ECE': {'CGPA': {'High': '@Yes@'}}, 'MECH': '@No@'}}, 'No': {'Branch': {'CSE': '@Yes@', 'ECE': '@Yes@', 'MECH': {'CGPA': {'High': '@No@'}}}}}}
    

# Pydot Graph


```python
#import os
from matplotlib import pyplot as plt
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydot
from PIL import Image


menu = finalDict
def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, str(k)+'_'+str(v))

graph = pydot.Dot(graph_type='graph')
visit(menu)
graph.write_png('graph.png')

fname = 'graph.png'
image = Image.open(fname)
arr = np.asarray(image)
plt.style.use("bmh")
plt.rcParams.update({"figure.figsize" : (25, 25),
                     "axes.facecolor" : "white",
                     "axes.edgecolor":  "black"})
plt.imshow(arr)
plt.show()
```

![graph.png](attachment:graph.png)

# Checking Model Accuracy


```python
dataSet=pd.read_csv("dataset.csv")

def dfs(i,dictionary=finalDict,dataSet=dataSet):
    try:
        key=list(dictionary.keys())
        ds=dataSet.loc[i,key]
        value=list(dictionary.values())
        return dfs(i,dictionary[key[0]][ds[0]],dataSet)
    except Exception as e:
    
        value=dictionary[key[0]][ds[0]] 
        return value
    
    

output=[]
for i in range(len(dataSet)):
    var=dfs(i,finalDict,dataSet)
    output.append(var)

output
def dicToStr(i):
    try:
        o=i.values()
        return dicToStr(o)
    except:
        return i
    
Foutput=[]
for i in output:
    if type(i)==str:
        Foutput.append(i)
        pass
    else:
        if type(i)==dict:
            aa=dicToStr(i)
            Foutput.append(list(list(aa)[0].values())[0])
loop=0
counter=0
for last in dataSet[[i for i in dataSet][-1]]:
    if last==Foutput[loop][1:-1]:
        counter+=1
    loop+=1
print("#######\n","------- Model Accuracy Is",(counter/len(dataSet))*100,"% -------","\n#######")
```

    #######
     ------- Model Accuracy Is 66.66666666666666 % ------- 
    #######
    


```python

```

# Cart (Recursive Implementation)


```python
import pandas as pd
import numpy as np
from collections import Counter
```


```python
openFile=pd.read_csv("DatasetOne.csv")
#openFile = openFile.drop('S.No', axis = 1)
openFile = openFile.drop('Day', axis = 1)
openFile = openFile.drop('Unnamed: 0', axis = 1)
dec_tree = dict()

colD=[i for i in openFile]
Yname=colD[-1]
```


```python
openFile
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Outlook</th>
      <th>Temp.</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>Decision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overcast</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Overcast</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



# Gini Index


```python
def giniCalculate(df,colD=colD):
    finals={}
    for i in colD[:-1]:
        counter=0
        values=df[i].unique().tolist()
        #print(values,"5678values")
        parentDic=dict(Counter(df[i]))
        totalElement=sum(parentDic.values())
        d={}
        for j in values:
            tempdf=df[df[i]==j]
            dic=dict(Counter(tempdf[Yname]))
           
            totalLength=sum(dic.values())
            index=0
            dicloop=list(dic.values())

            for k in dicloop:


                index+=pow((k/totalLength),2)


            d[j]=index

        for keys,values in parentDic.items():
        
            counter+=d[keys]*(values/totalElement)
        
        finals[i]=1-counter
        
   
    zipper=dict(sorted(finals.items(), key=lambda x: x[1]))

    for k,v in zipper.items():
        return k

    

output=[]

def cart(df):
    
    b=[i for i in df][:]
    
    if b==[]: return 0
        
    node=giniCalculate(df,b)
    #print(node)
    
    
    f = df[node].unique().tolist()
    dec_tree[node] = dict(zip(f, [0 for _ in range(len(f))]))
    for i, j in dec_tree[node].items():
        try:
            dfr = df[df[node] == i]
            dfr = dfr.drop(node, axis = 1)
            if len(dfr[Yname].unique().tolist())==1:
                dec_tree[node][i] =list(dfr[Yname].unique().tolist())[0]


            else:
                    
                node_inter = cart(dfr)
                dec_tree[node][i] = node_inter

        except Exception as e:
            pass
    
    return node

root = cart(openFile)
print(dec_tree, root)

    
```

    {'Outlook': {'Sunny': 'Humidity', 'Overcast': 'Yes', 'Rain': 'Wind'}, 'Humidity': {'High': 'No', 'Normal': 'Yes'}, 'Wind': {'Weak': 'Yes', 'Strong': 'No'}} Outlook
    

# Creating Graph


```python
di={}
for k,v in dec_tree.items():
    try:
        for i,o in v.items():
            if o in dec_tree:
                dec_tree[k][i]={o:dec_tree[o]}
    except:
        pass
#print(dec_tree)
        
for k,v in dec_tree.items():
    di[k]=v
    break
print(di)
```

    {'Outlook': {'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}, 'Overcast': 'Yes', 'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}}}
    

# Plotting Graph


```python
#import os
from matplotlib import pyplot as plt
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydot
from PIL import Image


menu = di
def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, str(k)+'_'+str(v))

graph = pydot.Dot(graph_type='graph')
visit(menu)
graph.write_png('graph.png')

fname = 'graph.png'
image = Image.open(fname)
arr = np.asarray(image)
plt.style.use("bmh")
plt.rcParams.update({"figure.figsize" : (25, 25),
                     "axes.facecolor" : "white",
                     "axes.edgecolor":  "black"})
plt.imshow(arr)
plt.show()
```

![graph.png](attachment:graph.png)

# Cart (Through Sklearn)


```python
import os
import numpy as np
import pandas as pd
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split 
```


```python
data = pd.read_csv('DatasetTwo.csv')
col=[i for i in data][2:]
print(col)
data = data.drop('Unnamed: 0', axis = 1)
data = data.drop('S.No', axis = 1)

data.head()
data.info()
```

    ['Branch', 'CGPA', 'Gamer', 'Movie_Fanatic', 'Committed?']
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15 entries, 0 to 14
    Data columns (total 5 columns):
    Branch           15 non-null object
    CGPA             15 non-null object
    Gamer            15 non-null object
    Movie_Fanatic    15 non-null object
    Committed?       15 non-null object
    dtypes: object(5)
    memory usage: 680.0+ bytes
    


```python
data[col[-1]],class_names = pd.factorize(data[col[-1]])
```


```python
print(class_names)
```

    Index(['No', 'Yes'], dtype='object')
    


```python
for i in col[:-1]:
    data[i],_=pd.factorize(data[i])

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Branch</th>
      <th>CGPA</th>
      <th>Gamer</th>
      <th>Movie_Fanatic</th>
      <th>Committed?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
```


```python
# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
```


```python
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=0, splitter='best')




```python
# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
```

    Misclassified samples: 2
    Accuracy: 0.60
    


```python
import graphviz
feature_names = X.columns

dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=class_names)
graph = graphviz.Source(dot_data)  


graph
```

![Untitled1.png](attachment:Untitled1.png)


```python

```
