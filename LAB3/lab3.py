import pandas as pd
import numpy as np
import statistics
data=pd.read_excel('lab3\\Lab Session1 Data.xlsx')
df=pd.DataFrame(data)
#data = pd.read_csv("nba.csv", index_col ="Name" )
A=df[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]].values
print("The matrix A is ")
print(A,end="\n\n")
print("The matrix C is ")
C=df[["Payment (Rs)"]].values
print(C)
print("Dimensionality of A is ",np.shape(A)[1])
print("No of vectors in the vector space are : ",np.shape(A)[0])
print("The rank of A is : ",np.linalg.matrix_rank(A))
print("The pseudo-inverse of A is : \n",np.linalg.pinv(A))

data_new=pd.read_excel('lab3\\Lab Session1 Data.xlsx',sheet_name=1)
df_new=pd.DataFrame(data_new)
stats=df_new[["Price"]].values
print(stats)
print("The mean is : ",np.mean(stats))
print("The variance is : ",np.var(stats))