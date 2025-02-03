import pandas as pd ;

d =[[1,15],[2,11],[3,11],[4,20]] ; 

df = pd.DataFrame(d)
df.columns = ['student_id', 'age']
print(df)