import dataget as dg

ds = dg.data(
    "udacity-selfdriving-simulator",
    path = "data/raw",
)
ds = ds.get(download = False)
df = ds.df

print(df.head())
print(df.filepath.iloc[0])
