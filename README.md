# NN model
'a' is the embedding space, and 'b' is the input data.
```go
loss := Avg(Quadratic(Mul(Dropout(Square(set.Get("a")), dropout), 
		Euclidean(set.Get("b"), set.Get("b"))),
	Euclidean(set.Get("b"), set.Get("b"))))
```

# Results for clustering the breast cancer wisconsin diagnostic
```
Sort with nn and cluster with Meta KMeans
B [354 3]
M [69 143]

Sort with nn and cluster Page Rank
B [208 149]
M [5 207]

Direct Meta KMeans
B [356 1]
M [80 132]

Direct Page Rank
B [94 263]
M [203 9]
```
