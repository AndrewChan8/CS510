# Assignment 1 - Andrew Chan 
## Problem 1: Document Similarity  

Documents:
- D1 = `[woof woof meow]`  
- D2 = `[woof woof squeak]`  

(a) Cosine similarity using TF weighting  

Vocabulary:  
```
["woof", "meow", "squeak"]
```

Term Frequency Vectors:
- D1 = `[2, 1, 0]`  → woof=2, meow=1, squeak=0  
- D2 = `[2, 0, 1]`  → woof=2, meow=0, squeak=1  

Dot Product:  
```
(2 * 2) + (1 * 0) + (0 * 1) = 4
```

Norms:  
```
||D1|| = sqrt(2^2 + 1^2 + 0^2) = sqrt(5)  
||D2|| = sqrt(2^2 + 0^2 + 1^2) = sqrt(5)
```

Cosine Similarity:  
```
cosine = 4 / (sqrt(5) * sqrt(5)) = 4 / 5 = 0.8
```

Answer (a): `0.8`

---

### (b) Cosine similarity using TF-IDF weighting (2 documents)

IDF values with 2 documents (N=2):  
- `idf(woof) = log(2/2) = 0`
- `idf(meow) = log(2/1) ≈ 0.6931`
- `idf(squeak) = log(2/1) ≈ 0.6931`

TF-IDF Vectors:  
- D1 = `[0, 0.6931, 0]`
- D2 = `[0, 0, 0.6931]`

Dot Product:  
```
(0 * 0) + (0.6931 * 0) + (0 * 0.6931) = 0
```

Norms:  
```
||D1|| = 0.6931  
||D2|| = 0.6931
```

Cosine Similarity:  
```
cosine = 0 / (0.6931 * 0.6931) = 0
```

✅ **Answer (b):** `0`

---

### (c) Adding third document D3 = [meow squeak]

New document collection:  
- **D1** = `[woof woof meow]`  
- **D2** = `[woof woof squeak]`  
- **D3** = `[meow squeak]`  
Total documents (N = 3)

**IDF values with 3 documents:**
- `idf(woof) = log(3/2) ≈ 0.4055`
- `idf(meow) = log(3/2) ≈ 0.4055`
- `idf(squeak) = log(3/2) ≈ 0.4055`

**TF-IDF Vectors:**
- **D1** = `[2*0.4055, 1*0.4055, 0]` = `[0.811, 0.4055, 0]`
- **D2** = `[2*0.4055, 0, 1*0.4055]` = `[0.811, 0, 0.4055]`

**Dot Product:**  
```
(0.811 * 0.811) + (0.4055 * 0) + (0 * 0.4055) = 0.6577
```

**Norms:**  
```
||D1|| = sqrt(0.811^2 + 0.4055^2) ≈ 0.9051  
||D2|| = sqrt(0.811^2 + 0.4055^2) ≈ 0.9051
```

**Cosine Similarity:**  
```
cosine = 0.6577 / (0.9051 * 0.9051) ≈ 0.8
```

✅ **Answer (c):** `0.8`  
_Adding D3 caused all terms to appear in 2/3 documents, making the IDF values equal. Hence, cosine similarity reverted to the TF case._
