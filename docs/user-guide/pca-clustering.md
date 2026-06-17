# PCA & Clustering

The PCA + Clustering tab combines two-component PCA with KMeans clustering.

## 🧭 Purpose

This tab helps analysts explore structure in selected numeric fields by reducing dimensions and grouping observations in a two-dimensional view.

## 🧱 Requirements

| Requirement | Reason |
|---|---|
| At least two numeric columns | PCA requires multiple numeric inputs. |
| Sufficient complete rows | PCA and KMeans need enough observations after missing-value removal. |
| Selected scaling option | Scaling affects PCA and cluster geometry. |

## ⚙️ Controls

| Control | Description |
|---|---|
| PCA Columns | Numeric columns used as PCA inputs. |
| Scaling | `standard`, `minmax`, or `none`. |
| Clusters `k` | Number of KMeans clusters, from 2 to 10. |

## 📈 Output

Cutey displays:

- PC1 vs PC2 scatter plot.
- KMeans cluster labels.
- Distinct markers and colors by cluster.
- Explained variance ratio for the two PCA components.

## ✅ Interpretation Guidance

- Use standardized scaling when fields have very different units or magnitudes.
- Treat clusters as exploratory groups, not official categories.
- Review source data for each cluster before drawing conclusions.
- PCA components are mathematical combinations of input fields; they are not direct budget categories.
