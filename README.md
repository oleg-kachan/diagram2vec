# diagram2vec

Diagram2vec provides the following finite-dimensional vector representations of the persistence diagrams:

- persistence statistics (statistics of quanitites such as persistence, midlife, mult.life)
- Euler characteristic curve
- k-th Betti number curves
- persistence curves (curves of quanitites such as persistence, midlife, multitplicative life)
- entropy curves (curves of entropy of quanitites such as persistence, midlife, multitplicative life)

All representations have &epsilon;-robust versions, i.e. not taking into account intervals with persistence below a certain threshold.

## Installation

### Dependencies

- python (>= 3.6)
- numpy
- scipy

The latest version of the diagram2vec can be installed with pip:

```
pip install diagram2vec
```

## Example of usage

```
import diagram2vec

# list of lists of ndarrays, representing a collection of persistence diagrams
diagrams = [
    [np.array([[0.0, np.inf], [0.0, 0.4], [0.0, 0.5]]), np.array([[0.1, 0.6], [0.2, 0.4]])], 
    [np.array([[0.0, np.inf], [0.0, 0.1]]), np.array([[0.4, 0.7]])],
]

# compute a list of statistics of a distribution of the persistence p := (d - b) quantity
stats = diagram2vec.statistics(diagrams, quantity="persistence", statistics=["min", "mean", "count"])

# compute an Euler characteristic curve (a vector of dimension 10)
euler_curve = diagram2vec.euler_curve(diagrams, m=10)

# compute a Betti number curve, which is default quantity for the `persistence_curve`
# (a vector of dimension 50, intervals with persistence below 0.05 are not used)
betti_curve = diagram2dec.persistence_curve(diagrams, m=50, f="linear", b=0.05)

# compute a curve of persistence quantity
# available quanities: 'betti', 'persistence', 'midlife', 'multlife'
persistence_curve = diagram2dec.persistence_curve(diagrams, quantity="persistence", m=50)

# compute a curve of midlife entropy
# available quanities: 'persistence', 'midlife', 'multlife'
entropy_curve = diagram2dec.entropy_curve(diagrams, quantity="midlife", m=20)
```
