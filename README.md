# Exercise 1

We want to find the solution to the elliptic boundary value problem defined by the differential equation:

$$ u_{xx} + u_{yy} = -10(x^2 + y^2 + 5) $$

with boundary conditions $u(0, y) = u(1, y) = u(x, 0) = 0$ and $u(x, 1) = 1$.

Initially, we use the Liebmann method on an orthonormal grid with limits $(0, 1) \times (0, 1)$, which we discretize into $N \times M = 400 \times 400$ points.

In essence, the problem we are studying is the Poisson equation in two dimensions:

$$ \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = S(x, y) $$

The Liebmann method states that during iteration $n + 1$, the value of the function at point $(i, j)$ depends on the values of the function at points $(i + 1, j)$, $(i - 1, j)$, $(i, j + 1)$, $(i, j - 1)$, and the value of $S$ at that point, as calculated during the previous iteration $n$:

$$ f_{i,j}^{n+1} = \frac{1}{4} \left[ f_{i+1,j}^n + f_{i-1,j}^n + f_{i,j+1}^n + f_{i,j-1}^n - h^2 S_{i,j} \right] $$

where $h$ is the distance between points in the $i$ direction, $h = \frac{x_{\text{max}} - x_{\text{min}}}{N - 1}$. Similarly, we define step $k$ in the $j$ direction.

The code we implemented creates two vectors where the values of successive iterations are stored. Initially, using an iterative process, we assign the boundary values as defined by the problem, while each intermediate value is set to zero.

Subsequently, with a second iterative process, as long as the error is greater than an acceptable accuracy, we calculate the solution to the Poisson equation at each point of the grid. The error in each iteration is defined as the average sum of the absolute change from one iteration to the next for all grid points:

$$ \text{tolerance} = \frac{1}{(N - 2)(M - 2)} \sum_{i,j} \left| u_{i,j}^{\text{new}} - u_{i,j}^{\text{old}} \right| $$

Our goal is an accuracy of the order of $10^{-7}$.

The program was parallelized with OpenMP directives and ran on 1, 2, 4, and 8 cores. Below is the results table for each number of threads, showing the number of iterations required, the value of the central point, and the time it took to run the program.

| #threads | #iterations | (N/2, M/2) | elapsed time (sec) |
|----------|-------------|------------|--------------------|
| 1        | 207,846     | 4.329148   | 93.884             |
| 2        | 207,846     | 4.329148   | 56.207             |
| 4        | 207,846     | 4.329148   | 39.680             |
| 8        | 207,846     | 4.329148   | 51.572             |

We also calculated the parallel speedup and efficiency for each execution and created the corresponding diagrams concerning the number of threads.

| #threads | speedup | efficiency (%) |
|----------|---------|----------------|
| 1        | 1       | 100            |
| 2        | 1.67    | 83.5           |
| 4        | 2.37    | 59.3           |
| 8        | 1.82    | 22.8           |

Finally, we plotted the solution $u(x, y)$ as a surface.

# Exercise 2

In the second part of the work, we solved the differential equation using the Gauss-Seidel method, which speeds up the process since it uses, where available, the values of points that have already been calculated for the current iteration. The problem that arises during parallel processing is that the task division among the threads is done 'linearly,' meaning each core is assigned a set of lines $i$ of the grid. This results in the values of the current iteration at points that a core needs not being calculated by the core assigned to compute them.

For this reason, we use the red-black algorithm, during which the central points of the sub-grids of size 5 elements (cross shape) corresponding to indices $(i, j)$ with an odd sum are calculated in parallel, and then, with a second parallel iterative process, the remaining points are calculated using the $u_{\text{New}}$ vector created in the previous step. The algorithm is analyzed into two steps:

The algorithm can be split into two steps:

For \( i + j \% 2 == 1 \):

$$ 
f_{i,j}^{n+1} = (1 - \omega) f_{i,j}^n + \frac{\omega}{4} \left( f_{i+1,j}^n + f_{i-1,j}^n + f_{i,j+1}^n + f_{i,j-1}^n - h^2 S_{i,j} \right) 
$$

For \( i + j \% 2 == 0 \):

$$ 
f_{i,j}^{n+1} = (1 - \omega) f_{i,j}^n + \frac{\omega}{4} \left( f_{i+1,j}^{n+1} + f_{i-1,j}^{n+1} + f_{i,j+1}^{n+1} + f_{i,j-1}^{n+1} - h^2 S_{i,j} \right) 
$$

where $\omega$ is the acceleration factor of the so-called Successive Over-Relaxation (SOR) method based on Gauss-Seidel.

We tested the method for values of the acceleration factor $\omega = 1$, $1.95$, and $1.99$ for 1, 2, 4, and 8 cores. Below are the cumulative results compared with the Liebmann method from the first part of the work.

| method   | #threads | #iterations | elapsed time (sec) | parallel speedup |
|----------|----------|-------------|-------------------|------------------|
| Liebmann | 1        | 207,846     | 93.884            | 1                |
| $\omega = 1$  | 121,645     | 98.303            | 1                |
| $\omega = 1.95$ | 4,543       | 3.698             | 1                |
| $\omega = 1.99$ | 1,318       | 1.079             | 1                |
| Liebmann | 2        | 207,846     | 56.207            | 1.67             |
| $\omega = 1$  | 121,645     | 55.900            | 1.76             |
| $\omega = 1.95$ | 4,543       | 2.108             | 1.75             |
| $\omega = 1.99$ | 1,318       | 0.609             | 1.77             |
| Liebmann | 4        | 207,846     | 39.680            | 2.37             |
| $\omega = 1$  | 121,645     | 38.122            | 2.58             |
| $\omega = 1.95$ | 4,543       | 1.338             | 2.76             |
| $\omega = 1.99$ | 1,318       | 0.391             | 2.76             |
| Liebmann | 8        | 207,846     | 51.572            | 1.82             |
| $\omega = 1$  | 121,645     | 45.696            | 2.15             |
| $\omega = 1.95$ | 4,543       | 1.814             | 2.04             |
| $\omega = 1.99$ | 1,318       | 0.543             | 1.99             |

We created a log-log plot of the error of each iteration versus the number of iterations for both methods (Liebmann, SOR) and for the three acceleration factor values.

The SOR method for acceleration factor \( $\omega$ = 1 \) coincides with the Gauss-Seidel method for \( i + j \% 2 == 1 \) and requires about half the number of iterations compared to the Liebmann method (207,000) to provide a solution. This makes sense because each iteration performs two steps. The execution time for each number of cores is approximately the same for both methods.

For \( $\omega$ = 1.95 \), the iterations of the SOR method drop from about 121,500 to 4,500, and the execution time drops from 98 seconds to 3.5 seconds.

For \( $\omega$ = 1.99 \), the iterations drop to 1,300, and the execution time drops to 1 second.

We observe that the number of iterations is not affected by the number of cores to which the calculations are distributed. For 8 cores, the execution time increases instead of decreasing, which is logical since the physical cores of the processor running the program are 4.
