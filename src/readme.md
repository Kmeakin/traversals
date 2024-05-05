Composable, fusable traversals over data structures. Define each traversal
separately, then combine them and run them in parallel in a single pass over a
data structure.

# Motivation
Many programs can be expressed in terms of *traversals* over a data structure.
For example, a compiler consists of many passes over a syntax tree or internal
representation, and a renderer consists of several passes over a scene. In
this example, we will consider traversals over a slice of `u32`s.

For the sake of simplicity, we wish to be able to define each traversal over the
data structure in isolation:
```rust
fn sum_and_product(xs: &[u32]) -> (u32, u32) {
    let sum = xs.iter().fold(0, |acc, x| acc + x);
    let product = xs.iter().fold(1, |acc, x| acc * x);
    (sum, product)
}
```

However, this requires traversing the slice twice: once to compute the sum, and
once to compute the product. This is inefficient.
A solution is to *fuse* the two traversals into one mega-traversal:
```rust
fn sum_and_product(xs: &[u32]) -> (u32, u32) {
    xs.iter().fold((0, 1), |(sum, prod), x| (sum + x, prod * x))
}
```

But this is less readable than the first attempt: two conceptually separate
peices of logic must be combined. Combining more complex traversals, like passes
over a syntax tree, will be even more confusing.


```rust
use traversals::prelude::*;

let sum = traversals::fold::from_fn(0, |acc, x| acc + x);
let product = traversals::fold::from_fn(1, |acc, x| acc * x);
let sum_and_product = sum.zip(product);
let xs = &[1, 2, 3, 4];
assert_eq!(sum_and_product.fold(xs), (10, 24));
```

# Foldable
Every type that implements the standard library's `Iterator` trait is also a `Foldable`:
```rust
# #![feature(try_trait_v2)]
# use traversals::prelude::*;
let producer = [1, 2, 3, 4].iter().into_foldable();
```

but `Foldable` can also be implemented for types that are hard to
implement `Iterator` for, like expression trees:

```rust
#![feature(try_trait_v2)]
use core::ops::Try;

use traversals::prelude::{Foldable, IntoFoldable};

#[derive(Copy, Clone)]
pub enum Expr<'this> {
    Int(u32),
    Add(&'this Self, &'this Self),
    Call(&'this Self, &'this [Self]),
}

impl<'this> IntoFoldable for &'this Expr<'this> {
    type Item = &'this Expr<'this>;
    type IntoFoldable = ExprFolder<'this>;
    fn into_foldable(self) -> Self::IntoFoldable { ExprFolder { expr: self } }
}

#[derive(Copy, Clone)]
pub struct ExprFolder<'this> {
    expr: &'this Expr<'this>,
}

impl<'this> Foldable for ExprFolder<'this> {
    type Item = &'this Expr<'this>;

    fn try_fold<A, R, F>(&mut self, acc: A, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(A, Self::Item) -> R,
        R: Try<Output = A>,
    {
        let expr = self.expr;
        match expr {
            Expr::Int(..) => f(acc, expr),
            Expr::Add(lhs, rhs) => [expr, lhs, rhs]
                .iter()
                .try_fold(acc, |acc, expr| f(acc, expr)),
            Expr::Call(fun, args) => {
                let acc = f(acc, expr)?;
                let acc = f(acc, fun)?;
                args.iter().try_fold(acc, f)
            }
        }
    }
}

```

## Internal vs external iteration
TODO

# Consumers
Traversals over a datastructure are known as **folds** and implement the
`Fold` trait.

# Credits
- https://docs.rs/internal-iterator/latest/internal_iterator/
- https://hackage.haskell.org/package/foldl
