use core::convert::Infallible;
use core::ops::Try;

pub trait Foldable {
    type Item;

    fn try_fold<A, R, F>(&mut self, init: A, f: F) -> R
    where
        Self: Sized,
        F: FnMut(A, Self::Item) -> R,
        R: Try<Output = A>;

    fn fold<A, F>(mut self, init: A, mut f: F) -> A
    where
        Self: Sized,
        F: FnMut(A, Self::Item) -> A,
    {
        match self.try_fold(init, |acc, item| Result::<_, Infallible>::Ok(f(acc, item))) {
            Ok(c) => c,
            Err(b) => match b {},
        }
    }

    fn try_for_each<R, F>(&mut self, mut f: F) -> R
    where
        Self: Sized,
        R: Try<Output = ()>,
        F: FnMut(Self::Item) -> R,
    {
        self.try_fold((), |(), item| f(item))
    }

    fn for_each<R, F>(mut self, mut f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        let f = |(), item| {
            f(item);
            Result::<_, Infallible>::Ok(())
        };
        match self.try_fold((), f) {
            Ok(()) => (),
            Err(b) => match b {},
        }
    }

    fn chain<Other>(self, other: Other) -> Chain<Self, Other>
    where
        Self: Sized,
        Other: Foldable<Item = Self::Item>,
    {
        Chain {
            f1: self,
            f2: other,
        }
    }
}

pub struct Chain<F1, F2> {
    f1: F1,
    f2: F2,
}

impl<F1, F2> Foldable for Chain<F1, F2>
where
    F1: Foldable,
    F2: Foldable<Item = F1::Item>,
{
    type Item = F1::Item;

    fn try_fold<A, R, F>(&mut self, init: A, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(A, Self::Item) -> R,
        R: Try<Output = A>,
    {
        let acc = self.f1.try_fold(init, &mut f)?;
        self.f2.try_fold(acc, f)
    }
}

pub trait IntoFoldable {
    type Item;
    type IntoFoldable: Foldable<Item = Self::Item>;
    fn into_foldable(self) -> Self::IntoFoldable;
}

impl<I> IntoFoldable for I
where
    I: IntoIterator,
{
    type Item = I::Item;
    type IntoFoldable = IteratorFoldable<I::IntoIter>;
    fn into_foldable(self) -> Self::IntoFoldable { IteratorFoldable(self.into_iter()) }
}

pub struct IteratorFoldable<I>(I);

impl<I> Foldable for IteratorFoldable<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn try_fold<A, R, F>(&mut self, init: A, f: F) -> R
    where
        Self: Sized,
        F: FnMut(A, Self::Item) -> R,
        R: Try<Output = A>,
    {
        Iterator::try_fold(&mut self.0, init, f)
    }

    fn fold<A, F>(self, init: A, f: F) -> A
    where
        Self: Sized,
        F: FnMut(A, Self::Item) -> A,
    {
        Iterator::fold(self.0, init, f)
    }
}
