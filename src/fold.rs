use core::convert::Infallible;
use core::marker::PhantomData;
use core::ops::ControlFlow;

use crate::foldable::{Foldable, IntoFoldable};

pub trait Fold {
    type Item;
    type Continue;
    type Break;
    type Output;

    fn start(&mut self) -> Self::Continue;

    fn step(
        &mut self,
        acc: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue>;

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output;

    fn fold(mut self, producer: impl IntoFoldable<Item = Self::Item>) -> Self::Output
    where
        Self: Sized,
    {
        let init = self.start();
        let state = producer
            .into_foldable()
            .try_fold(init, |acc, item| self.step(acc, item));
        self.finish(state)
    }

    /// Zip up two folds into a single fold. This is the magic that allows folds
    /// defined separately to be fused into a single traversal.
    /// # Examples
    /// ```
    /// # use traversals::fold::Fold;
    /// let sum = traversals::fold::from_fn(0, |sum, x| sum + x);
    /// let product = traversals::fold::from_fn(1, |prod, x| prod * x);
    /// assert_eq!(sum.zip(product).fold([1, 2, 3, 4]), (10, 24));
    /// ```
    fn zip<Other>(self, other: Other) -> Zip<Self, Other>
    where
        Self: Sized,
        Other: Fold,
    {
        Zip {
            f1: self,
            f2: other,
        }
    }
}

pub const fn from_fn<Item, Continue, StepFn>(
    init: Continue,
    step_fn: StepFn,
) -> FromFn<Item, Continue, StepFn> {
    FromFn {
        item: PhantomData,
        init: Some(init),
        step_fn,
    }
}

pub struct FromFn<Item, Continue, StepFn> {
    item: PhantomData<Item>,
    init: Option<Continue>,
    step_fn: StepFn,
}

impl<Item, Output, StepFn> Fold for FromFn<Item, Output, StepFn>
where
    StepFn: FnMut(Output, Item) -> Output,
{
    type Item = Item;
    type Break = Infallible;
    type Continue = Output;
    type Output = Output;

    fn start(&mut self) -> Self::Continue { self.init.take().unwrap() }

    fn step(
        &mut self,
        acc: Self::Continue,
        item: Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        ControlFlow::Continue((self.step_fn)(acc, item))
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Output {
        match state {
            ControlFlow::Continue(c) => c,
            ControlFlow::Break(b) => match b {},
        }
    }
}

pub struct Zip<Fold1, Fold2>
where
    Fold1: Fold,
    Fold2: Fold,
{
    f1: Fold1,
    f2: Fold2,
}

pub enum ZipContinue<C1, B1, C2, B2> {
    Both(C1, C2),
    First(C1, B2),
    Second(B1, C2),
}

impl<F1, F2> Fold for Zip<F1, F2>
where
    F1: Fold,
    F1::Item: Clone,
    F2: Fold<Item = F1::Item>,
{
    type Item = F1::Item;
    type Break = (F1::Break, F2::Break);
    type Continue = ZipContinue<F1::Continue, F1::Break, F2::Continue, F2::Break>;
    type Output = (F1::Output, F2::Output);

    fn start(&mut self) -> Self::Continue {
        let c1 = self.f1.start();
        let c2 = self.f2.start();
        ZipContinue::Both(c1, c2)
    }

    fn step(
        &mut self,
        acc: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match acc {
            ZipContinue::Both(c1, c2) => {
                let s1 = self.f1.step(c1, item.clone());
                let s2 = self.f2.step(c2, item);
                match (s1, s2) {
                    (ControlFlow::Continue(c1), ControlFlow::Continue(c2)) => {
                        ControlFlow::Continue(ZipContinue::Both(c1, c2))
                    }
                    (ControlFlow::Continue(c1), ControlFlow::Break(b2)) => {
                        ControlFlow::Continue(ZipContinue::First(c1, b2))
                    }
                    (ControlFlow::Break(b1), ControlFlow::Continue(c2)) => {
                        ControlFlow::Continue(ZipContinue::Second(b1, c2))
                    }
                    (ControlFlow::Break(b1), ControlFlow::Break(b2)) => {
                        ControlFlow::Break((b1, b2))
                    }
                }
            }
            ZipContinue::First(c1, b2) => {
                let s1 = self.f1.step(c1, item);
                match s1 {
                    ControlFlow::Continue(c1) => ControlFlow::Continue(ZipContinue::First(c1, b2)),
                    ControlFlow::Break(b1) => ControlFlow::Break((b1, b2)),
                }
            }
            ZipContinue::Second(b1, c2) => {
                let s2 = self.f2.step(c2, item);
                match s2 {
                    ControlFlow::Continue(c2) => ControlFlow::Continue(ZipContinue::Second(b1, c2)),
                    ControlFlow::Break(b2) => ControlFlow::Break((b1, b2)),
                }
            }
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(c) => match c {
                ZipContinue::Both(c1, c2) => (
                    self.f1.finish(ControlFlow::Continue(c1)),
                    self.f2.finish(ControlFlow::Continue(c2)),
                ),
                ZipContinue::First(c1, b2) => (
                    self.f1.finish(ControlFlow::Continue(c1)),
                    self.f2.finish(ControlFlow::Break(b2)),
                ),
                ZipContinue::Second(b1, c2) => (
                    self.f1.finish(ControlFlow::Break(b1)),
                    self.f2.finish(ControlFlow::Continue(c2)),
                ),
            },
            ControlFlow::Break((b1, b2)) => (
                self.f1.finish(ControlFlow::Break(b1)),
                self.f2.finish(ControlFlow::Break(b2)),
            ),
        }
    }
}

pub struct Map<Fold1, FinishFn> {
    fold1: Fold1,
    finish_fn: FinishFn,
}

impl<Fold1, FinishFn, Output2> Fold for Map<Fold1, FinishFn>
where
    Fold1: Fold,
    FinishFn: FnMut(Fold1::Output) -> Output2,
{
    type Item = Fold1::Item;
    type Continue = Fold1::Continue;
    type Break = Fold1::Break;
    type Output = Output2;

    fn start(&mut self) -> Self::Continue { self.fold1.start() }

    fn step(
        &mut self,
        acc: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        self.fold1.step(acc, item)
    }

    fn finish(mut self, state: ControlFlow<Self::Break, Self::Continue>) -> Output2 {
        let output1 = self.fold1.finish(state);
        (self.finish_fn)(output1)
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::first;
/// assert_eq!(first().fold([1, 2, 3]), Some(1));
/// assert_eq!(first::<u32>().fold([]), None);
/// ```
pub const fn first<Item>() -> First<Item> { First { item: PhantomData } }

pub struct First<Item> {
    item: PhantomData<Item>,
}

impl<Item> Fold for First<Item> {
    type Item = Item;
    type Continue = ();
    type Break = Item;
    type Output = Option<Item>;

    fn start(&mut self) -> Self::Continue {}

    fn step(
        &mut self,
        (): Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        ControlFlow::Break(item)
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(()) => None,
            ControlFlow::Break(item) => Some(item),
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::last;
/// assert_eq!(last().fold([1, 2, 3]), Some(3));
/// assert_eq!(last::<u32>().fold([]), None);
/// ```
pub const fn last<Item>() -> Last<Item> { Last { item: PhantomData } }

pub struct Last<Item> {
    item: PhantomData<Item>,
}

impl<Item> Fold for Last<Item> {
    type Item = Item;
    type Continue = Option<Item>;
    type Break = Infallible;
    type Output = Option<Item>;

    fn start(&mut self) -> Self::Continue { None }

    fn step(
        &mut self,
        _: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        ControlFlow::Continue(Some(item))
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(item) => item,
            ControlFlow::Break(b) => match b {},
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::all;
/// assert_eq!(all(|x: &u32| *x % 2 == 0).fold([2, 4, 6]), true);
/// assert_eq!(all(|x: &u32| *x % 2 == 0).fold([2, 4, 7]), false);
/// assert_eq!(all(|x: &u32| *x % 2 == 0).fold([]), true);
/// ```
pub const fn all<Item, PredFn>(pred_fn: PredFn) -> All<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    All {
        item: PhantomData,
        pred_fn,
    }
}

pub struct All<Item, PredFn> {
    item: PhantomData<Item>,
    pred_fn: PredFn,
}

impl<Item, PredFn> Fold for All<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    type Item = Item;
    type Continue = ();
    type Break = ();
    type Output = bool;

    fn start(&mut self) -> Self::Continue {}

    fn step(&mut self, (): Self::Continue, item: Item) -> ControlFlow<Self::Break, Self::Continue> {
        match (self.pred_fn)(&item) {
            true => ControlFlow::Continue(()),
            false => ControlFlow::Break(()),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> bool {
        match state {
            ControlFlow::Continue(()) => true,
            ControlFlow::Break(()) => false,
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::any;
/// assert_eq!(any(|x: &u32| *x % 2 == 0).fold([2, 4, 6]), true);
/// assert_eq!(any(|x: &u32| *x % 2 == 0).fold([3, 5, 7]), false);
/// assert_eq!(any(|x: &u32| *x % 2 == 0).fold([]), false);
/// ```
pub const fn any<Item, PredFn>(pred_fn: PredFn) -> Any<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    Any {
        item: PhantomData,
        pred_fn,
    }
}

pub struct Any<Item, PredFn> {
    item: PhantomData<Item>,
    pred_fn: PredFn,
}

impl<Item, PredFn> Fold for Any<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    type Item = Item;
    type Continue = ();
    type Break = ();
    type Output = bool;

    fn start(&mut self) -> Self::Continue {}

    fn step(&mut self, (): Self::Continue, item: Item) -> ControlFlow<Self::Break, Self::Continue> {
        match (self.pred_fn)(&item) {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> bool {
        match state {
            ControlFlow::Continue(()) => false,
            ControlFlow::Break(()) => true,
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::count;
/// assert_eq!(count().fold([1, 2, 3]), 3);
/// assert_eq!(count::<u32>().fold([]), 0);
/// ```
pub const fn count<Item>() -> Count<Item> { Count { item: PhantomData } }

pub struct Count<Item> {
    item: PhantomData<Item>,
}

impl<Item> Fold for Count<Item> {
    type Item = Item;
    type Continue = usize;
    type Break = Infallible;
    type Output = usize;

    fn start(&mut self) -> Self::Continue { 0 }

    fn step(
        &mut self,
        acc: Self::Continue,
        _: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        ControlFlow::Continue(acc + 1)
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(count) => count,
            ControlFlow::Break(b) => match b {},
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::find;
/// assert_eq!(find(|x: &u32| *x % 2 == 0).fold([1u32, 2, 3]), Some(2));
/// assert_eq!(find(|x: &u32| *x % 2 == 0).fold([1, 3]), None);
/// assert_eq!(find(|x: &u32| *x % 2 == 0).fold([]), None);
/// ```
pub const fn find<Item, PredFn>(pred_fn: PredFn) -> Find<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    Find {
        item: PhantomData,
        pred_fn,
    }
}

pub struct Find<Item, PredFn> {
    item: PhantomData<Item>,
    pred_fn: PredFn,
}

impl<Item, PredFn> Fold for Find<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    type Item = Item;
    type Continue = ();
    type Break = Item;
    type Output = Option<Item>;

    fn start(&mut self) -> Self::Continue {}

    fn step(
        &mut self,
        (): Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match (self.pred_fn)(&item) {
            true => ControlFlow::Break(item),
            false => ControlFlow::Continue(()),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(()) => None,
            ControlFlow::Break(item) => Some(item),
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::find_map;
/// assert_eq!(
///     find_map(|s: &str| s.parse().ok()).fold(["lol", "NaN", "2", "5"]),
///     Some(2)
/// );
/// ```
pub const fn find_map<Item, Output, FilterFn>(filter_fn: FilterFn) -> FindMap<Item, FilterFn>
where
    FilterFn: FnMut(Item) -> Option<Output>,
{
    FindMap {
        item: PhantomData,
        filter_fn,
    }
}

pub struct FindMap<Item, FilterFn> {
    item: PhantomData<Item>,
    filter_fn: FilterFn,
}

impl<Item, Output, FilterFn> Fold for FindMap<Item, FilterFn>
where
    FilterFn: FnMut(Item) -> Option<Output>,
{
    type Item = Item;
    type Continue = ();
    type Break = Output;
    type Output = Option<Output>;

    fn start(&mut self) -> Self::Continue {}

    fn step(
        &mut self,
        (): Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match (self.filter_fn)(item) {
            Some(item) => ControlFlow::Break(item),
            None => ControlFlow::Continue(()),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(()) => None,
            ControlFlow::Break(item) => Some(item),
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::is_sorted;
/// assert_eq!(is_sorted().fold([1, 2, 3]), true);
/// assert_eq!(is_sorted().fold([1, 2, 3, 0]), false);
/// assert_eq!(is_sorted::<u32>().fold([]), true);
/// ```
pub const fn is_sorted<Item>() -> IsSorted<Item>
where
    Item: PartialOrd,
{
    IsSorted { item: PhantomData }
}

pub struct IsSorted<Item> {
    item: PhantomData<Item>,
}

impl<Item> Fold for IsSorted<Item>
where
    Item: PartialOrd,
{
    type Item = Item;
    type Continue = Option<Item>;
    type Break = ();
    type Output = bool;

    fn start(&mut self) -> Self::Continue { None }

    fn step(
        &mut self,
        acc: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match acc {
            None => ControlFlow::Continue(Some(item)),
            Some(prev) => match prev <= item {
                true => ControlFlow::Continue(Some(item)),
                false => ControlFlow::Break(()),
            },
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(_) => true,
            ControlFlow::Break(()) => false,
        }
    }
}

pub const fn is_sorted_by() { todo!() }
pub const fn is_sorted_by_key() { todo!() }

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::max;
/// assert_eq!(max().fold([3, 1, 2]), Some(3));
/// assert_eq!(max().fold([3, 4, 2]), Some(4));
/// assert_eq!(max::<u32>().fold([]), None);
/// ```
pub const fn max<Item>() -> Max<Item>
where
    Item: Ord,
{
    Max { item: PhantomData }
}

pub struct Max<Item> {
    item: PhantomData<Item>,
}

impl<Item> Fold for Max<Item>
where
    Item: Ord,
{
    type Item = Item;
    type Continue = Option<Item>;
    type Break = Infallible;
    type Output = Option<Item>;

    fn start(&mut self) -> Self::Continue { None }

    fn step(
        &mut self,
        max: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match max {
            None => ControlFlow::Continue(Some(item)),
            Some(max) => ControlFlow::Continue(Some(core::cmp::max(max, item))),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(max) => max,
            ControlFlow::Break(b) => match b {},
        }
    }
}

pub const fn max_by() {}
pub const fn max_by_key() {}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::min;
/// assert_eq!(min().fold([3, 1, 2]), Some(1));
/// assert_eq!(min().fold([3, 4, 2]), Some(2));
/// assert_eq!(min::<u32>().fold([]), None);
/// ```
pub const fn min<Item>() -> Min<Item>
where
    Item: Ord,
{
    Min { item: PhantomData }
}

pub struct Min<Item> {
    item: PhantomData<Item>,
}

impl<Item> Fold for Min<Item>
where
    Item: Ord,
{
    type Item = Item;
    type Continue = Option<Item>;
    type Break = Infallible;
    type Output = Option<Item>;

    fn start(&mut self) -> Self::Continue { None }

    fn step(
        &mut self,
        max: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match max {
            None => ControlFlow::Continue(Some(item)),
            Some(min) => ControlFlow::Continue(Some(core::cmp::min(min, item))),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(min) => min,
            ControlFlow::Break(b) => match b {},
        }
    }
}

pub const fn min_by() { todo!() }
pub const fn min_by_key() { todo!() }

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::nth;
/// assert_eq!(nth(0).fold([1, 2, 3]), Some(1));
/// assert_eq!(nth(1).fold([1, 2, 3]), Some(2));
/// assert_eq!(nth::<u32>(0).fold([]), None);
/// ```
pub const fn nth<Item>(n: usize) -> Nth<Item> {
    Nth {
        item: PhantomData,
        n,
    }
}

pub struct Nth<Item> {
    item: PhantomData<Item>,
    n: usize,
}

impl<Item> Fold for Nth<Item> {
    type Item = Item;
    type Continue = usize;
    type Break = Item;
    type Output = Option<Item>;

    fn start(&mut self) -> Self::Continue { self.n }

    fn step(
        &mut self,
        acc: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match acc {
            0 => ControlFlow::Break(item),
            _ => ControlFlow::Continue(acc - 1),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(_) => None,
            ControlFlow::Break(item) => Some(item),
        }
    }
}

/// # Examples
/// ```
/// # use traversals::fold::Fold;
/// # use traversals::fold::position;
/// assert_eq!(position(|x: &u32| *x % 2 == 0).fold([1, 2, 3]), Some(1));
/// assert_eq!(position(|x: &u32| *x % 2 == 0).fold([1, 3]), None);
/// assert_eq!(position(|x: &u32| *x % 2 == 0).fold([]), None);
/// ```
pub const fn position<Item, PredFn>(pred_fn: PredFn) -> Position<Item, PredFn> {
    Position {
        item: PhantomData,
        pred_fn,
    }
}

pub struct Position<Item, PredFn> {
    item: PhantomData<Item>,
    pred_fn: PredFn,
}

impl<Item, PredFn> Fold for Position<Item, PredFn>
where
    PredFn: FnMut(&Item) -> bool,
{
    type Item = Item;
    type Continue = usize;
    type Break = usize;
    type Output = Option<usize>;

    fn start(&mut self) -> Self::Continue { 0 }

    fn step(
        &mut self,
        pos: Self::Continue,
        item: Self::Item,
    ) -> ControlFlow<Self::Break, Self::Continue> {
        match (self.pred_fn)(&item) {
            true => ControlFlow::Break(pos),
            false => ControlFlow::Continue(pos + 1),
        }
    }

    fn finish(self, state: ControlFlow<Self::Break, Self::Continue>) -> Self::Output {
        match state {
            ControlFlow::Continue(_) => None,
            ControlFlow::Break(pos) => Some(pos),
        }
    }
}

pub const fn collect() { todo!() }
