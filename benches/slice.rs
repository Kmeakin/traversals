use criterion::{black_box, criterion_group, criterion_main, Criterion};
use traversals::fold::Fold;

#[derive(Debug, PartialEq, Eq)]
pub struct IntStats {
    sum: u32,
    product: u32,
    min: Option<u32>,
    max: Option<u32>,
}

fn unfused(ints: &[u32]) -> IntStats {
    let sum = ints.iter().copied().sum();
    let product = ints.iter().copied().product();
    let min = ints.iter().copied().min();
    let max = ints.iter().copied().max();
    IntStats {
        sum,
        product,
        min,
        max,
    }
}

fn fused(ints: &[u32]) -> IntStats {
    let sum = traversals::fold::from_fn(0, |sum, x| sum + x);
    let product = traversals::fold::from_fn(1, |prod, x| prod * x);
    let min = traversals::fold::min();
    let max = traversals::fold::max();

    let (((sum, product), min), max) = sum
        .zip(product)
        .zip(min)
        .zip(max)
        .fold(ints.iter().copied());

    IntStats {
        sum,
        product,
        min,
        max,
    }
}

fn check_equal(ints: &[u32]) {
    let unfused_stats = unfused(ints);
    let fused_stats = fused(ints);
    assert_eq!(unfused_stats, fused_stats);
}

#[test]
fn unit_tests() {
    check_equal(&[]);
    check_equal(&[1, 2, 3, 4]);
}

fn benchmarks(c: &mut Criterion) {
    let xs: Vec<_> = (1..).take(10_000_000).collect();

    c.bench_function("unfused", |b| {
        b.iter(|| unfused(black_box(&xs)));
    });

    c.bench_function("fused", |b| {
        b.iter(|| fused(black_box(&xs)));
    });
}

criterion_group!(slice, benchmarks);
criterion_main!(slice);
