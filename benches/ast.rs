#[derive(Debug, Copy, Clone)]
pub enum Expr<'expr> {
    Int(u32),
    Array(&'expr [Self]),
    Call(&'expr Self, &'expr [Self]),
    Binop(Binop, &'expr Self, &'expr Self),
}

#[derive(Debug, Copy, Clone)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

fn main() {}
