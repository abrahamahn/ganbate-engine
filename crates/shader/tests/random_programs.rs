//! Differential validity fuzzing: generate random well-typed shaders in the supported subset,
//! compile them, and require `spirv-val --target-env vulkan1.0` to accept every module.
//! `GANBATE_SHADER_RANDOM_PROGRAMS` / `GANBATE_SHADER_RANDOM_SEED` control the campaign size and seed.

use std::fmt::Write;
use std::process::Command;

use ganbate_shader::{Stage, compile, compile_msl};

#[path = "msl/check.rs"]
mod msl_check;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum S {
    F,
    I,
    U,
    B,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum T {
    Scalar(S),
    Vec(u8, S),
    Mat(u8, u8),
    /// `Item { a: vec3f, b: u32, c: array<f32, 3>, m: mat2x2f }`
    Item,
    /// `array<f32, 3>`
    Arr,
}

impl T {
    fn wgsl(self) -> String {
        match self {
            T::Scalar(s) => sname(s).to_string(),
            T::Vec(n, s) => format!("vec{n}<{}>", sname(s)),
            T::Mat(c, r) => format!("mat{c}x{r}<f32>"),
            T::Item => "Item".into(),
            T::Arr => "array<f32, 3>".into(),
        }
    }
    fn scalar(self) -> S {
        match self {
            T::Scalar(s) | T::Vec(_, s) => s,
            T::Mat(..) | T::Item | T::Arr => S::F,
        }
    }
}

fn sname(s: S) -> &'static str {
    match s {
        S::F => "f32",
        S::I => "i32",
        S::U => "u32",
        S::B => "bool",
    }
}

#[derive(Clone)]
struct Var {
    name: String,
    ty: T,
    mutable: bool,
}

struct Gen {
    rng: Rng,
    scopes: Vec<Vec<Var>>,
    counter: usize,
    /// (name, param types, return type) of helpers defined so far.
    helpers: Vec<(String, Vec<T>, T)>,
    in_loop: usize,
    /// Inside a loop body whose declarations `continuing` may read: no `continue` allowed.
    no_continue: usize,
    fragment: bool,
}

impl Gen {
    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn vars_of(&self, ty: T, mutable: bool) -> Vec<Var> {
        self.scopes
            .iter()
            .flatten()
            .filter(|v| v.ty == ty && (!mutable || v.mutable))
            .cloned()
            .collect()
    }

    fn any_var(&mut self, ty: T) -> Option<String> {
        let vs = self.vars_of(ty, false);
        if vs.is_empty() {
            None
        } else {
            Some(vs[self.rng.below(vs.len())].name.clone())
        }
    }

    fn random_type(&mut self) -> T {
        let s = [S::F, S::F, S::I, S::U, S::B][self.rng.below(5)];
        match self.rng.below(9) {
            0..=2 => T::Scalar(s),
            3 | 4 => T::Vec(2 + self.rng.below(3) as u8, s),
            5 | 6 => T::Mat(2 + self.rng.below(3) as u8, 2 + self.rng.below(3) as u8),
            7 => T::Item,
            _ => T::Arr,
        }
    }

    fn literal(&mut self, s: S) -> String {
        match s {
            S::F => {
                let v = self.rng.below(2000) as f32 / 100.0 - 10.0;
                if self.rng.chance(30) {
                    format!("{v:?}f")
                } else {
                    format!("{v:?}")
                }
            }
            S::I => {
                let v = self.rng.below(200) as i32 - 100;
                if self.rng.chance(40) {
                    format!("{v}i")
                } else {
                    format!("i32({v})")
                }
            }
            S::U => format!("{}u", self.rng.below(100)),
            S::B => if self.rng.chance(50) { "true" } else { "false" }.into(),
        }
    }

    /// An expression of type `ty`; `depth` bounds the tree.
    fn expr(&mut self, ty: T, depth: usize) -> String {
        if depth == 0 || self.rng.chance(20) {
            if let Some(v) = self.any_var(ty)
                && self.rng.chance(75)
            {
                return v;
            }
            return self.leaf(ty);
        }
        let d = depth - 1;
        match ty {
            T::Scalar(s) => self.scalar_expr(s, d),
            T::Vec(n, s) => self.vector_expr(n, s, d),
            T::Item => match self.rng.below(4) {
                0 => "items[lane % arrayLength(&items)]".to_string(),
                1 => format!("make_item({})", self.expr(T::Scalar(S::F), d)),
                2 => format!(
                    "Item({}, {}, {}, {})",
                    self.expr(T::Vec(3, S::F), d),
                    self.expr(T::Scalar(S::U), d),
                    self.expr(T::Arr, d),
                    self.expr(T::Mat(2, 2), d)
                ),
                _ => format!("bounce({})", self.expr(T::Item, d)),
            },
            T::Arr => match self.rng.below(4) {
                0 => "items[lane % arrayLength(&items)].c".to_string(),
                1 => format!("({}).c", self.expr(T::Item, d)),
                2 => format!("rotate3({})", self.expr(T::Arr, d)),
                _ => format!(
                    "array<f32, 3>({}, {}, {})",
                    self.expr(T::Scalar(S::F), d),
                    self.expr(T::Scalar(S::F), d),
                    self.expr(T::Scalar(S::F), d)
                ),
            },
            T::Mat(c, r) => match self.rng.below(4) {
                0 => {
                    let cols: Vec<String> = (0..c).map(|_| self.expr(T::Vec(r, S::F), d)).collect();
                    format!("mat{c}x{r}f({})", cols.join(", "))
                }
                1 => format!("transpose({})", self.expr(T::Mat(r, c), d)),
                2 => format!(
                    "({} * {})",
                    self.expr(T::Mat(c, r), d),
                    self.expr(T::Scalar(S::F), d)
                ),
                _ => {
                    let k = 2 + self.rng.below(3) as u8;
                    format!(
                        "({} * {})",
                        self.expr(T::Mat(k, r), d),
                        self.expr(T::Mat(c, k), d)
                    )
                }
            },
        }
    }

    fn leaf(&mut self, ty: T) -> String {
        match ty {
            T::Scalar(s) => self.literal(s),
            T::Vec(n, s) => {
                if self.rng.chance(50) {
                    format!("vec{n}<{}>({})", sname(s), self.literal(s))
                } else {
                    let parts: Vec<String> = (0..n).map(|_| self.literal(s)).collect();
                    format!("vec{n}<{}>({})", sname(s), parts.join(", "))
                }
            }
            T::Mat(c, r) => format!("mat{c}x{r}<f32>()"),
            T::Item => "Item()".into(),
            T::Arr => {
                let parts: Vec<String> = (0..3).map(|_| self.literal(S::F)).collect();
                format!("array<f32, 3>({})", parts.join(", "))
            }
        }
    }

    fn call_helper(&mut self, ty: T, d: usize) -> Option<String> {
        let cands: Vec<(String, Vec<T>)> = self
            .helpers
            .iter()
            .filter(|h| h.2 == ty)
            .map(|h| (h.0.clone(), h.1.clone()))
            .collect();
        if cands.is_empty() {
            return None;
        }
        let (name, params) = cands[self.rng.below(cands.len())].clone();
        let args: Vec<String> = params.iter().map(|p| self.expr(*p, d)).collect();
        Some(format!("{name}({})", args.join(", ")))
    }

    fn scalar_expr(&mut self, s: S, d: usize) -> String {
        if self.rng.chance(8)
            && let Some(c) = self.call_helper(T::Scalar(s), d)
        {
            return c;
        }
        match s {
            S::F => match self.rng.below(14) {
                0 => format!(
                    "({} {} {})",
                    self.expr(T::Scalar(S::F), d),
                    ["+", "-", "*", "/", "%"][self.rng.below(5)],
                    self.expr(T::Scalar(S::F), d)
                ),
                1 => format!(
                    "{}({})",
                    [
                        "sin", "cos", "abs", "floor", "fract", "exp2", "trunc", "round",
                        "saturate", "sign"
                    ][self.rng.below(10)],
                    self.expr(T::Scalar(S::F), d)
                ),
                2 => format!("sqrt(abs({}))", self.expr(T::Scalar(S::F), d)),
                3 => format!(
                    "clamp({}, {}, {})",
                    self.expr(T::Scalar(S::F), d),
                    self.literal(S::F),
                    self.expr(T::Scalar(S::F), d)
                ),
                4 => {
                    let n = 2 + self.rng.below(3) as u8;
                    format!(
                        "{}({}, {})",
                        ["dot", "distance"][self.rng.below(2)],
                        self.expr(T::Vec(n, S::F), d),
                        self.expr(T::Vec(n, S::F), d)
                    )
                }
                5 => {
                    let n = 2 + self.rng.below(3) as u8;
                    format!("length({})", self.expr(T::Vec(n, S::F), d))
                }
                6 => {
                    let from = [S::I, S::U, S::B][self.rng.below(3)];
                    format!("f32({})", self.expr(T::Scalar(from), d))
                }
                7 => {
                    let n = 2 + self.rng.below(3) as u8;
                    let v = self.expr(T::Vec(n, S::F), d);
                    if self.rng.chance(50) {
                        format!("({v}).{}", ["x", "y", "r", "g"][self.rng.below(4)])
                    } else {
                        format!("({v})[{} % {n}u]", self.expr(T::Scalar(S::U), d))
                    }
                }
                8 => format!(
                    "select({}, {}, {})",
                    self.expr(T::Scalar(S::F), d),
                    self.expr(T::Scalar(S::F), d),
                    self.expr(T::Scalar(S::B), d)
                ),
                9 => format!(
                    "mix({}, {}, {})",
                    self.expr(T::Scalar(S::F), d),
                    self.expr(T::Scalar(S::F), d),
                    self.expr(T::Scalar(S::F), d)
                ),
                10 => format!("-({})", self.expr(T::Scalar(S::F), d)),
                11 => {
                    let c = 2 + self.rng.below(3) as u8;
                    format!("determinant({})", self.expr(T::Mat(c, c), d))
                }
                12 => match self.rng.below(5) {
                    0 => format!("(u.scale * {})", self.expr(T::Scalar(S::F), d)),
                    1 => format!(
                        "({})[{} % 3u]",
                        self.expr(T::Arr, d),
                        self.expr(T::Scalar(S::U), d)
                    ),
                    2 => format!("({}).c[{}]", self.expr(T::Item, d), self.rng.below(3)),
                    3 => format!(
                        "({}).a.{}",
                        self.expr(T::Item, d),
                        ["x", "y", "z"][self.rng.below(3)]
                    ),
                    _ => format!(
                        "({}).m[{} % 2u].y",
                        self.expr(T::Item, d),
                        self.expr(T::Scalar(S::U), d)
                    ),
                },
                _ => format!(
                    "bitcast<f32>({} & 0x3fffffffu)",
                    self.expr(T::Scalar(S::U), d)
                ),
            },
            S::I | S::U => {
                let t = T::Scalar(s);
                match self.rng.below(11) {
                    0 => format!(
                        "({} {} {})",
                        self.expr(t, d),
                        ["+", "-", "*", "&", "|", "^"][self.rng.below(6)],
                        self.expr(t, d)
                    ),
                    1 => {
                        let lhs = self.any_var(t).unwrap_or_else(|| "lane".to_string());
                        let lhs = if s == S::I {
                            format!("i32({lhs})")
                        } else {
                            format!("u32({lhs})")
                        };
                        format!(
                            "({lhs} {} ({} | {}))",
                            ["/", "%"][self.rng.below(2)],
                            self.expr(t, d),
                            if s == S::I { "1i" } else { "1u" }
                        )
                    }
                    2 => format!(
                        "({} {} {}u)",
                        self.expr(t, d),
                        ["<<", ">>"][self.rng.below(2)],
                        self.rng.below(31)
                    ),
                    3 => format!(
                        "({} >> ({} & 31u))",
                        self.expr(t, d),
                        self.expr(T::Scalar(S::U), d)
                    ),
                    4 => format!(
                        "{}({}, {})",
                        ["min", "max"][self.rng.below(2)],
                        self.expr(t, d),
                        self.expr(t, d)
                    ),
                    5 => format!(
                        "{}({})",
                        ["abs", "countOneBits", "reverseBits"][self.rng.below(3)],
                        self.expr(t, d)
                    ),
                    6 => {
                        let from = [S::F, S::I, S::U, S::B][self.rng.below(4)];
                        format!("{}({})", sname(s), self.expr(T::Scalar(from), d))
                    }
                    7 => format!("~({})", self.expr(t, d)),
                    8 if s == S::U && self.rng.chance(40) => {
                        format!("({}).b", self.expr(T::Item, d))
                    }
                    8 => {
                        let n = 2 + self.rng.below(3) as u8;
                        format!(
                            "({}).{}",
                            self.expr(T::Vec(n, s), d),
                            ["x", "y"][self.rng.below(2)]
                        )
                    }
                    9 => {
                        if s == S::I {
                            format!("-({})", self.expr(t, d))
                        } else {
                            format!("(arrayLength(&data) + {})", self.expr(t, d))
                        }
                    }
                    _ => format!(
                        "clamp({}, {}, {})",
                        self.expr(t, d),
                        self.literal(s),
                        self.expr(t, d)
                    ),
                }
            }
            S::B => match self.rng.below(8) {
                0 => {
                    let ot = T::Scalar([S::F, S::I, S::U][self.rng.below(3)]);
                    format!(
                        "({} {} {})",
                        self.expr(ot, d),
                        ["<", "<=", ">", ">=", "==", "!="][self.rng.below(6)],
                        self.expr(ot, d)
                    )
                }
                1 => format!(
                    "({} {} {})",
                    self.expr(T::Scalar(S::B), d),
                    ["&&", "||", "&", "|", "==", "!="][self.rng.below(6)],
                    self.expr(T::Scalar(S::B), d)
                ),
                2 => format!("!({})", self.expr(T::Scalar(S::B), d)),
                3 => {
                    let n = 2 + self.rng.below(3) as u8;
                    format!(
                        "{}({})",
                        ["all", "any"][self.rng.below(2)],
                        self.expr(T::Vec(n, S::B), d)
                    )
                }
                4 => format!("({} && helper_flag())", self.expr(T::Scalar(S::B), d)),
                5 => format!(
                    "(lane < arrayLength(&data) && data[lane] > {})",
                    self.expr(T::Scalar(S::F), d)
                ),
                6 => format!("bool({})", self.expr(T::Scalar(S::F), d)),
                _ => format!(
                    "select({}, {}, {})",
                    self.expr(T::Scalar(S::B), d),
                    self.expr(T::Scalar(S::B), d),
                    self.expr(T::Scalar(S::B), d)
                ),
            },
        }
    }

    fn vector_expr(&mut self, n: u8, s: S, d: usize) -> String {
        let t = T::Vec(n, s);
        match self.rng.below(10) {
            0 if s != S::B => format!(
                "({} {} {})",
                self.expr(t, d),
                ["+", "-", "*"][self.rng.below(3)],
                self.expr(t, d)
            ),
            1 if s != S::B => format!("({} * {})", self.expr(t, d), self.expr(T::Scalar(s), d)),
            2 if s == S::F => format!("({} / {})", self.expr(t, d), self.expr(T::Scalar(S::F), d)),
            3 => {
                let src_n = 2 + self.rng.below(3) as u8;
                let comps: String = (0..n)
                    .map(|_| ['x', 'y', 'z', 'w'][self.rng.below(src_n as usize)])
                    .collect();
                format!("({}).{comps}", self.expr(T::Vec(src_n, s), d))
            }
            4 if s == S::F => format!(
                "{}({})",
                ["normalize", "abs", "fract", "floor", "sin"][self.rng.below(5)],
                self.expr(t, d)
            ),
            5 if s == S::F && self.rng.chance(30) => {
                let c = 2 + self.rng.below(3) as u8;
                format!(
                    "({})[{} % {c}u]",
                    self.expr(T::Mat(c, n), d),
                    self.expr(T::Scalar(S::U), d)
                )
            }
            5 if s == S::F => format!("({} * {})", self.expr(T::Mat(n, n), d), self.expr(t, d)),
            6 if s == S::F && n == 3 => format!("cross({}, {})", self.expr(t, d), self.expr(t, d)),
            6 if s == S::F => format!("({} * {})", self.expr(t, d), self.expr(T::Mat(n, n), d)),
            7 => format!(
                "select({}, {}, {})",
                self.expr(t, d),
                self.expr(t, d),
                if self.rng.chance(50) {
                    self.expr(T::Scalar(S::B), d)
                } else {
                    self.expr(T::Vec(n, S::B), d)
                }
            ),
            8 if s == S::B => {
                let ot = T::Vec(n, [S::F, S::I, S::U][self.rng.below(3)]);
                format!(
                    "({} {} {})",
                    self.expr(ot, d),
                    ["<", ">=", "==", "!="][self.rng.below(4)],
                    self.expr(ot, d)
                )
            }
            8 => {
                let from = [S::F, S::I, S::U, S::B][self.rng.below(4)];
                format!("vec{n}<{}>({})", sname(s), self.expr(T::Vec(n, from), d))
            }
            9 if s == S::F && n == 4 => format!(
                "textureSampleLevel(tex, samp, {}, 0.0)",
                self.expr(T::Vec(2, S::F), d)
            ),
            9 if s == S::F && n == 4 && self.fragment => format!(
                "textureSample(tex, samp, {})",
                self.expr(T::Vec(2, S::F), d)
            ),
            _ => {
                if n >= 3 && self.rng.chance(50) {
                    format!(
                        "vec{n}<{}>({}, {})",
                        sname(s),
                        self.expr(T::Vec(n - 1, s), d),
                        self.expr(T::Scalar(s), d)
                    )
                } else {
                    let parts: Vec<String> = (0..n).map(|_| self.expr(T::Scalar(s), d)).collect();
                    format!("vec{n}<{}>({})", sname(s), parts.join(", "))
                }
            }
        }
    }

    fn indent(out: &mut String, level: usize) {
        for _ in 0..level {
            out.push_str("  ");
        }
    }

    fn block(&mut self, out: &mut String, level: usize, budget: usize, ret: Option<T>) {
        self.scopes.push(Vec::new());
        let n = 1 + self.rng.below(budget.max(1));
        for _ in 0..n {
            self.stmt(out, level, budget / 2, ret);
        }
        self.scopes.pop();
    }

    fn stmt(&mut self, out: &mut String, level: usize, budget: usize, ret: Option<T>) {
        Self::indent(out, level);
        let choice = if budget == 0 {
            self.rng.below(4)
        } else {
            self.rng.below(14)
        };
        match choice {
            0 => {
                let ty = self.random_type();
                let name = self.fresh("l");
                let e = self.expr(ty, 3);
                let _ = writeln!(out, "let {name}: {} = {e};", ty.wgsl());
                self.scopes.last_mut().unwrap().push(Var {
                    name,
                    ty,
                    mutable: false,
                });
            }
            1 => {
                let ty = self.random_type();
                let name = self.fresh("v");
                if self.rng.chance(30) {
                    let _ = writeln!(out, "var {name}: {};", ty.wgsl());
                } else {
                    let e = self.expr(ty, 3);
                    let _ = writeln!(out, "var {name} = {e};");
                }
                self.scopes.last_mut().unwrap().push(Var {
                    name,
                    ty,
                    mutable: true,
                });
            }
            2 | 3 => self.assignment(out),
            4 => {
                let c = self.expr(T::Scalar(S::B), 3);
                let _ = writeln!(out, "if ({c}) {{");
                self.block(out, level + 1, budget, ret);
                Self::indent(out, level);
                if self.rng.chance(50) {
                    out.push_str("} else {\n");
                    self.block(out, level + 1, budget, ret);
                    Self::indent(out, level);
                }
                out.push_str("}\n");
            }
            5 => {
                let i = self.fresh("i");
                let _ = writeln!(
                    out,
                    "for (var {i} = 0u; {i} < {}u; {i}++) {{",
                    1 + self.rng.below(5)
                );
                self.scopes.push(vec![Var {
                    name: i.clone(),
                    ty: T::Scalar(S::U),
                    mutable: false,
                }]);
                self.in_loop += 1;
                self.block(out, level + 1, budget, ret);
                self.in_loop -= 1;
                self.scopes.pop();
                Self::indent(out, level);
                out.push_str("}\n");
            }
            6 => {
                let c = self.expr(T::Scalar(S::B), 2);
                let _ = writeln!(out, "while ({c}) {{");
                self.in_loop += 1;
                self.block(out, level + 1, budget, ret);
                Self::indent(out, level + 1);
                out.push_str("break;\n");
                self.in_loop -= 1;
                Self::indent(out, level);
                out.push_str("}\n");
            }
            7 => {
                out.push_str("loop {\n");
                // Half of the loops keep their body scope open so `continuing` can read the
                // body's `let`s; those bodies contain no `continue` (WGSL forbids skipping a
                // declaration that `continuing` uses).
                let shared = self.rng.chance(50);
                self.in_loop += 1;
                if shared {
                    self.no_continue += 1;
                    self.scopes.push(Vec::new());
                    for _ in 0..1 + self.rng.below(budget.max(1)) {
                        self.stmt(out, level + 1, budget / 2, ret);
                    }
                    self.no_continue -= 1;
                } else {
                    self.block(out, level + 1, budget, ret);
                }
                self.in_loop -= 1;
                Self::indent(out, level + 1);
                out.push_str("continuing {\n");
                self.scopes.push(Vec::new());
                if self.rng.chance(60) {
                    self.assignment_at(out, level + 2);
                }
                if shared && self.rng.chance(70) {
                    let e = self.expr(T::Scalar(S::F), 2);
                    Self::indent(out, level + 2);
                    let _ = writeln!(out, "acc.y += {e};");
                }
                self.scopes.pop();
                if shared {
                    self.scopes.pop();
                }
                let c = self.expr(T::Scalar(S::B), 2);
                Self::indent(out, level + 2);
                let _ = writeln!(out, "break if {c} || true;");
                Self::indent(out, level + 1);
                out.push_str("}\n");
                Self::indent(out, level);
                out.push_str("}\n");
            }
            8 if self.in_loop > 0 => {
                let c = self.expr(T::Scalar(S::B), 2);
                let jump = if self.no_continue > 0 || self.rng.chance(50) {
                    "break;"
                } else {
                    "continue;"
                };
                let _ = writeln!(out, "if ({c}) {{ {jump} }}");
            }
            9 => {
                if let Some(rt) = ret {
                    let e = self.expr(rt, 3);
                    let c = self.expr(T::Scalar(S::B), 2);
                    let _ = writeln!(out, "if ({c}) {{ return {e}; }}");
                } else if self.fragment {
                    let c = self.expr(T::Scalar(S::B), 2);
                    let _ = writeln!(out, "if ({c}) {{ discard; }}");
                } else {
                    out.push_str("{}\n");
                }
            }
            10 => {
                let vs: Vec<Var> = self
                    .scopes
                    .iter()
                    .flatten()
                    .filter(|v| v.mutable && matches!(v.ty, T::Scalar(S::I | S::U)))
                    .cloned()
                    .collect();
                if vs.is_empty() {
                    out.push_str("_ = lane;\n");
                } else {
                    let v = &vs[self.rng.below(vs.len())];
                    let _ = writeln!(
                        out,
                        "{}{};",
                        v.name,
                        if self.rng.chance(50) { "++" } else { "--" }
                    );
                }
            }
            11 => {
                out.push_str("{\n");
                self.block(out, level + 1, budget, ret);
                Self::indent(out, level);
                out.push_str("}\n");
            }
            12 => match self.rng.below(4) {
                0 => {
                    let e = self.expr(T::Vec(4, S::F), 3);
                    let _ = writeln!(out, "data[lane % arrayLength(&data)] += ({e}).x;");
                }
                1 => {
                    let e = self.expr(T::Item, 3);
                    let _ = writeln!(out, "items[lane % arrayLength(&items)] = {e};");
                }
                2 => {
                    let e = self.expr(T::Scalar(S::F), 3);
                    let _ = writeln!(
                        out,
                        "items[lane % arrayLength(&items)].c[{}] -= {e};",
                        self.rng.below(3)
                    );
                }
                _ => {
                    let e = self.expr(T::Arr, 3);
                    let _ = writeln!(out, "items[0].c = {e};");
                }
            },
            _ => {
                // `vecN += scalar` is valid WGSL, so a scalar right-hand side works for every target.
                let target = ["x", "y", "z", "w", "xy", "zw", "xyz"][self.rng.below(7)];
                let e = self.expr(T::Scalar(S::F), 3);
                let _ = writeln!(out, "acc.{target} += {e};");
            }
        }
    }

    fn assignment(&mut self, out: &mut String) {
        self.assignment_at(out, 0);
    }

    fn assignment_at(&mut self, out: &mut String, level: usize) {
        if level > 0 {
            Self::indent(out, level);
        }
        let vs: Vec<Var> = self
            .scopes
            .iter()
            .flatten()
            .filter(|v| v.mutable)
            .cloned()
            .collect();
        if vs.is_empty() {
            out.push_str("acc.w += 1.0;\n");
            return;
        }
        let v = vs[self.rng.below(vs.len())].clone();
        match v.ty {
            T::Vec(n, s) if self.rng.chance(50) => {
                let k = 1 + self.rng.below(n as usize);
                let mut comps: Vec<char> = ['x', 'y', 'z', 'w'][..n as usize].to_vec();
                for i in 0..comps.len() {
                    let j = self.rng.below(comps.len());
                    comps.swap(i, j);
                }
                let sw: String = comps[..k].iter().collect();
                let rhs_t = if k == 1 {
                    T::Scalar(s)
                } else {
                    T::Vec(k as u8, s)
                };
                let e = self.expr(rhs_t, 3);
                let op = if s == S::B {
                    "="
                } else {
                    ["=", "+=", "*="][self.rng.below(3)]
                };
                let _ = writeln!(out, "{}.{sw} {op} {e};", v.name);
            }
            T::Vec(n, s) if self.rng.chance(30) => {
                let i = self.expr(T::Scalar(S::U), 2);
                let e = self.expr(T::Scalar(s), 3);
                let _ = writeln!(out, "{}[{i} % {n}u] = {e};", v.name);
            }
            T::Item if self.rng.chance(70) => {
                let line = match self.rng.below(4) {
                    0 => format!("{}.a = {};", v.name, self.expr(T::Vec(3, S::F), 3)),
                    1 => format!(
                        "{}.c[{} % 3u] += {};",
                        v.name,
                        self.expr(T::Scalar(S::U), 2),
                        self.expr(T::Scalar(S::F), 3)
                    ),
                    2 => format!("{}.m[1].x = {};", v.name, self.expr(T::Scalar(S::F), 3)),
                    _ => format!("{}.c = {};", v.name, self.expr(T::Arr, 3)),
                };
                out.push_str(&line);
                out.push('\n');
            }
            T::Arr if self.rng.chance(60) => {
                let i = self.expr(T::Scalar(S::U), 2);
                let e = self.expr(T::Scalar(S::F), 3);
                let _ = writeln!(out, "{}[{i} % 3u] = {e};", v.name);
            }
            T::Mat(c, r) if self.rng.chance(50) => {
                let e = self.expr(T::Vec(r, S::F), 3);
                let _ = writeln!(out, "{}[{}] = {e};", v.name, self.rng.below(c as usize));
            }
            ty => {
                let e = self.expr(ty, 3);
                let op = match ty.scalar() {
                    S::B => "=",
                    S::F => ["=", "+=", "-=", "*="][self.rng.below(4)],
                    _ => ["=", "+=", "-=", "*=", "&=", "|=", "^="][self.rng.below(7)],
                };
                let op = if matches!(ty, T::Mat(..)) && op == "*=" {
                    "+="
                } else {
                    op
                };
                let op = if matches!(ty, T::Item | T::Arr) {
                    "="
                } else {
                    op
                };
                let _ = writeln!(out, "{} {op} {e};", v.name);
            }
        }
    }

    fn helper(&mut self, out: &mut String) {
        let name = self.fresh("helper");
        let nparams = self.rng.below(3);
        let params: Vec<T> = (0..nparams).map(|_| self.random_type()).collect();
        let ret = self.random_type();
        let mut sig = Vec::new();
        let mut scope = vec![Var {
            name: "lane".into(),
            ty: T::Scalar(S::U),
            mutable: false,
        }];
        for (i, p) in params.iter().enumerate() {
            sig.push(format!("a{i}: {}", p.wgsl()));
            scope.push(Var {
                name: format!("a{i}"),
                ty: *p,
                mutable: false,
            });
        }
        let _ = writeln!(out, "fn {name}({}) -> {} {{", sig.join(", "), ret.wgsl());
        out.push_str("  var acc = vec4f(0.0);\n");
        scope.push(Var {
            name: "acc".into(),
            ty: T::Vec(4, S::F),
            mutable: true,
        });
        self.scopes = vec![scope];
        for _ in 0..1 + self.rng.below(4) {
            self.stmt(out, 1, 3, Some(ret));
        }
        let e = self.expr(ret, 3);
        let _ = writeln!(out, "  return {e};\n}}\n");
        self.helpers.push((name, params, ret));
    }

    fn program(&mut self) -> String {
        let mut out = String::from(
            "struct U { scale: f32, tint: vec4f, m: mat4x4f }
struct Item { a: vec3f, b: u32, c: array<f32, 3>, m: mat2x2f }
@group(0) @binding(4) var<storage, read_write> items: array<Item>;
fn make_item(x: f32) -> Item { return Item(vec3f(x), u32(abs(x)), array<f32, 3>(x, -x, 2.0 * x), mat2x2f(x, 0.0, 0.0, x)); }
fn bounce(it: Item) -> Item { var c = it; c.c[c.b % 3u] += 1.0; return c; }
fn rotate3(a: array<f32, 3>) -> array<f32, 3> { return array<f32, 3>(a[1], a[2], a[0]); }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
var<private> lane: u32;
var<private> flag_calls: u32;
fn helper_flag() -> bool { flag_calls++; return (flag_calls & 1u) == 0u; }
",
        );
        self.fragment = false;
        for _ in 0..self.rng.below(4) {
            self.helper(&mut out);
        }
        self.fragment = self.rng.chance(60);
        let header = if self.fragment {
            "@fragment fn main(@location(0) p: vec4f, @builtin(position) pos: vec4f) -> @location(0) vec4f {\n  lane = u32(pos.x);\n"
        } else {
            "@compute @workgroup_size(8) fn main(@builtin(global_invocation_id) gid: vec3u) {\n  lane = gid.x;\n  let p = vec4f(vec3f(gid), 1.0);\n"
        };
        out.push_str(header);
        out.push_str("  var acc = p * u.tint;\n");
        self.scopes = vec![vec![
            Var {
                name: "lane".into(),
                ty: T::Scalar(S::U),
                mutable: false,
            },
            Var {
                name: "p".into(),
                ty: T::Vec(4, S::F),
                mutable: false,
            },
            Var {
                name: "acc".into(),
                ty: T::Vec(4, S::F),
                mutable: true,
            },
        ]];
        for _ in 0..2 + self.rng.below(8) {
            self.stmt(&mut out, 1, 4, None);
        }
        if self.fragment {
            out.push_str("  return acc + u.m * p;\n}\n");
        } else {
            out.push_str("  data[lane % arrayLength(&data)] = acc.x + acc.y + acc.z + acc.w;\n}\n");
        }
        out
    }
}

fn spirv_val(words: &[u32], n: usize) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("ganbate-shader-random-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("r{n}.spv"));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    let out = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.0"])
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

#[test]
fn random_well_typed_programs_validate() {
    let count: usize = std::env::var("GANBATE_SHADER_RANDOM_PROGRAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);
    let seed: u64 = std::env::var("GANBATE_SHADER_RANDOM_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x5eed_1234_abcd_0001);
    let validate = Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !validate {
        eprintln!(
            "note: spirv-val not found on PATH; random programs are compiled but not validated"
        );
    }
    let clang = msl_check::clang();
    let mut failures = 0;
    let mut rejected = 0;
    for n in 0..count {
        let mut g = Gen {
            rng: Rng(seed
                .wrapping_add(n as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                | 1),
            scopes: Vec::new(),
            counter: 0,
            helpers: Vec::new(),
            in_loop: 0,
            no_continue: 0,
            fragment: false,
        };
        let src = g.program();
        if let Ok(dir) = std::env::var("GANBATE_SHADER_RANDOM_DUMP") {
            let _ = std::fs::write(
                std::path::Path::new(&dir).join(format!("random{n}.wgsl")),
                &src,
            );
        }
        let stage = if g.fragment {
            Stage::Fragment
        } else {
            Stage::Compute
        };
        // Every program also goes through the MSL emitter; a sample is compiled with clang++
        // against the stand-in metal_stdlib when it is available.
        match compile_msl(&src, stage, "main") {
            Ok(c) => {
                if let (Some(clang), true) = (&clang, n % 3 == 0)
                    && let Err(msg) = msl_check::clang_check(
                        clang,
                        &format!("random{n}"),
                        &c.msl.expect("compile_msl fills msl").source,
                    )
                {
                    failures += 1;
                    eprintln!("program {n}: MSL rejected:\n{msg}\n{src}");
                }
            }
            Err(e) if CONST_EVAL_ERRORS.iter().any(|m| e.message.contains(m)) => {}
            Err(e) => {
                failures += 1;
                eprintln!("program {n}: MSL emission failed: {e}\n{src}");
            }
        }
        match compile(&src, stage, "main") {
            // Random constant subexpressions can legitimately overflow (`0u - 28u`) or divide
            // by zero; WGSL makes those shader-creation errors.
            Err(e) if CONST_EVAL_ERRORS.iter().any(|m| e.message.contains(m)) => rejected += 1,
            Err(e) => {
                failures += 1;
                eprintln!("program {n}: compile error {e}\n{src}");
            }
            Ok(c) => {
                if validate && let Err(msg) = spirv_val(&c.words, n) {
                    failures += 1;
                    eprintln!("program {n}: spirv-val failed:\n{msg}\n{src}");
                }
            }
        }
        if failures >= 3 {
            break;
        }
    }
    assert_eq!(failures, 0, "random program failures (see stderr)");
    eprintln!(
        "{count} random programs: {} compiled and validated, {rejected} rejected by constant evaluation",
        count - rejected
    );
    assert!(
        rejected * 5 < count,
        "too many programs rejected ({rejected}/{count})"
    );
}

const CONST_EVAL_ERRORS: &[&str] = &[
    "in constant expression",
    "infinity or NaN",
    "cannot be represented",
    "is not less than the bit width",
];
