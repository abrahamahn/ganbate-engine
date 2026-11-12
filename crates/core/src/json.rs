//! JSON for authored content: a bounded reader, and a canonical writer that gives every value
//! one spelling, so authored files diff and merge line by line.
//!
//! Canonical text: objects print one key per line in byte order of their keys, indented two
//! spaces. Arrays of numbers, booleans and nulls print on one line (`[0, 1.5, 0]`: vectors,
//! colours, matrices); any other nonempty array prints one element per line. Strings escape
//! only what JSON requires. A number prints in the shortest form that reads back to the same
//! value, and a number that is exactly an `f32` prints in `f32`'s shortest form (`0.1`, not
//! `0.10000000149011612`), so values computed in single precision stay readable. The text ends
//! with a newline. Writing what was read back from canonical text gives the same bytes.
//!
//! The reader refuses text longer than `MAX_BYTES`, nested deeper than `MAX_DEPTH` or holding
//! more than `MAX_VALUES` values, and the canonical writer refuses to write such text, so a
//! document that could be saved can always be read again.
use std::collections::BTreeMap;

/// Longest text `parse` reads and `to_canonical` writes.
pub const MAX_BYTES: usize = 32 * 1024 * 1024;
/// Deepest nesting `parse` reads and `to_canonical` writes: the outermost value is at depth 0,
/// the members of an object or array one deeper than it.
pub const MAX_DEPTH: usize = 64;
/// Most values `parse` reads and `to_canonical` writes, counting every array element, object
/// member and the outermost value. Authored documents spend about 20 bytes on a value, so they
/// meet `MAX_BYTES` long before this; it bounds what dense text can make the reader allocate.
pub const MAX_VALUES: usize = MAX_BYTES / 4;

#[derive(Clone, Copy, Debug)]
struct Limits {
    bytes: usize,
    depth: usize,
    values: usize,
}

const LIMITS: Limits = Limits {
    bytes: MAX_BYTES,
    depth: MAX_DEPTH,
    values: MAX_VALUES,
};

impl Limits {
    fn too_long(&self) -> String {
        if self.bytes % (1 << 20) == 0 {
            format!("JSON exceeds {} MiB", self.bytes >> 20)
        } else {
            format!("JSON exceeds {} bytes", self.bytes)
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}
impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        if let Self::Object(m) = self {
            m.get(key)
        } else {
            None
        }
    }
    pub fn array(&self) -> Result<&[Value], String> {
        if let Self::Array(a) = self {
            Ok(a)
        } else {
            Err("expected array".into())
        }
    }
    pub fn string(&self) -> Result<&str, String> {
        if let Self::String(s) = self {
            Ok(s)
        } else {
            Err("expected string".into())
        }
    }
    pub fn integer(&self) -> Result<usize, String> {
        usize::try_from(self.unsigned()?).map_err(|_| "integer exceeds index range".into())
    }
    /// Exactly representable, nonnegative JSON integer, independent of pointer width.
    pub fn unsigned(&self) -> Result<u64, String> {
        if let Self::Number(n) = self {
            if (0.0..=9_007_199_254_740_991.0).contains(n) && n.fract() == 0.0 {
                return Ok(*n as u64);
            }
        }
        Err("expected unsigned integer".into())
    }
    pub fn float(&self) -> Result<f32, String> {
        if let Self::Number(n) = self {
            let n = *n as f32;
            if n.is_finite() {
                return Ok(n);
            }
        }
        Err("expected finite float".into())
    }
}
pub fn parse(input: &[u8]) -> Result<Value, String> {
    parse_within(input, LIMITS)
}
fn parse_within(input: &[u8], limits: Limits) -> Result<Value, String> {
    if input.len() > limits.bytes {
        return Err(limits.too_long());
    }
    let mut p = Parser {
        b: input,
        i: 0,
        values: 0,
        limits,
    };
    let result = p.value(0)?;
    p.space();
    if p.i != input.len() {
        return Err(p.err("trailing data"));
    }
    Ok(result)
}
struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    values: usize,
    limits: Limits,
}
impl Parser<'_> {
    fn err(&self, msg: &str) -> String {
        format!("JSON byte {}: {msg}", self.i)
    }
    #[inline]
    fn byte(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    #[inline]
    fn eat(&mut self, b: u8) -> bool {
        if self.byte() == Some(b) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    #[inline]
    fn space(&mut self) {
        let len = self.b.len();
        while self.i < len {
            let b = self.b[self.i];
            if b == b' ' || b == b'\n' || b == b'\r' || b == b'\t' {
                self.i += 1;
            } else {
                break;
            }
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        self.space();
        self.values += 1;
        if depth > self.limits.depth || self.values > self.limits.values {
            return Err(self.err("structure limit"));
        }
        match self.byte() {
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut out = vec![];
                self.space();
                if self.eat(b']') {
                    return Ok(Value::Array(out));
                }
                loop {
                    out.push(self.value(depth + 1)?);
                    self.space();
                    if self.eat(b']') {
                        return Ok(Value::Array(out));
                    }
                    if !self.eat(b',') {
                        return Err(self.err("expected array comma"));
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut out = BTreeMap::new();
                self.space();
                if self.eat(b'}') {
                    return Ok(Value::Object(out));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    if !self.eat(b':') {
                        return Err(self.err("expected colon"));
                    }
                    if out.insert(key, self.value(depth + 1)?).is_some() {
                        return Err(self.err("duplicate key"));
                    }
                    self.space();
                    if self.eat(b'}') {
                        return Ok(Value::Object(out));
                    }
                    if !self.eat(b',') {
                        return Err(self.err("expected object comma"));
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => {
                for (s, v) in [
                    (b"null".as_slice(), Value::Null),
                    (b"true", Value::Bool(true)),
                    (b"false", Value::Bool(false)),
                ] {
                    if self.b[self.i..].starts_with(s) {
                        self.i += s.len();
                        return Ok(v);
                    }
                }
                Err(self.err("expected value"))
            }
        }
    }
    fn digits(&mut self) -> Result<(), String> {
        let start = self.i;
        while matches!(self.byte(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.i == start {
            Err(self.err("expected digit"))
        } else {
            Ok(())
        }
    }
    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        let neg = self.eat(b'-');
        let mut int_acc = 0u64;
        let mut int_digits = 0usize;
        if self.eat(b'0') {
            int_digits = 1;
        } else {
            let d_start = self.i;
            while let Some(b @ b'0'..=b'9') = self.byte() {
                if int_digits < 16 {
                    int_acc = int_acc * 10 + u64::from(b - b'0');
                }
                int_digits += 1;
                self.i += 1;
            }
            if self.i == d_start {
                return Err(self.err("expected digit"));
            }
        }
        let mut is_float = false;
        if self.eat(b'.') {
            is_float = true;
            self.digits()?
        }
        if self.eat(b'e') || self.eat(b'E') {
            is_float = true;
            if !self.eat(b'+') {
                self.eat(b'-');
            }
            self.digits()?
        }
        if !is_float && int_digits <= 15 {
            let f = int_acc as f64;
            return Ok(Value::Number(if neg { -f } else { f }));
        }
        let n = std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .ok_or_else(|| self.err("invalid number"))?;
        Ok(Value::Number(n))
    }
    fn hex(&mut self) -> Result<u32, String> {
        let e = self.i + 4;
        let raw = self
            .b
            .get(self.i..e)
            .ok_or_else(|| self.err("short unicode escape"))?;
        let n = raw
            .iter()
            .try_fold(0u32, |a, &v| (v as char).to_digit(16).map(|d| a * 16 + d))
            .ok_or_else(|| self.err("invalid unicode escape"))?;
        self.i = e;
        Ok(n)
    }
    fn string(&mut self) -> Result<String, String> {
        if !self.eat(b'"') {
            return Err(self.err("expected string"));
        }
        let first_start = self.i;
        let len = self.b.len();
        while self.i < len {
            let c = self.b[self.i];
            if c < 32 || c == b'"' || c == b'\\' {
                break;
            }
            self.i += 1;
        }
        let first_end = self.i;
        if self.eat(b'"') {
            let s = std::str::from_utf8(&self.b[first_start..first_end])
                .map_err(|_| self.err("invalid UTF-8"))?;
            return Ok(s.to_owned());
        }
        let mut out = String::with_capacity((first_end - first_start) + 16);
        out.push_str(
            std::str::from_utf8(&self.b[first_start..first_end])
                .map_err(|_| self.err("invalid UTF-8"))?,
        );
        loop {
            if !self.eat(b'\\') {
                return Err(self.err("unterminated string or control character"));
            }
            let e = self.byte().ok_or_else(|| self.err("short escape"))?;
            self.i += 1;
            out.push(match e {
                b'"' => '"',
                b'\\' => '\\',
                b'/' => '/',
                b'b' => '\u{8}',
                b'f' => '\u{c}',
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'u' => {
                    let mut cp = self.hex()?;
                    if (0xd800..=0xdbff).contains(&cp) {
                        if !self.eat(b'\\') || !self.eat(b'u') {
                            return Err(self.err("missing low surrogate"));
                        }
                        let low = self.hex()?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(self.err("invalid low surrogate"));
                        }
                        cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                    }
                    char::from_u32(cp).ok_or_else(|| self.err("invalid unicode scalar"))?
                }
                _ => return Err(self.err("invalid string escape")),
            });
            let start = self.i;
            while self.i < len {
                let c = self.b[self.i];
                if c < 32 || c == b'"' || c == b'\\' {
                    break;
                }
                self.i += 1;
            }
            out.push_str(
                std::str::from_utf8(&self.b[start..self.i])
                    .map_err(|_| self.err("invalid UTF-8"))?,
            );
            if self.eat(b'"') {
                return Ok(out);
            }
        }
    }
}

/// The canonical text of `value`; refuses non-finite numbers, and text that `parse` would
/// refuse (see `MAX_BYTES`, `MAX_DEPTH` and `MAX_VALUES`).
pub fn to_canonical(value: &Value) -> Result<String, String> {
    canonical_within(value, LIMITS)
}

fn canonical_within(value: &Value, limits: Limits) -> Result<String, String> {
    let cap_hint = match value {
        Value::Object(m) => (m.len() * 32).min(65536).min(limits.bytes),
        Value::Array(a) => (a.len() * 8).min(65536).min(limits.bytes),
        Value::String(s) => (s.len() + 3).min(limits.bytes),
        _ => 32,
    };
    let mut w = Canonical {
        out: String::with_capacity(cap_hint),
        values: 0,
        limits,
    };
    w.value(value, 0)?;
    w.out.push('\n');
    if w.out.len() > limits.bytes {
        return Err(limits.too_long());
    }
    Ok(w.out)
}

/// `value` on one line with the same number and string spelling (for messages and keys).
pub fn to_compact(value: &Value) -> Result<String, String> {
    let mut out = String::with_capacity(64);
    write_compact(&mut out, value)?;
    Ok(out)
}

fn inline(items: &[Value]) -> bool {
    items
        .iter()
        .all(|v| matches!(v, Value::Number(_) | Value::Bool(_) | Value::Null))
}

#[inline]
fn pad(out: &mut String, depth: usize) {
    const SPACES: &str = "                                                                                                                                ";
    let n = depth * 2;
    if n <= SPACES.len() {
        out.push_str(&SPACES[..n]);
    } else {
        for _ in 0..depth {
            out.push_str("  ");
        }
    }
}

/// Canonical text being written, with the values counted as `Parser` counts them.
struct Canonical {
    out: String,
    values: usize,
    limits: Limits,
}

impl Canonical {
    /// Counts `n` values at `depth`, refusing what the reader would refuse.
    fn count(&mut self, n: usize, depth: usize) -> Result<(), String> {
        self.values = self.values.saturating_add(n);
        if depth > self.limits.depth {
            return Err(format!(
                "JSON nests deeper than {} levels",
                self.limits.depth
            ));
        }
        if self.values > self.limits.values {
            return Err(format!(
                "JSON holds more than {} values",
                self.limits.values
            ));
        }
        if self.out.len() > self.limits.bytes {
            return Err(self.limits.too_long());
        }
        Ok(())
    }

    fn value(&mut self, value: &Value, depth: usize) -> Result<(), String> {
        self.count(1, depth)?;
        match value {
            Value::Array(items) if !items.is_empty() && !inline(items) => {
                self.out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    pad(&mut self.out, depth + 1);
                    self.value(item, depth + 1)?;
                    if i + 1 < items.len() {
                        self.out.push(',');
                    }
                    self.out.push('\n');
                }
                pad(&mut self.out, depth);
                self.out.push(']');
            }
            Value::Object(map) if !map.is_empty() => {
                self.out.push_str("{\n");
                for (i, (key, item)) in map.iter().enumerate() {
                    pad(&mut self.out, depth + 1);
                    write_string(&mut self.out, key);
                    self.out.push_str(": ");
                    self.value(item, depth + 1)?;
                    if i + 1 < map.len() {
                        self.out.push(',');
                    }
                    self.out.push('\n');
                }
                pad(&mut self.out, depth);
                self.out.push('}');
            }
            Value::Array(items) => {
                // Scalars on one line, each a value one level down.
                if !items.is_empty() {
                    self.count(items.len(), depth + 1)?;
                }
                write_compact(&mut self.out, value)?;
            }
            other => write_compact(&mut self.out, other)?,
        }
        Ok(())
    }
}

fn write_compact(out: &mut String, value: &Value) -> Result<(), String> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, *n)?,
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_compact(out, item)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_compact(out, item)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Shortest round-trip spelling; `f32` values in `f32`'s; no exponent between 1e-6 and 1e21.
pub fn write_number(out: &mut String, n: f64) -> Result<(), String> {
    use std::fmt::Write;
    if !n.is_finite() {
        return Err(format!("non-finite number {n}"));
    }
    if n == 0.0 {
        out.push('0'); // also -0
        return Ok(());
    }
    if (-16_777_216.0..=16_777_216.0).contains(&n) {
        let iv = n as i32;
        if f64::from(iv) == n {
            write_i32(out, iv);
            return Ok(());
        }
    }
    let single = n as f32;
    let r = if f64::from(single) == n {
        // Decided on the 32-bit value, whose spelling is what is printed: the float nearest
        // 1e-6 is a little below the 64-bit 1e-6 but prints as 1e-6, and the text must read
        // back to a value that is printed the same way.
        if (1e-6f32..1e21f32).contains(&single.abs()) {
            write!(out, "{single}")
        } else {
            write!(out, "{single:e}")
        }
    } else if (1e-6..1e21).contains(&n.abs()) {
        write!(out, "{n}")
    } else {
        write!(out, "{n:e}")
    };
    r.map_err(|e| e.to_string())
}

#[inline]
fn write_i32(out: &mut String, v: i32) {
    let mut mag = if v < 0 {
        out.push('-');
        v.unsigned_abs()
    } else {
        v as u32
    };
    let mut buf = [0u8; 10];
    let mut pos = buf.len();
    while mag >= 10 {
        pos -= 1;
        buf[pos] = b'0' + (mag % 10) as u8;
        mag /= 10;
    }
    pos -= 1;
    buf[pos] = b'0' + mag as u8;
    // SAFETY: `buf[pos..]` contains only ASCII digits `b'0'..=b'9'`.
    out.push_str(std::str::from_utf8(&buf[pos..]).unwrap());
}

pub fn write_string(out: &mut String, s: &str) {
    use std::fmt::Write;
    out.push('"');
    let bytes = s.as_bytes();
    let mut last = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        let esc = match b {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            b'\n' => "\\n",
            b'\r' => "\\r",
            b'\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            0..=0x1f => {
                if last < i {
                    out.push_str(&s[last..i]);
                }
                let _ = write!(out, "\\u{:04x}", b);
                last = i + 1;
                continue;
            }
            _ => continue,
        };
        if last < i {
            out.push_str(&s[last..i]);
        }
        out.push_str(esc);
        last = i + 1;
    }
    if last < bytes.len() {
        out.push_str(&s[last..]);
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(text: &str) -> String {
        to_canonical(&parse(text.as_bytes()).unwrap()).unwrap()
    }

    #[test]
    fn objects_sort_and_indent_arrays_of_scalars_stay_inline() {
        let t = round(r#"{"b":[1,2.5,-3],"a":{"z":true,"y":null},"c":["x","y"],"d":[],"e":{}}"#);
        assert_eq!(
            t,
            "{\n  \"a\": {\n    \"y\": null,\n    \"z\": true\n  },\n  \"b\": [1, 2.5, -3],\n  \"c\": [\n    \"x\",\n    \"y\"\n  ],\n  \"d\": [],\n  \"e\": {}\n}\n"
        );
    }

    #[test]
    fn numbers_take_their_shortest_spelling() {
        let mut s = String::new();
        for n in [
            0.1f64,
            f64::from(0.1f32),
            3.0,
            -0.0,
            1e21,
            1.5e-7,
            123456.789,
            16777217.0,
            f64::from(1e-7f32),
            f64::from(1e-6f32),
            -f64::from(1e-6f32),
            f64::from(1e21f32),
        ] {
            write_number(&mut s, n).unwrap();
            s.push(' ');
        }
        assert_eq!(
            s,
            "0.1 0.1 3 0 1e21 1.5e-7 123456.789 16777217 1e-7 0.000001 -0.000001 1e21 "
        );
        assert!(write_number(&mut s, f64::NAN).is_err());
        assert!(write_number(&mut s, f64::INFINITY).is_err());
    }

    #[test]
    fn strings_escape_only_what_json_requires() {
        let mut s = String::new();
        write_string(&mut s, "a\"b\\c\nd\u{1}e/ü桜");
        assert_eq!(s, "\"a\\\"b\\\\c\\nd\\u0001e/ü桜\"");
        let v = parse(s.as_bytes()).unwrap();
        assert_eq!(v, Value::String("a\"b\\c\nd\u{1}e/ü桜".into()));
    }

    #[test]
    fn canonical_text_is_a_fixed_point() {
        let inputs = [
            r#"{"t":[0.30000001192092896,1e-9,2e30],"n":"x","m":[[1,2],[3]],"o":[{"a":1},{"b":[true,false]}]}"#,
            r#"[1, "two", {"three": 3.25}]"#,
            r#""just a string""#,
            r#"-12.5e2"#,
        ];
        for input in inputs {
            let once = round(input);
            let twice = round(&once);
            assert_eq!(once, twice, "not a fixed point: {input}");
        }
        // Values computed in single precision, written before they were ever read: the first
        // text must already be the one every later save writes.
        let edges = [
            1e-6f32,
            1e21,
            9.999999e-7,
            1.0000001e-6,
            9.9999994e20,
            1e-7,
            0.1,
        ];
        for x in edges.into_iter().flat_map(|x| [x, -x]) {
            let once = to_canonical(&Value::Array(vec![Value::Number(f64::from(x))])).unwrap();
            assert_eq!(round(&once), once, "not a fixed point: {x:e}");
        }
    }

    #[test]
    fn the_writer_refuses_exactly_what_the_reader_refuses() {
        // Small limits stand in for the real ones: the same code with smaller numbers.
        let limits = Limits {
            bytes: 96,
            depth: 3,
            values: 10,
        };
        let mut values = vec![];
        for d in 0..6 {
            // Nesting around the depth limit, ending in a scalar or in an empty object.
            let mut v = Value::Number(1.0);
            let mut e = Value::Array(vec![]);
            for _ in 0..d {
                v = Value::Array(vec![v]);
                e = Value::Object(BTreeMap::from([("k".to_string(), e)]));
            }
            values.extend([v, e]);
        }
        for n in 0..14 {
            // Counts around the value limit, inline and one per line.
            values.push(Value::Array(vec![Value::Null; n]));
            values.push(Value::Array(vec![Value::String("s".into()); n]));
            values.push(Value::Object(
                (0..n)
                    .map(|i| (format!("k{i}"), Value::Bool(true)))
                    .collect(),
            ));
        }
        for n in 85..100 {
            // Lengths around the size limit.
            values.push(Value::String("x".repeat(n)));
        }
        let mut written = 0;
        let mut refused = [0; 3];
        for v in &values {
            let text = to_canonical(v).unwrap();
            let read = parse_within(text.as_bytes(), limits);
            match canonical_within(v, limits) {
                Ok(t) => {
                    assert_eq!(t, text);
                    assert_eq!(read.as_ref(), Ok(v), "written but not readable: {text}");
                    written += 1;
                }
                Err(e) => {
                    assert!(read.is_err(), "refused ({e}) but readable: {text}");
                    let why = ["deeper", "more than", "exceeds"]
                        .iter()
                        .position(|w| e.contains(w));
                    refused[why.expect("a limit's message")] += 1;
                }
            }
        }
        assert!(
            written > 0 && refused.iter().all(|n| *n > 0),
            "{written} {refused:?}"
        );
    }

    #[test]
    fn the_real_limits_are_shared_and_hold_large_documents() {
        let nest = |d: usize| (0..d).fold(Value::Number(0.0), |v, _| Value::Array(vec![v]));
        let deepest = to_canonical(&nest(MAX_DEPTH)).unwrap();
        assert_eq!(parse(deepest.as_bytes()).unwrap(), nest(MAX_DEPTH));
        let e = to_canonical(&nest(MAX_DEPTH + 1)).unwrap_err();
        assert!(e.contains("deeper than 64"), "{e}");
        let text = format!("{}0{}", "[".repeat(65), "]".repeat(65));
        assert!(parse(text.as_bytes()).is_err());
        // More values than the reader used to take (1,000,000), as a large scene has.
        let many = Value::Array(vec![Value::Number(0.0); 1_000_001]);
        let text = to_canonical(&many).unwrap();
        assert_eq!(parse(text.as_bytes()).unwrap(), many);
        // One byte longer than the reader takes.
        let long = Value::String("x".repeat(MAX_BYTES - 2));
        let e = to_canonical(&long).unwrap_err();
        assert!(e.contains("32 MiB"), "{e}");
    }

    #[test]
    fn compact_is_one_line() {
        let v = parse(br#"{"b":[1,{"c":"d"}],"a":0.5}"#).unwrap();
        assert_eq!(
            to_compact(&v).unwrap(),
            r#"{"a": 0.5, "b": [1, {"c": "d"}]}"#
        );
    }
}
