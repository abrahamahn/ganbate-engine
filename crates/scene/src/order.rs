//! Order keys: strings that sort in byte order and always leave room between two neighbours,
//! so inserting a sibling rewrites no other line of a file (fractional indexing over base 62).
//! A key never ends in the zero digit `0`; `None` stands for the start or the end.

const DIGITS: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

fn digit(c: u8) -> Option<usize> {
    DIGITS.iter().position(|d| *d == c)
}

/// Whether `key` is a valid order key.
pub fn valid(key: &str) -> bool {
    !key.is_empty() && !key.ends_with('0') && key.bytes().all(|c| digit(c).is_some())
}

/// A key strictly between `low` and `high` (`None`: unbounded on that side).
pub fn between(low: Option<&str>, high: Option<&str>) -> Result<String, String> {
    let a = low.unwrap_or("");
    for k in [low, high].into_iter().flatten() {
        if !valid(k) {
            return Err(format!("invalid order key {k:?}"));
        }
    }
    if let Some(b) = high {
        if a >= b {
            return Err(format!("order keys out of order: {a:?} >= {b:?}"));
        }
    }
    Ok(midpoint(a.as_bytes(), high.map(str::as_bytes)))
}

fn midpoint(a: &[u8], b: Option<&[u8]>) -> String {
    if let Some(b) = b {
        let mut n = 0;
        while n < b.len() && a.get(n).copied().unwrap_or(b'0') == b[n] {
            n += 1;
        }
        if n > 0 {
            let rest_a = if n < a.len() { &a[n..] } else { &[] };
            return String::from_utf8_lossy(&b[..n]).into_owned()
                + &midpoint(rest_a, Some(&b[n..]));
        }
    }
    let da = a.first().and_then(|c| digit(*c)).unwrap_or(0);
    let db = b.map_or(DIGITS.len(), |b| digit(b[0]).unwrap_or(0));
    if db - da > 1 {
        let mid = (da + db).div_ceil(2);
        return (DIGITS[mid] as char).to_string();
    }
    if let Some(b) = b.filter(|b| b.len() > 1) {
        return (b[0] as char).to_string();
    }
    let rest = if a.len() > 1 { &a[1..] } else { &[] };
    (DIGITS[da] as char).to_string() + &midpoint(rest, None)
}

/// `n` evenly spread keys between `low` and `high`, for bulk insertion (an import's children)
/// without the keys growing one digit per few appends.
pub fn spread(low: Option<&str>, high: Option<&str>, n: usize) -> Result<Vec<String>, String> {
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![between(low, high)?]);
    }
    let mid = between(low, high)?;
    let left = spread(low, Some(&mid), n / 2)?;
    let right = spread(Some(&mid), high, n - n / 2 - 1)?;
    Ok(left.into_iter().chain([mid]).chain(right).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_fall_between_their_neighbours() {
        let first = between(None, None).unwrap();
        assert!(valid(&first));
        let mut keys = vec![first];
        // Insert repeatedly at the front, the back and in the middle.
        for i in 0..300 {
            let k = match i % 3 {
                0 => between(None, Some(&keys[0])).unwrap(),
                1 => between(Some(keys.last().unwrap()), None).unwrap(),
                _ => {
                    let j = keys.len() / 2;
                    between(Some(&keys[j - 1]), Some(&keys[j])).unwrap()
                }
            };
            assert!(valid(&k), "{k}");
            keys.push(k);
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), i + 2, "duplicate key");
        }
        assert!(keys.iter().all(|k| k.len() < 40));
    }

    #[test]
    fn spread_keys_are_short_sorted_and_distinct() {
        let keys = spread(None, None, 1000).unwrap();
        assert_eq!(keys.len(), 1000);
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
        assert!(
            keys.iter().all(|k| valid(k) && k.len() <= 4),
            "{:?}",
            keys.iter().map(|k| k.len()).max()
        );
        let inner = spread(Some(&keys[10]), Some(&keys[11]), 50).unwrap();
        assert!(
            inner
                .iter()
                .all(|k| k.as_str() > keys[10].as_str() && k.as_str() < keys[11].as_str())
        );
    }

    #[test]
    fn invalid_keys_are_refused() {
        assert!(between(Some("a0"), None).is_err());
        assert!(between(Some("b"), Some("a")).is_err());
        assert!(between(Some("a"), Some("a")).is_err());
        assert!(between(Some("a-"), None).is_err());
    }
}
