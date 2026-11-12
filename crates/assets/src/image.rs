//! Image file types and sizes from their headers, without decoding: PNG (IHDR), JPEG (the first
//! start-of-frame marker) and WebP (VP8, VP8L and VP8X chunks). The cooker keeps images encoded
//! for hosts that decode them (browsers), and still checks their size against the limits.

/// The MIME type of a PNG, JPEG or WebP file.
pub fn mime(b: &[u8]) -> Option<&'static str> {
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

fn be16(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_be_bytes([
        *b.get(at)?,
        *b.get(at + 1)?,
    ])))
}
fn le16(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_le_bytes([
        *b.get(at)?,
        *b.get(at + 1)?,
    ])))
}
fn le24(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes([
        *b.get(at)?,
        *b.get(at + 1)?,
        *b.get(at + 2)?,
        0,
    ]))
}

/// Width and height of a PNG, JPEG or WebP file.
pub fn size(b: &[u8]) -> Option<(u32, u32)> {
    let (w, h) = match mime(b)? {
        "image/png" => {
            if b.get(12..16)? != b"IHDR" {
                return None;
            }
            let w = u32::from_be_bytes(b.get(16..20)?.try_into().ok()?);
            let h = u32::from_be_bytes(b.get(20..24)?.try_into().ok()?);
            (w, h)
        }
        "image/jpeg" => {
            let mut at = 2;
            loop {
                // Markers may be padded with 0xFF fill bytes.
                while *b.get(at)? == 0xFF && *b.get(at + 1)? == 0xFF {
                    at += 1;
                }
                if *b.get(at)? != 0xFF {
                    return None;
                }
                let marker = *b.get(at + 1)?;
                at += 2;
                match marker {
                    // Standalone markers carry no length.
                    0x01 | 0xD0..=0xD7 => continue,
                    // Start of frame (not DHT 0xC4, JPG 0xC8 or DAC 0xCC): precision, height,
                    // width.
                    0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                        break (be16(b, at + 5)?, be16(b, at + 3)?);
                    }
                    0xD9 | 0xDA => return None,
                    _ => at += be16(b, at)? as usize,
                }
            }
        }
        _ => {
            // RIFF size, WEBP, then the first chunk.
            match b.get(12..16)? {
                b"VP8 " => {
                    // Frame tag (3 bytes), start code 9d 01 2a, then 14-bit sizes.
                    if b.get(23..26)? != [0x9d, 0x01, 0x2a] {
                        return None;
                    }
                    (le16(b, 26)? & 0x3FFF, le16(b, 28)? & 0x3FFF)
                }
                b"VP8L" => {
                    if *b.get(20)? != 0x2F {
                        return None;
                    }
                    let bits = u32::from_le_bytes(b.get(21..25)?.try_into().ok()?);
                    ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
                }
                b"VP8X" => (le24(b, 24)? + 1, le24(b, 27)? + 1),
                _ => return None,
            }
        }
    };
    (w > 0 && h > 0).then_some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_from_headers() {
        let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
        png.extend_from_slice(&640u32.to_be_bytes());
        png.extend_from_slice(&480u32.to_be_bytes());
        assert_eq!(
            (mime(&png), size(&png)),
            (Some("image/png"), Some((640, 480)))
        );

        // SOI, an APP0 segment, padding fill, then SOF2 (progressive): 8-bit, 300 high, 200 wide.
        let jpeg = [
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00, 0xFF, 0xFF, 0xC2, 0x00, 0x11, 0x08,
            0x01, 0x2C, 0x00, 0xC8, 0x03,
        ];
        assert_eq!(size(&jpeg), Some((200, 300)));
        // A DHT segment before the frame is skipped, not taken for a frame.
        let dht = [
            0xFF, 0xD8, 0xFF, 0xC4, 0x00, 0x03, 0x00, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10,
            0x00, 0x20,
        ];
        assert_eq!(size(&dht), Some((32, 16)));

        let mut lossy = b"RIFF\0\0\0\0WEBPVP8 \0\0\0\0".to_vec();
        lossy.extend_from_slice(&[0, 0, 0, 0x9d, 0x01, 0x2a]);
        lossy.extend_from_slice(&(1024u16 | 0x4000).to_le_bytes()); // scale bits ignored
        lossy.extend_from_slice(&512u16.to_le_bytes());
        assert_eq!(
            (mime(&lossy), size(&lossy)),
            (Some("image/webp"), Some((1024, 512)))
        );

        let mut lossless = b"RIFF\0\0\0\0WEBPVP8L\0\0\0\0\x2f".to_vec();
        let bits: u32 = (100 - 1) | ((50 - 1) << 14);
        lossless.extend_from_slice(&bits.to_le_bytes());
        assert_eq!(size(&lossless), Some((100, 50)));

        let mut extended = b"RIFF\0\0\0\0WEBPVP8X\0\0\0\0\0\0\0\0".to_vec();
        extended.extend_from_slice(&[0xFF, 0x0F, 0x00, 0x7F, 0x00, 0x00]);
        assert_eq!(size(&extended), Some((4096, 128)));

        for bad in [&b"GIF89a"[..], &png[..20], &jpeg[..12], &lossy[..24], b""] {
            assert_eq!(size(bad), None);
        }
    }
}
