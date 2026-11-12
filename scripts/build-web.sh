#!/usr/bin/env bash
# Builds the Ganbate web runtime & React + TypeScript player:
# 1. crates/web (ganbate-web) for wasm32-unknown-unknown (release, zero third-party Cargo crates)
# 2. web/player (React + TypeScript + Vite) into OUT_DIR (default web/dist)
# 3. web/sdk/{ganbate.js,ganbate.d.ts,character-customization.*,graph-session.*} + ganbate_web.wasm
#
#   scripts/build-web.sh [OUT_DIR]
#
# Toolchains:
# * rustup (normal): `rustup target add wasm32-unknown-unknown` once; the script then runs a
#   plain `cargo build`. Cargo links with the rust-lld that rustup ships.
# * A rustc without the wasm32 standard library (for example a distribution's rustc-1.91):
#   the script rebuilds std from source with -Z build-std. Set
#     GANBATE_WEB_RUSTC / GANBATE_WEB_CARGO   the compiler and cargo (default rustc-1.91, cargo-1.91)
#     GANBATE_WEB_RUST_SRC                    the matching rust-src `library` directory
#     GANBATE_WEB_LINKER                      a wasm linker (default wasm-ld from LLVM/lld)
#   std's own dependencies (dlmalloc and friends) come from crates.io on the first build.
# Set GANBATE_WEB_TOOLCHAIN=rustup or =build-std to skip the detection.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
raw_out=${1:-$root/web/dist}
mkdir -p "$raw_out"
out=$(cd "$raw_out" && pwd)
target_dir=${CARGO_TARGET_DIR:-$root/target}
target=wasm32-unknown-unknown

# Release settings for the module only (the workspace profile is untouched).
export CARGO_PROFILE_RELEASE_PANIC=abort
export CARGO_PROFILE_RELEASE_LTO=true
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
export CARGO_PROFILE_RELEASE_OPT_LEVEL=3
export CARGO_PROFILE_RELEASE_STRIP=symbols
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="${CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS:--C target-feature=+simd128}"

mode=${GANBATE_WEB_TOOLCHAIN:-}
if [ -z "$mode" ]; then
  if command -v rustup >/dev/null 2>&1 && rustup target list --installed 2>/dev/null | grep -qx "$target"; then
    mode=rustup
  else
    mode=build-std
  fi
fi

cd "$root"
case "$mode" in
  rustup)
    cargo build --release --locked -p ganbate-web --target "$target"
    ;;
  build-std)
    rustc_bin=${GANBATE_WEB_RUSTC:-rustc-1.91}
    cargo_bin=${GANBATE_WEB_CARGO:-cargo-1.91}
    src=${GANBATE_WEB_RUST_SRC:-$("$rustc_bin" --print sysroot)/lib/rustlib/src/rust/library}
    if [ ! -d "$src/std" ]; then
      echo "build-web: no rust-src at $src; set GANBATE_WEB_RUST_SRC to the 'library' directory of the" >&2
      echo "           rust-src matching $("$rustc_bin" --version), or use rustup" >&2
      exit 1
    fi
    RUSTC_BOOTSTRAP=1 \
    __CARGO_TESTS_ONLY_SRC_ROOT="$src" \
    RUSTC="$rustc_bin" \
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER="${GANBATE_WEB_LINKER:-wasm-ld}" \
      "$cargo_bin" build --release --locked -p ganbate-web --target "$target" \
        -Z build-std=std,panic_abort
    ;;
  *)
    echo "build-web: GANBATE_WEB_TOOLCHAIN must be rustup or build-std" >&2
    exit 1
    ;;
esac

if [ ! -d "$root/web/node_modules" ]; then
  if [ -f "$root/web/package-lock.json" ]; then
    npm --prefix "$root/web" ci --prefer-offline --no-audit --no-fund >/dev/null
  else
    npm --prefix "$root/web" install --prefer-offline --no-audit --no-fund >/dev/null
  fi
fi

npm --prefix "$root/web" run typecheck >/dev/null
node "$root/web/scripts/build-sdk.mjs" --outDir "$out" >/dev/null
npx --prefix "$root/web" vite build --config "$root/web/vite.config.ts" --outDir "$out" >/dev/null

cp "$target_dir/$target/release/ganbate_web.wasm" "$out/ganbate_web.wasm"

size=$(wc -c < "$out/ganbate_web.wasm")
echo "build-web: $out (ganbate_web.wasm $((size / 1024)) KiB, $mode)"
