#!/usr/bin/env bash
# Encrypt / decrypt private content with the LAMH_KEY secret (AES-256-CBC, PBKDF2).
# The repository is public: scripts, narration, logs and finished videos only ever
# leave the runner encrypted. Only the code is readable.
#   lamh_crypt.sh pack   <dir>  <out.enc>    tar + encrypt a folder
#   lamh_crypt.sh unpack <in.enc> <dir>      decrypt + untar into a folder
#   lamh_crypt.sh text   <file>              print a file as encrypted base64 (safe for public logs)
#   lamh_crypt.sh untext <file>              decode text produced by "text"
set -euo pipefail
: "${LAMH_KEY:?LAMH_KEY is not set (add it under Settings > Secrets and variables > Actions)}"
enc() { openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -pass env:LAMH_KEY "$@"; }
case "$1" in
  pack)   tar -C "$2" -cf - . | enc -out "$3" ;;
  unpack) mkdir -p "$3"; enc -d -in "$2" | tar -C "$3" -xf - ;;
  text)   echo "::group::encrypted log"; enc -a -A -in "$2"; echo; echo "::endgroup::" ;;
  untext) tr -d '\n ' < "$2" | enc -d -a -A ;;
  *) echo "usage: $0 pack|unpack|text|untext ..." >&2; exit 2 ;;
esac
