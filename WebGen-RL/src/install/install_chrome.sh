#!/usr/bin/env bash
set -euxo pipefail          # stop on first error, echo each cmd

#######################################################################
# 0) OPTIONAL – switch the Ubuntu repos themselves to a CN mirror
#######################################################################
# Comment-out if you are already inside a domestic mirror image
sed -i 's@http://archive.ubuntu.com/ubuntu@http://mirrors.aliyun.com/ubuntu@g' \
       /etc/apt/sources.list

rm -f /etc/apt/sources.list.d/google-chrome.list || true

apt-get update
apt-get install -y wget gnupg ca-certificates unzip

#######################################################################
# 1) Google Chrome – download pre-mirrored .deb and install
#######################################################################
# Aliyun keeps the official chrome .deb in its “partner” pool
tmp=/tmp/google-chrome.deb
wget -q https://mirrors.aliyun.com/ubuntukylin/pool/partner/google-chrome-stable_current_amd64.deb \
     -O "$tmp"          # :contentReference[oaicite:0]{index=0}

# Pull runtime deps automatically
apt-get install -y "$tmp"
rm -f "$tmp"

google-chrome --version   # e.g. “Google Chrome 125.0.…”