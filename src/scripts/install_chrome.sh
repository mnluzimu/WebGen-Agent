# install prereqs the base image may lack
apt-get update && \
apt-get install -y wget gnupg ca-certificates unzip

# 1) add Google’s signing key & repo
wget -qO- https://dl.google.com/linux/linux_signing_key.pub \
  | gpg --dearmor -o /usr/share/keyrings/google.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] \
  https://dl.google.com/linux/chrome/deb/ stable main" \
  > /etc/apt/sources.list.d/google-chrome.list

apt-get update

# 2) install the browser
apt-get install -y google-chrome-stable        # /usr/bin/google-chrome

# sanity check – numbers must match
google-chrome   --version

sudo sysctl -w fs.inotify.max_user_instances=128