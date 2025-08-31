sudo apt-get purge -y nodejs npm

curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs           # installs node + npm


# 1) install nvm (loads into ~/.nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.6/install.sh | bash
exec $SHELL        # reload shell so `nvm` is on PATH

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# 2) install and activate the exact version
nvm install 22.14.0
nvm use     22.14.0
nvm alias default 22.14.0   # optional – make it the new default


