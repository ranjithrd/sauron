# Source - https://stackoverflow.com/a/4412338
# Posted by Paul Tomblin, modified by community. See post 'Timeline' for change history
# Retrieved 2026-06-05, License - CC BY-SA 3.0

ssh ubuntu@beta << EOF
  cd /home/ubuntu/sauron
  git pull
  cd server
  docker compose up -d
EOF
