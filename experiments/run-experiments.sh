#!/bin/sh -u


spinner() {
  local pid="$1"
  local message="$2"  
  i=1
  sp='/-\|'
  while [ -d /proc/$pid ]; do
      i=$(expr $i + 1)
      if [ "$i" -ge '5' ]; then
	  i=1
      fi
      printf "\r [$(echo $sp | cut -c$i)] $message"
      sleep 0.1
  done
  printf '\r'
}


usage() {
    cat <<EOF
Usage: $(basename $0) [options]

Options:
  -h, --help       Print help page
EOF
}


OPTS=$(getopt --options "h" --long "help" --name "$(basename $0)" -- "$@")
if [ "$?" -ne 0 ]; then
    usage
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
	-h|--help)
	    usage
	    exit 0
	    ;;
	--)
	    shift
	    break
	    ;;
	*)
	    usage
	    exit 1
	    ;;
    esac
done


if ! [ -d data/datasets ]; then
  echo " [-] Dataset files not found. Downloading them..."
  ./data/download_datasets.py > /dev/null 2>&1 &
  pid="$!"
  spinner "$pid" 'Downloading dataset files...'

  wait "$pid"
  if [ "$?" -eq 0 ]; then
      echo ' [+] Successfully downloaded all dataset files.'
  else
      echo ' [!] Failed to download dataset files.'
      exit 1
  fi
fi
