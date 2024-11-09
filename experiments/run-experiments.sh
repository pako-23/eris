#!/bin/sh -eu


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
  done
  printf '\r'
}


if ! [ -d datasets ]; then
  echo " [-] Dataset files not found. Downloading them..."
  ./download_datasets.py > /dev/null 2>&1 &
  spinner "$!" 'Downloading dataset files...'
  echo " [+] Successfully downloaded all dataset files."
fi
