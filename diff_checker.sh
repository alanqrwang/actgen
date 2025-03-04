#!/bin/bash

# Usage: ./compare_dirs.sh user@server1:/path/to/directory user@server2:/path/to/directory
# Example: ./compare_dirs.sh user1@server1:/data user2@server2:/data

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 user@server1:/path/to/directory user@server2:/path/to/directory"
    exit 1
fi

DIR1=$1
DIR2=$2

TMP_DIR=$(mktemp -d)
FILE1="$TMP_DIR/server1_files.txt"
FILE2="$TMP_DIR/server2_files.txt"

echo "Generating file checksums for $DIR1..."
ssh "$(echo "$DIR1" | cut -d':' -f1)" "find $(echo "$DIR1" | cut -d':' -f2) -type f -exec md5sum {} +" | sort > "$FILE1"

echo "Generating file checksums for $DIR2..."
ssh "$(echo "$DIR2" | cut -d':' -f1)" "find $(echo "$DIR2" | cut -d':' -f2) -type f -exec md5sum {} +" | sort > "$FILE2"

echo "Comparing directories..."
diff -u "$FILE1" "$FILE2"

if [ $? -eq 0 ]; then
    echo "✅ Directories are identical!"
else
    echo "❌ Directories differ!"
fi

# Cleanup
rm -rf "$TMP_DIR"
