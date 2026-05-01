#!/usr/bin/env python3
import sys
import random

def shuffle_string(s):
    chars = list(s)
    random.shuffle(chars)
    return ''.join(chars)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: shuffle.py \"your string here\"")
        sys.exit(1)

    input_string = sys.argv[1]
    print(shuffle_string(input_string))
    