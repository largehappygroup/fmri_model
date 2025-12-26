#!/bin/bash


# look ahead by logspaced values: 0, 1, 3, 5, 10
# default has been 4
python prepare_prompts_from_keystrokes.py --look_ahead 0
python prepare_prompts_from_keystrokes.py --look_ahead 1
python prepare_prompts_from_keystrokes.py --look_ahead 3
python prepare_prompts_from_keystrokes.py --look_ahead 5
python prepare_prompts_from_keystrokes.py --look_ahead 10



