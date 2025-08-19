special_characters = {
    # 'À'     : '`',
    'à'     : '`',
    # '\x10À' : '~',
    '\x10à' : '~',
    '\x101' : '!',
    '\x103' : '#',
    '\x104' : '$',
    '\x105' : '%',
    '\x106' : '^',
    '\x107' : '&',   
    '\x108' : '*',
    '½'     : '-',
    '\x10»' : '+',
    '»'     : '=',
    # '\x10Ü' : '|',
    '\x10ü' : '|',
    'º'     : ';',
    '\x10º' : ':',
    '\x10Þ' : '"',
    'Þ' : '"',
    '¼'     : ',',
    '¾'     : '.',
    '\x10¼' : '<',
    '\x10¾' : '>',
    '¿'     : '/',
    '\x10¿' : '?',

    'Û'     : '[OPEN BRACKET]', # opening bracket
    'Ý'     : '[CLOSE BRACKET]', # closing bracket
    '\x11'  : '[CONTROL]', # control key
    '\x08'  : '[BACKSPACE]', # backspace
    '\r'    : '[ENTER]', # enter
    '\x10'  : '[SHIFT]', # shift
    '\x1b'  : '[ESCAPE]', # escape key
    '\x109' : '[OPEN PARENTHESIS]', # opening parenthesis 
    '\x100' : '[CLOSE PARENTHESIS]', # closing parenthesis
    'Þ'     : '[SINGLE QUOTE]', # single quote
    # '\x10Û' : '@ro', # opening brace
    '\x10û' : '[OPEN BRACE]', # opening brace
    'û' : '[OPEN BRACE]', # opening brace
    # '\x10Ý' : '@rc', # closing brace
    '\x10ý' : '[CLOSE BRACE]', # closing brace
    '\x10'  : '[CLOSE BRACE]', # closing brace
    'ý'     : '[CLOSE BRACE]', # closing brace

    '\t'    : '[TAB]', # tab
    # 'Ü'     : '@y', # backslash
    'ü'     : '[BACKSLASH]', # backslash

    
    '%'     : '[LEFT ARROW]', # left arrow
    "'"     : '[RIGHT ARROW]', # right arrow
    '&'     : '[UP ARROW]', # up arrow
    '('     : '[DOWN ARROW]'  # down arrow
}

with open("/home/zachkaras/fmri/fmri_model/midprocessing/special_character_symbols.pkl", 'wb') as f:
# with open("/Users/zacharykaras/Desktop/fmri_model/midprocessing/special_character_symbols.pkl", 'wb') as f:
    pickle.dump(special_characters, f)
import pickle

shift_chars = {
    # 'SHIFT' : '',
    '[SHIFT]`': '~',
    '[SHIFT]1': '!',
    '[SHIFT]2': '@', # planning to use @ symbol for enter
    '[SHIFT]3': '#',
    '[SHIFT]4': '$', # planning to use $ symbol for backspace
    '[SHIFT]5': '%',
    '[SHIFT]6': '^',
    '[SHIFT]7': '&',
    '[SHIFT]8': '*',
    '[SHIFT]9': '(',
    '[SHIFT]0': ')',
    
    '[SHIFT]a': 'A',
    '[SHIFT]b': 'B',
    '[SHIFT]c': 'C',
    '[SHIFT]d': 'D',
    '[SHIFT]e': 'E',
    '[SHIFT]f': 'F',
    '[SHIFT]g': 'G',
    '[SHIFT]h': 'H',
    '[SHIFT]i': 'I',
    '[SHIFT]j': 'J',
    '[SHIFT]k': 'K',
    '[SHIFT]l': 'L',
    '[SHIFT]m': 'M',
    '[SHIFT]n': 'N',
    '[SHIFT]o': 'O',
    '[SHIFT]p': 'P',
    '[SHIFT]q': 'Q',
    '[SHIFT]r': 'R',
    '[SHIFT]s': 'S',
    '[SHIFT]t': 'T',
    '[SHIFT]u': 'U',
    '[SHIFT]v': 'V',
    '[SHIFT]w': 'W',
    '[SHIFT]x': 'X',
    '[SHIFT]y': 'Y',
    '[SHIFT]z': 'Z',


    '[SHIFT]=': '+',
    '[SHIFT][': '{',
    '[SHIFT]]': '}',
    '[SHIFT][BACKSLASH]': '|',
    '[SHIFT];': ':',
    '[SHIFT][SINGLE QUOTE]': '"',
    '[SHIFT],': '<',
    '[SHIFT].': '>',
    '[SHIFT]/': '?',
}

with open("/home/zachkaras/fmri/fmri_model/midprocessing/shift_chars.pkl", 'wb') as f:
# with open("/Users/zacharykaras/Desktop/fmri_model/midprocessing/shift_chars.pkl", 'wb') as f:
    pickle.dump(shift_chars, f)
    
    