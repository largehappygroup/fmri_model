import pickle

#special_characters = {
#    # 'À'     : '`',
#    'à'     : '`',
#    # '\x10À' : '~',
#    '\x10à' : '~',
#    '\x101' : '!',
#    '\x103' : '#',
#    '\x104' : '$',
#    '\x105' : '%',
#    '\x106' : '^',
#    '\x107' : '&',   
#    '\x108' : '*',
#    '½'     : '-',
#    '\x10»' : '+',
#    '»'     : '=',
#    # '\x10Ü' : '|',
#    '\x10ü' : '|',
#    'º'     : ';',
#    '\x10º' : ':',
#    '\x10Þ' : '"',
#    'Þ' : '"',
#    '¼'     : ',',
#    '¾'     : '.',
#    '\x10¼' : '<',
#    '\x10¾' : '>',
#    '¿'     : '/',
#    '\x10¿' : '?',
#
#    'Û'     : '[OPEN BRACKET]', # opening bracket
#    'Ý'     : '[CLOSE BRACKET]', # closing bracket
#    '\x11'  : '[CONTROL]', # control key
#    '\x08'  : '[BACKSPACE]', # backspace
#    '\r'    : '[ENTER]', # enter
#    '\x10'  : '[SHIFT]', # shift
#    '\x1b'  : '[ESCAPE]', # escape key
#    '\x109' : '[OPEN PARENTHESIS]', # opening parenthesis 
#    '\x100' : '[CLOSE PARENTHESIS]', # closing parenthesis
#    'Þ'     : '[SINGLE QUOTE]', # single quote
#    # '\x10Û' : '@ro', # opening brace
#    '\x10û' : '[OPEN BRACE]', # opening brace
#    'û' : '[OPEN BRACE]', # opening brace
#    # '\x10Ý' : '@rc', # closing brace
#    '\x10ý' : '[CLOSE BRACE]', # closing brace
#    '\x10'  : '[CLOSE BRACE]', # closing brace
#    'ý'     : '[CLOSE BRACE]', # closing brace
#
#    '\t'    : '[TAB]', # tab
#    # 'Ü'     : '@y', # backslash
#    'ü'     : '[BACKSLASH]', # backslash
#
#    
#    '%'     : '[LEFT ARROW]', # left arrow
#    "'"     : '[RIGHT ARROW]', # right arrow
#    '&'     : '[UP ARROW]', # up arrow
#    '('     : '[DOWN ARROW]'  # down arrow
#}
special_characters = {
    'À'     : '`',
    '\x10À' : '~',
    '\x101' : '!',
    '\x103' : '#',
    '\x104' : '$',
    '\x105' : '%',
    '\x106' : '^',
    '\x107' : '&',   
    '\x108' : '*',
    '\x109' : '(',
    '\x100' : ')',
    '½'     : '-',
    '\x10»' : '+',
    '»'     : '=',
    # '\t'    : 'TAB',
    'Û'     : '[',
    'Ý'     : ']',
    '\x10Û' : '{',
    '\x10Ý' : '}',
    'Ü'     : '\\',
    '\x10Ü' : '|',
    'º'     : ';',
    '\x10º' : ':',
    'Þ'     : '\'',
    '\x10Þ' : '"',
    # '\r'    : 'ENTER',
    '¼'     : ',',
    '¾'     : '.',
    '\x10¼' : '<',
    '\x10¾' : '>',
    '¿'     : '/',
    '\x10¿' : '?',
    '\x10'  : '<K:S>', # shift
    '\x08'  : '<K:BS>', # backspace
    '\x11'  : '<K:CTRL>', # ctrl
    '\x1b'  : '<K:ESC>', # escape
    '%'     : '<K:L>', # left arrow
    "'"     : '<K:R>', # right arrow
    '&'     : '<K:U>', # up arrow
    '('     : '<K:D>' # down arrow
}
with open("/home/zachkaras/fmri/fmri_model/midprocessing/special_character_symbols.pkl", 'wb') as f:
# with open("/Users/zacharykaras/Desktop/fmri_model/midprocessing/special_character_symbols.pkl", 'wb') as f:
    pickle.dump(special_characters, f)
    
shift_chars = {
    '`': '~',
    '1': '!',
    '2': '@', # planning to use @ symbol for enter
    '3': '#',
    '4': '$', # planning to use $ symbol for backspace
    '5': '%',
    '6': '^',
    '7': '&',
    '8': '*',
    '9': '(',
    '0': ')',
    
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D',
    'e': 'E',
    'f': 'F',
    'g': 'G',
    'h': 'H',
    'i': 'I',
    'j': 'J',
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'o': 'O',
    'p': 'P',
    'q': 'Q',
    'r': 'R',
    's': 'S',
    't': 'T',
    'u': 'U',
    'v': 'V',
    'w': 'W',
    'x': 'X',
    'y': 'Y',
    'z': 'Z',


    '=': '+',
    '[': '{',
    ']': '}',
    '[BACKSLASH]': '|',
    ';': ':',
    '[SINGLE QUOTE]': '"',
    ',': '<',
    '.': '>',
    '/': '?',
}

with open("/home/zachkaras/fmri/fmri_model/midprocessing/shift_chars.pkl", 'wb') as f:
# with open("/Users/zacharykaras/Desktop/fmri_model/midprocessing/shift_chars.pkl", 'wb') as f:
    pickle.dump(shift_chars, f)
    
    
