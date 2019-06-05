"""
Author: Bas Cornelissen (github.com/bacor)
Date: February 2019
"""
import re

def add_flats(volpiano, omit_notes=False):
    """
    Adds flats to all lowered notes in the scopes of accidentals
    
    In CANTUS transcriptions, flats are added only once, directly in front
    of the B. This function adds flat signs before all successive Bs until
    the next natural sign. Note that all natural signs are removed.

    This function does not assume that when a note is flattened, the notes an
    octave higher and lower are also flattened. (So it assumes that after a 
    central b flat `ij`, a lower b-flat `yb` indeed has an accidental)

    
    Flat/Nat. Note  Description
    --------------------------
    i/I       j     Central b flat
    y/Y       b     Low b flat
    z/Z       q     High b flat
    w/W       e     Low e flat
    x/X       m     High e flat
    
    Params:
    - omit_notes    if True only shows the accidental, not the actual note;
                    so accidentals function as notes
    """
    in_scope = { 'i': False, 'y': False, 'z': False, 'w': False, 'x': False}
    output = ''
    for char in volpiano:
        # If the character is a flat, enter its scope
        if char in 'iyzwx': 
            in_scope[char] = True

        # If a natural, exit the corresponding flats scope
        elif char in 'IYZWX': 
            in_scope[char.lower()] = False

        # Central b flat?
        elif in_scope['i'] and char == 'j':
            output += 'i' if omit_notes else 'ij'

        # Low b flat?
        elif in_scope['y'] and char == 'b':
            output += 'y' if omit_notes else 'yb'
        
        # High b flat?
        elif in_scope['z'] and char == 'q':
            output += 'z' if omit_notes else 'zq'

        # Low e flat?
        elif in_scope['w'] and char == 'e':
            output += 'w' if omit_notes else 'we'

        # High e flat?
        elif in_scope['x'] and char == 'm':
            output += 'x' if omit_notes else 'xm'

        # Another note
        else:
            output += char

    return output

def clean_volpiano(volpiano, allowed_chars=None, keep_boundaries=False, 
    neume_boundary=' ', syllable_boundary=' ', word_boundary=' ',
    keep_bars=False, allowed_bars='345', bar='|'):
    """
    Extracts only the allowed characters (and optionally boundaries) from a volpiano string.

    By default, the allowed characters are only notes and accidentals. The cleaning
    then amounts to removing clefs, bars, etc. The function can retain boundaries,
    if `add_boundaries=True`. Neume, syllable and word boundaries are then replaced 
    by special boundary markers (`neume_boundary`, `syllable_boundary` and `word_boundary`).

    Args:
        - volpiano: volpiano string
        - allowed_chars: string with allowed characters. Default: notes, liquescents, flats and naturals
        - keep_boundaries: boolean. If True keeps boundaries. Default: False
        - neume_boundary: string, neume boundary marker. Default: ' '
        - syllable_boundary: string, syllable boundary marker. Default: ' '
        - word_boundary: string, word boundary marker. Default: ' '
    
    Returns:
        - volpiano string, cleaned.
    """
    if not allowed_chars:
        allowed_chars = volpiano_characters('liquescents', 'notes', 'flats', 'naturals')    

    if keep_boundaries:
        # Remove dashes from the allowed characters
        allowed_chars = ''.join(c for c in allowed_chars if c not in '-')

    output = ''
    num_spaces = 0
    boundaries = { 1: neume_boundary, 2: syllable_boundary, 3: word_boundary }
    for char in volpiano:
        if char in allowed_chars:
            if num_spaces > 0:
                output += boundaries[num_spaces]
                num_spaces = 0
            
            output += char

        elif keep_boundaries and char == '-':
            num_spaces += 1
            if num_spaces == 3:
                output += word_boundary
                num_spaces = 0

        elif keep_bars and char in allowed_bars:
            output += bar
    
    # Handle spaces at the end
    if num_spaces > 0:
        output += boundaries[num_spaces]
        
    return output

def volpiano_characters(*keys):
    """
    Returns 'legal' Volpiano symbols

    The symbols are organized in several groups: bars, clefs, liquescents,
    naturals, notes, flats, spaces and others. You can pass these group
    names as optional arguments to return only those. So
    `volpiano_characters('naturals', 'flats')` return `IWXYZiwxyz`: the naturals
    and flats.
    """
    symbols = {
        'bars': '34567',
        'clefs': '12',
        'liquescents': '()ABCDEFGHJKLMNOPQRS',
        'naturals': 'IWXYZ',
        'notes': '89abcdefghjklmnopqrs',
        'flats': 'iwxyz',
        'spaces': '.,-',
        'others': "[]{¶",
    }
    if not keys:
        keys = symbols.keys()
        
    return "".join((symbols[key] for key in keys))

def has_no_notes(volpiano):
    """Tests if a volpiano string has no volpiano notes"""
    expr = '['+volpiano_characters('notes', 'liquescents', 'flats', 'naturals')+']+'
    return bool(re.search(expr, volpiano)) == False

def split_string(mystring, sep, keep_sep=True):
    """Splits a string, with an option for keeping the separator in"""

    if keep_sep == False:
        keep_sep = ''
    elif keep_sep == True:
        keep_sep = sep
    
    items = mystring.split(sep)
    for i in range(len(items) - 1):
        items[i] += keep_sep
    return items

def split_volpiano(volpiano, sep, keep_sep=True):
    """Splits a volpiano string while ignoring the final dashes"""
    # Find number of final dashes
    num_final_dashes = 0
    try:
        while volpiano[-(num_final_dashes + 1)] == '-':
            num_final_dashes += 1
    except IndexError:
        pass
    
    # Split without the final dashes; then append them to the last item
    if num_final_dashes > 0:
        volpiano = volpiano[:-num_final_dashes]

    items = split_string(volpiano, sep, keep_sep=keep_sep)
    items[-1] += '-' * num_final_dashes
    return items

def get_syllabifier():
    from syllabifier import ChantSyllabifier
    return ChantSyllabifier()

def syllabify(text):
    syllabifier = get_syllabifier()
    return syllabifier.syllabify(text)

_VOLPIANO_TO_MIDI = {
    "8": 53, # F
    "9": 55, # G
    "a": 57,
    "y": 58, # B flat
    "b": 59,
    "c": 60,
    "d": 62,
    "w": 63, # E flat
    "e": 64,
    "f": 65,
    "g": 67,
    "h": 69,
    "i": 70, # B flat
    "j": 71,
    "k": 72, # C
    "l": 74,
    "x": 75, # E flat
    "m": 76,
    "n": 77,
    "o": 79,
    "p": 81,
    "z": 82, # B flat
    "q": 83, # B
    "r": 84, # C
    "s": 86,
    
    # Liquescents
    "(": 53,
    ")": 55,
    "A": 57,
    "B": 59,
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "H": 69,
    "J": 71,
    "K": 72, # C
    "L": 74,
    "M": 76,
    "N": 77,
    "O": 79,
    "P": 81,
    "Q": 83,
    "R": 84, # C
    "S": 86, # D
    
    # Naturals
    "Y": 59, # Natural at B
    "W": 64, # Natural at E
    "I": 71, # Natural at B
    "X": 76, # Natural at E
    "Z": 83,
}

def volpiano_to_midi(volpiano, fill_na=False, skip_accidentals=False):
    """
    Translates volpiano pitches to a list of midi pitches

    All non-note characters are ignored or filled with `None`, if `fill_na=True`
    Unless `skip_accidentals=True`, accidentals are converted to midi pitches
    as well. So an i (flat at the B) becomes 70, a B flat. Or a W (a natural at
    the E) becomes 64 (E).
    """
    accidentals = volpiano_characters('flats', 'naturals')
    midi = []
    for char in volpiano:
        if skip_accidentals and char in accidentals:
            pass
        elif char in _VOLPIANO_TO_MIDI:
            midi.append(_VOLPIANO_TO_MIDI[char])
        elif fill_na:
            midi.append(None)
    return midi

# There are no intervals larger than +19 or -19 semitones,
# so this coding should suffice:
INTERVAL_TO_CODE = {
    -23: "n",
    -22: "m",
    -21: "l",
    -20: "k",
    -19: "j",
    -18: "i",
    -17: "h",
    -16: "g",
    -15: "f",
    -14: "e",
    -13: "d",
    -12: "c",
    -11: "b",
    -10: "a",
    -9:  "₉",
    -8:  "₈",
    -7:  "₇",
    -6:  "₆",
    -5:  "₅",
    -4:  "₄",
    -3:  "₃",
    -2:  "₂",
    -1:  "₁",
    0:   "0",
    1:   "¹",
    2:   "²",
    3:   "³",
    4:   "⁴",
    5:   "⁵",
    6:   "⁶",
    7:   "⁷",
    8:   "⁸",
    9:   "⁹",
    10:  "A",
    11:  "B",
    12:  "C",
    13:  "D",
    14:  "E",
    15:  "F",
    16:  "G",
    17:  "H",
    18:  "I",
    19:  "J",
    20:  "K",
    21:  "L",
    22:  "M",
    23:  "N",
    None: '.'
}

CODE_TO_INTERVAL = { code: interval for interval, code in INTERVAL_TO_CODE.items() }

def encode_intervals(intervals):
    """Encode an iterable of intervals to its string representation"""
    return "".join(INTERVAL_TO_CODE[i] for i in intervals)

def encode_contour(intervals):
    contour = []
    chars = {
        'None': '.',
        'zero': '–',
        'up': '‾',
        'down': '_'
    }
    for interval in intervals:
        if interval is None:
            contour.append(chars['None'])
        elif interval is 0:
            contour.append(chars['zero'])
        elif interval > 0:
            contour.append(chars['up'])
        elif interval < 0:
            contour.append(chars['down'])      
    return "".join(contour)

def to_intervals(notes, keep_empty=True, encode=False):
    """Convert list of (midi) notes to intervals"""
    pairs = zip(notes[:-1], notes[1:])
    get_interval = lambda pair: pair[1] - pair[0]
    intervals = list(map(get_interval, pairs))
    if keep_empty:
        intervals = [None] + intervals

    if encode:
        return encode_intervals(intervals)
    else:
        return intervals
    
def copy_segmentation(source, target, sep=' ', validate=True):
    """Copy the segmentation from a source string to the target string
    
    ```
    >>> copy_segmentation('ab cde', '12345')
    '12 345'
    >>> copy_segmentation('ab|cd|e', '12345', sep='|')
    '12|34|5'
    >>> copy_segmentation('ab cd', '123456')
    ValueError: source and target should have the same number of (non-sep) characters
    ```
    """
    # Test input
    source_chars = source.replace(sep, "")
    if not len(source_chars) == len(target):
        raise ValueError('source and target should have the same number of (non-sep) characters')
    
    # Copy segmentation
    start = 0
    target_units = []
    source_units = source.split(sep)
    for i, source_unit in enumerate(source_units):
        target_unit = target[start:start+len(source_unit)]
        target_units.append(target_unit)
        start += len(source_unit)
    
    # Validate result
    if validate:
        assert "".join(target_units) == target
    return sep.join(target_units)

def get_interval_representation(volpiano, segment=True, sep=' '):
    """Get interval representation of a volpiano string, keeping the segmentation intact.
    
    ```
    >>> get_interval_representation('ab caa b')
    '.² ¹₃0 ²'
    >>> get_interval_representation('ab|caa|b', sep='|')
    '.²|¹₃0|²'
    >>> get_interval_representation('ab|caa|b', sep='|', segment=False)
    '.²¹₃0²'
    ```
    """
    notes = volpiano_to_midi(volpiano)
    intervals = to_intervals(notes, keep_empty=True, encode=True)
    if segment:
        return copy_segmentation(volpiano, intervals, sep=sep)
    else:
        return intervals

def get_contour_representation(volpiano, segment=True, sep=' '):
    """Get the contour representation of a volpiano string; see get_interval_representation"""
    notes = volpiano_to_midi(volpiano)
    intervals = to_intervals(notes, keep_empty=True, encode=False)
    contour = encode_contour(intervals)
    if segment:
        return copy_segmentation(volpiano, contour, sep=sep)
    else:
        return contour