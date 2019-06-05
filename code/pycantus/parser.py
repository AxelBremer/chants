"""
CANTUS Chant parser

Heuristically parses a CANTUS chant in sections, words and syllables.
It tries to align text to volpiano, and tracks whether this can be done.
The parses can be exported to objects, HTML and VolText, a simple format
for text-volpiano alignments (see below).

Cantus stores different 'types of text' for every chant: the full text,
the manuscript full text and the incipit. The manuscript full text contains
section boundaries (bars) that are usually not in the full text. The parsing
function `parse_chant` identifies which of the texts to use and tries to
infer the section boundaries from the manuscript full text, and insert those
into the full text, which has a standardized spelling.

# VolText

VolText is just a plain text string with section, word and syllable boundaries
explicitly marked, using non-volpiano characters. Both the text and volpiano
contain all these boundaries. A valid VolText melody-text pair has the same
number of markers in both the text and volpiano. 

VolText can be parsed by the parser (using slightly different setting).

Author: Bas Cornelissen
Date: March 2019
"""
import re
from pandas import isnull
from itertools import zip_longest

from .volpiano import split_volpiano, has_no_notes
from .syllabifier import *


# Constants and defaults

VT_SECTION = '/'
VT_WORD = '$'
VT_SYLLABLE = '_'
VT_MISSING = '!MISSING!'
VT_EMPTY_TEXT = ''

DEFAULT_PARSING_PARAMS = {
    'split_sections': True,
    'split_syllables': True,
    'split_neumes': True,
    'keep_section_boundaries': True,
    'txt_section_sep': ' *(\||r\.) *',       # Regex for section boundaries in text
    'vol_section_sep': '[345]-*',        # Regex for section boundaries in volpiano
    'vol_word_sep': '---',                    # Separator for volpiano words
    'txt_word_sep': None,
    'vol_syllable_sep': '--',                 # Separator for volpiano syllables
    'txt_syllable_sep': None,
    'neume_sep': '-',                     # Separator for neumes 
    'missing_volpiano_marker': '---',     # String inserted for missing volpiano
    'missing_text_marker': '?',           # Inserted for missing text
    'add_hyphens': True,                  # Add hyphens between syllables of a word
    'keep_sep': True,
    'missing': VT_MISSING,
}

VOLTEXT_PARSING_PARAMS = {
    'txt_section_sep': VT_SECTION,
    'vol_section_sep': VT_SECTION,
    'vol_word_sep': VT_WORD,
    'txt_word_sep': VT_WORD,
    'vol_syllable_sep': VT_SYLLABLE,
    'txt_syllable_sep': VT_SYLLABLE,
    'keep_sep': False,
    'add_hyphens': False,
    'keep_section_boundaries': False,
}


### Helpers

def split_regex(string, pattern, keep_sep=True):
    parts = []
    start = 0
    for match in re.finditer(pattern, string):
        end = match.end() if keep_sep else match.start()
        part = string[start:end]
        parts.append(part)
        start = match.end()

    # Add possible final section (this is strange; usually the last character
    # of volpiano is a barline)
    if not start == len(string):
        part = string[start:]
        parts.append(part)

    return parts

#### Main classes

class ParsedChant(object):
    __slots__: ('volpiano', 'text', 'sections', 'parts_aligned',
                'is_complete', 'parsing_params')

    def __init__(self, volpiano, text, 
        parse=True,
        **kwargs):
        """
        """
        if not (type(volpiano) is str) or (volpiano is None):
            raise ValueError('Volpiano should be a string or None')
        if not (type(text) is str) or (text is None):
            raise ValueError('Text should be a string or None')

        self.volpiano = volpiano
        self.text = text
        self.sections = []
        self.parts_aligned = None
        self.is_complete = None
        self.parsing_params = dict(DEFAULT_PARSING_PARAMS, **kwargs)
        if parse: self.parse()
    
    def parse(self, **kwargs):
        params = dict(self.parsing_params, **kwargs)
        split_sections = params['split_sections']
        txt_bar_regex = params['txt_section_sep']
        vol_bar_regex = params['vol_section_sep']
        sect_bounds = params['keep_section_boundaries']

        if split_sections:
            # Split volpiano and text at barlines using a regex
            txt_sections = split_regex(self.text, txt_bar_regex, keep_sep=False)
            vol_sections = split_regex(self.volpiano, vol_bar_regex, keep_sep=sect_bounds)
        
        self.parts_aligned = len(txt_sections) == len(vol_sections)
        
        # If the number of text and volpiano sections is different, don't
        # use sections at all!
        if not split_sections: # or not self.parts_aligned:
            txt_sections = [self.text]
            vol_sections = [self.volpiano]

        self.is_complete = True
        for vol, txt in zip_longest(vol_sections, txt_sections):
            section = Section(vol, txt, **params)
            self.sections.append(section)
            self.is_complete = self.is_complete and section.is_complete

    def export(self):
        chant = dict(sections=[])
        for section in self.sections:
            exported_section = section.export()
            chant['sections'].append(exported_section)
        return chant

    def html(self, 
        font_size=14, 
        section_boundaries=False,
        word_boundaries=False, 
        neume_boundaries=False, 
        syllable_boundaries=False):
        """"""
        html = """<style type="text/css">
            .parsed-chant {
                line-height: 1;
                position: relative;
            }
            .section {
                display: inline;
            }
            .section.boundary {
                display: inline-block;
                border: 0;
                border-bottom: 1px dashed black;
                margin-bottom: 1em;
            }
            .volpiano {
                font-family: volpiano;
                font-size: 2.5em;
            }
            .word, .syllable, .neume {
                display: inline-block;
            }
            .word {
                height: 5em;
                margin: .75em 0;
            }
            .word.boundary {
                border-left: 1px solid #00c;
                border-right: 1px solid #00c;
            }
            .syllable.boundary {
                border-color: #c00;
            }
            .neume.boundary {
                border-color: #fff;
                border-width: 5px;
                margin: 0px;
                padding-right: 1px;
                border-left:0;
            }
            .text {
                position: absolute;
                margin-top: 1em;
                background: #fff;
            }
            .boundary {
                padding-right: 3px;
                padding-left: 1px;
                margin: 0 4px;
                border-left: 1px solid #999;
                border-right: 1px solid #999;
            }
            .section.incomplete {
                color: hsl(10, 30%, 40%);
            }
            word.incomplete {
                color: hsl(10, 50%, 40%);
            }
            .syllable.incomplete {
                color: hsl(10, 100%, 40%);
            }
            .syllable.incomplete .volpiano {
                background: hsl(10, 50%, 95%);
            }
            p.status {
                font-weight:normal;
                color: #e88;
            }
            </style>
        """
        html += f'<div class="parsed-chant" style="font-size:{font_size}px">'
        for section in self.sections:
            html += section.html(
                section_boundaries=section_boundaries,
                word_boundaries=word_boundaries, 
                neume_boundaries=neume_boundaries, 
                syllable_boundaries=syllable_boundaries)
        html += '</div>'
        return html

    def voltext(self, verify_sections=True, verify_words=True, verify_syllables=True):
        volpiano = ''
        text = ''
        for i, section in enumerate(self.sections):
            vol, txt = section.voltext()
            volpiano += vol
            text += txt
            if i < len(self.sections) - 1:
                volpiano += VT_SECTION
                text += VT_SECTION
        
        if verify_sections and not volpiano.count(VT_SECTION) == text.count(VT_SECTION):
            raise Warning('Number of sections do not match')
        if verify_words and not volpiano.count(VT_WORD) == text.count(VT_WORD):
            raise Warning('Number of words do not match')
        if verify_syllables and not volpiano.count(VT_SYLLABLE) == text.count(VT_SYLLABLE):
            raise Warning('Number of syllables do not match')
            
        return volpiano, text

class Section(object):

    # https://stackoverflow.com/questions/1336791/dictionary-vs-object-which-is-more-efficient-and-why
    __slots__: ('volpiano', 'text', 'words', 'is_complete', 
                'parts_aligned', 'parsing_params')

    def __init__(self, volpiano, text, parse=True, **kwargs):
        self.volpiano = volpiano
        self.text = text
        self.words = []
        self.is_complete = None
        self.parts_aligned = None
        self.parsing_params = dict(DEFAULT_PARSING_PARAMS, **kwargs)

        self.incomplete_manuscript = False
        if text is not None:
            if re.search('^ *~', self.text):
                self.incomplete_manuscript = True
        
        if parse: self.parse()

    def parse(self, **kwargs):
        self.words = []
        params = dict(self.parsing_params, **kwargs)
        vol_sep = params['vol_word_sep']
        txt_sep = params['txt_word_sep']
        keep_sep = params['keep_sep']

        # Todo: handle 
        # Any text that doesn't align with pitches starts with ~
        if False:
            pass
            # if re.search('^ *~', self.text):
            #     params['split_syllables'] = False
            #     word = Word(self.volpiano, self.text, parse=True, **params)
            #     self.words.append(word)
            #     self.parts_aligned = True
            #     self.is_complete = True
        
        else:
            if self.text is None:
                txt_words = []
            else:
                txt_words = self.text.split(txt_sep)

            if self.volpiano is None:
                vol_words = []
            else:
                vol_words = split_volpiano(self.volpiano, vol_sep, keep_sep=keep_sep)
            
            # Insert empty words in the text when there are no notes in the volpiano
            num_txt_words = len(txt_words)
            num_vol_words = 0
            EMPTY_TEXT = ''
            for i in range(len(vol_words)):
                if has_no_notes(vol_words[i]):
                    # Insert empty word only if there is none
                    if i >= len(txt_words):
                        txt_words.insert(i, EMPTY_TEXT)
                    elif not txt_words[i] == EMPTY_TEXT:
                        txt_words.insert(i, EMPTY_TEXT)
                else:
                    num_vol_words += 1
                    if i >= len(txt_words):
                        # More txt_words than neume_words? Append None for every txt_word=
                        txt_words.insert(i, None)

            # Align words, return false if cannot be aligned
            self.parts_aligned = num_vol_words == num_txt_words
            self.is_complete = True
            for word_vol, word_txt in zip_longest(vol_words, txt_words):
                word = Word(word_vol, word_txt, parse=True, **params)
                self.words.append(word)
                self.is_complete = self.is_complete and word.is_complete

    def export(self):
        section = dict(words=[], is_complete=self.is_complete, parts_aligned=self.parts_aligned)
        for word in self.words:
            exported_word = word.export()
            section['words'].append(exported_word)
        return section

    def html(self, 
        section_boundaries=False,
        word_boundaries=False,
        syllable_boundaries=False, 
        neume_boundaries=False):
        """"""
        section_cls = ' boundary' if section_boundaries else ''
        section_cls += ' incomplete' if not self.is_complete else ''
        html = f'<div class="{section_cls} section">'
        for word in self.words:
            html += word.html(word_boundaries=word_boundaries, 
                neume_boundaries=neume_boundaries, syllable_boundaries=syllable_boundaries)
        html += '</div>'
        return html

    def voltext(self):
        volpiano = ''
        text = ''
        for i, word in enumerate(self.words):
            vol, txt = word.voltext()
            volpiano += vol
            text += txt
            
            if i < len(self.words) - 1:
                volpiano += VT_WORD
                text += VT_WORD
        return volpiano, text

class Word(object):
    __slots__: ('volpiano', 'text', 'syllables', 'is_complete', 
                'parts_aligned', 'parsing_params')

    def __init__(self, volpiano, text, 
        parse=True,
        **kwargs):
        """"""
        assert type(volpiano) == str or volpiano is None
        assert type(text) == str or text is None
        self.volpiano = volpiano
        self.text = text
        self.syllables = []
        self.is_complete = None
        self.parts_aligned = None
        self.parsing_params = dict(DEFAULT_PARSING_PARAMS, **kwargs)
        if parse:
            self.parse()

    def parse(self, **kwargs):
        self.syllables = []
        params = dict(self.parsing_params, **kwargs)
        vol_sep = params['vol_syllable_sep']
        txt_sep = params['txt_syllable_sep']
        keep_sep = params['keep_sep']

        # Something is missing: only add a single syllable
        if (not params['split_syllables']
            or (self.text is None or self.text == params['missing'])
            or (self.volpiano is None or self.volpiano == params['missing'])
            or (self.text == '')):
            
            # Support for Voltext
            if self.text == params['missing']:
                self.text = None
            elif txt_sep is not None and self.text is not None:
                self.text = self.text.replace(txt_sep, '')
            
            if self.volpiano == params['missing']:
                self.volpiano = None
            elif vol_sep is not None and self.volpiano is not None:
                self.volpiano = self.volpiano.replace(vol_sep, '')

            # Create single syllable
            syllable = Syllable(self.volpiano, self.text, is_word_final=True, **params)
            self.syllables.append(syllable)
            
            # If the text is empty, that's fine.
            if self.text == '':
                self.parts_aligned = True
                self.is_complete = syllable.is_complete    
            else:                
                self.parts_aligned = False
                self.is_complete = False

        # Or all is fine
        else:
            # Split volpiano/text in syllables
            vol_syllables = split_volpiano(self.volpiano, vol_sep, keep_sep=keep_sep)
            if txt_sep is None: 
                syllabifier = ChantSyllabifier()
                txt_syllables = syllabifier.syllabify(self.text)
            else:
                txt_syllables = self.text.split(txt_sep)
            self.parts_aligned = len(vol_syllables) == len(txt_syllables)

            # Parse all syllables
            self.is_complete = True
            syll_alignment = list(zip_longest(vol_syllables, txt_syllables))
            for j, (vol, txt) in enumerate(syll_alignment):
                is_word_final = j == len(syll_alignment) - 1
                syllable = Syllable(vol, txt, is_word_final=is_word_final, **params)
                self.syllables.append(syllable)
                self.is_complete = self.is_complete and syllable.is_complete

    def export(self):
        word = dict(syllables=[], parts_aligned=self.parts_aligned, is_complete=self.is_complete)
        for syllable in self.syllables:
            exported_syllable = syllable.export()
            word['syllables'].append(exported_syllable)
        return word

    def html(self, 
        word_boundaries=False, 
        neume_boundaries=False, 
        syllable_boundaries=False):
        """"""
        word_class = 'boundary' if word_boundaries else ''
        word_class += ' incomplete' if not self.is_complete else ''
        html = f'<div class="word {word_class}">'
        for syllable in self.syllables:
            html += syllable.html(neume_boundaries=neume_boundaries, syllable_boundaries=syllable_boundaries)
        html += '</div>'
        return html

    def voltext(self):
        volpiano = ''
        text = ''
        for i, syllable in enumerate(self.syllables):
            vol, txt = syllable.voltext()
            volpiano += vol
            text += txt
            
            if i < len(self.syllables) - 1:
                volpiano += VT_SYLLABLE
                text += VT_SYLLABLE

        return volpiano, text

class Syllable(object):
    __slots__: ('volpiano', 'text', 'neumes', 'is_word_final', 'text_missing',
                'text_empty', 'volpiano_missing', 'is_complete')

    def __init__(self, volpiano, text, is_word_final=False, **kwargs):
        # Parsing params
        params = dict(DEFAULT_PARSING_PARAMS, **kwargs)
        add_hyphens = params['add_hyphens']
        split_neumes = params['split_neumes']
        missing_volpiano_marker = params['missing_volpiano_marker']
        missing_text_marker = params['missing_text_marker']
        neume_sep = params['neume_sep']
        keep_sep = params['keep_sep']

        # Flags
        self.is_word_final = is_word_final
        self.text_missing = (text is None) or (text == params['missing'])
        self.text_empty = text == ''
        self.volpiano_missing = (volpiano is None) or (volpiano == params['missing']) 
        self.is_complete = not (self.text_missing or self.volpiano_missing)

        # Set text and volpiano
        self.text = missing_text_marker if self.text_missing else text
        self.volpiano = missing_volpiano_marker if self.volpiano_missing else volpiano
        self.neumes = []

        # Add dashes after text syllables
        if add_hyphens and (not self.is_word_final):
            self.text += '-'

        # Segment volpiano in neumes
        if split_neumes and (not has_no_notes(self.volpiano)):
            self.neumes = split_volpiano(self.volpiano, neume_sep, keep_sep=keep_sep)# todo 

    def export(self):
        syllable = {
            'volpiano': self.volpiano,
            'text': self.text,
            'is_complete': self.is_complete
        }
        if self.volpiano_missing:
            syllable['volpiano_missing'] = True
        if self.text_missing:
            syllable['text_missing'] = True
        return syllable

    def html(self, neume_boundaries=False, syllable_boundaries=False):
        # CSS classes
        neume_class = 'boundary' if neume_boundaries else ''
        syll_class = 'boundary' if syllable_boundaries else ''
        syll_class += ' incomplete' if self.text_missing or self.volpiano_missing else ''
        txt_class = 'text-empty' if self.text_empty else ''
        txt_class += ' text-missing' if self.text_missing else ''
        vol_class = 'volpiano-missing' if self.volpiano_missing else ''

        # Volpiano
        if len(self.neumes) > 0:
            volpiano_html = ''
            for neume in self.neumes:
                volpiano_html += f'<div class="neume {neume_class}">{neume}</div>'
        else:
            volpiano_html = self.volpiano
            
        # Complete html
        html = f"""<div class="syllable {syll_class}">
                    <div class="volpiano {vol_class}">{volpiano_html}</div>
                    <div class="text {txt_class}">{self.text}</div>
                </div>"""
        return html

    def voltext(self):
        # Volpiano
        if self.volpiano_missing:
            volpiano = VT_MISSING
        else:
            volpiano = self.volpiano
            
        # Text
        if self.text_missing:
            text = VT_MISSING
        elif self.text == '':
            text = VT_EMPTY_TEXT
        else:
            text = self.text
        return volpiano, text

####

def insert_barlines(text, full_text_manuscript, fill_missing_words=True):
    words = text.split()
    manuscript_words = full_text_manuscript.split()
    
    bar_regex = '\w*\|\w*'
    BARLINE = '|'
    for i, manuscript_word in enumerate(manuscript_words):
        # CASE 1: insert barline
        if re.search(bar_regex, manuscript_word):
            if i < len(words):
                if re.search(bar_regex, words[i]):
                    # Replace barline symbol if there's one already there
                    words[i] = BARLINE
                else:
                    # Else insert barline
                    words.insert(i, BARLINE)
            else:
                words.insert(i, BARLINE)

        # CASE 2: Add missing words from manuscript
        elif i >= len(words) and fill_missing_words:
            words.insert(i, manuscript_word)
            
        # CASE 3: Add placeholders for all missing words
        elif i >= len(words):
            words.insert(i, MISSING_WORD)
    
    return " ".join(words)

def parse_chant(chant, 
    fix_alternative_bounds=True,
    **params):
    """"""
    has_text = not isnull(chant['full_text'])
    has_text_manuscript = not isnull(chant['full_text_manuscript'])
    has_incipit = not isnull(chant['incipit'])
    has_volpiano = not isnull(chant['volpiano'])

    # Determine volpiano
    if not has_volpiano:
        volpiano = None
    else:
        volpiano = chant['volpiano']

    # Determine text
    if has_text:
        if has_text_manuscript:
            text = insert_barlines(chant['full_text'], chant['full_text_manuscript'])
        else:
            text = chant['full_text']
    elif has_incipit:
        text = chant['incipit']
    else:
        text = None

    # Determine parsing params
    if volpiano is not None and fix_alternative_bounds and not re.search('---', volpiano):
        # If there are no word boundaries,
        # Use syllable boundaries as word boundaries.
        SYLL = VT_SYLLABLE
        WORD = VT_WORD

        # Replace separators
        volpiano = chant.volpiano.replace('--', WORD)
        volpiano = volpiano.replace('-', SYLL)
        
        # Add original separators in front
        volpiano = volpiano.replace(WORD, '---' + WORD)
        volpiano = volpiano.replace(SYLL, '--' + SYLL)

        if volpiano.endswith(SYLL):
            volpiano = volpiano[:-len(SYLL)]
        params['vol_word_sep'] =  WORD
        params['vol_syllable_sep'] = SYLL
        params['keep_sep'] = False
    
    # Correct: 2 instead of 3 hyphens after the clef
    if re.search('^1--[^-]', volpiano):
        volpiano = '1-' + volpiano[1:]

    return ParsedChant(volpiano, text, **params)

def parse_voltext(volpiano, text):
    return ParsedChant(volpiano, text, **VOLTEXT_PARSING_PARAMS)