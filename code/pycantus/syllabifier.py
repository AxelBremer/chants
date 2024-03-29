"""
Latin Syllabifier

Wrapper around the CLTK syllabifier, specifically adjusted for chants.

Author: Bas Cornelissen
Date: March 2019
"""
from cltk.stem.latin.syllabifier import Syllabifier, LATIN
from copy import deepcopy

class Singleton(type):
    """https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python"""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ChantSyllabifier(metaclass=Singleton):
    def __init__(self):
        latin = deepcopy(LATIN)
        exceptions = self.get_exceptions()
        latin['exceptions'] = exceptions 
        latin['diphthongs'] = ["ae", "au", "oe"] # Not: eu, ei
        latin['mute_consonants_and_f'].append('h')
        self.syllabifier = Syllabifier(latin)
    
    def get_exceptions(self):
        # See notebook "Identify syllabification errors" for background:
        # We checked the most frequent under/oversegmentations, and
        # manually corrected those

        undersegmented = {
            'euouae': ['e', 'u', 'o', 'u', 'a', 'e'],             # 29.43%
            'quia': ['qui', 'a'],                                 # 13.91%
            'seuouae': ['se', 'u', 'o', 'u', 'a', 'e'],           # 6.65%
            'israel': ['is', 'ra', 'el'],                         # 2.64%
            'cui': ['cu', 'i'],                                   # 1.74%
            'michael': ['mic', 'ha', 'el'],                       # 0.84%
            #'qui': ['qui'],                                   # 0.50%
            'requiem': ['re', 'qui', 'em'],                       # 0.41%
            'huic': ['hu', 'ic'],                                 # 0.41%
            #'jerusalem': ['je', 'ru', 'sa', 'lem'],           # 0.38%
            # 'alleluia': ['al', 'le', 'lu', 'ia'],             # 0.27%
            #'noe': ['noe'],                                   # 0.22%
            'requiescet': ['re', 'qui', 'es', 'cet'],             # 0.21%
            'exiit': ['ex', 'i', 'it'],                           # 0.17%
            'exierunt': ['ex', 'i', 'e', 'runt'],                 # 0.13%
            'eloquium': ['e', 'lo', 'qui', 'um'],                 # 0.12%
            'exiet': ['ex', 'i', 'et'],                           # 0.12%
            # 'gelboe': ['gel', 'boe'],                         # 0.11%
            'ierit': ['i', 'e', 'rit'],                           # 0.10%
            'christi': ['chris', 'ti'],                       # 0.10%
            'saul': ['sa', 'ul'],                                 # 0.09%
            'colloquiis': ['col', 'lo', 'qui', 'is'],             # 0.09%
            'israelita': ['is', 'ra', 'e', 'li', 'ta'],           # 0.09%
            'michaele': ['mic', 'ha', 'e', 'le'],                 # 0.08%
            'requiescit': ['re', 'qui', 'es', 'cit'],             # 0.08%
            'obsequia': ['ob', 'se', 'qui', 'a'],                 # 0.07%
            # 'jesus': ['je', 'sus'],                           # 0.07%
            'nicolaum': ['ni', 'co', 'laum'],                 # 0.06%
            'requies': ['re', 'qui', 'es'],                       # 0.06%
            'requiescunt': ['re', 'qui', 'es', 'cunt'],           # 0.06%
            'exierit': ['ex', 'i', 'e', 'rit'],                   # 0.06%
            'michaelis': ['mic', 'ha', 'e', 'lis'],               # 0.05%
            'requiescent': ['re', 'qui', 'es', 'cent'],           # 0.05%
        }
        
        # Recurring issues are "guen" and "quu"
        oversegmented = {
            'sanguine': ['san', 'gui', 'ne'],             # 1.45%
            'sanguinem': ['san', 'gui', 'nem'],           # 1.43%
            'lingua': ['lin', 'gua'],                     # 1.11%
            'alleluya': ['al', 'le', 'lu', 'ya'],         # 0.88%
            'sanguis': ['san', 'guis'],                   # 0.83%
            'est*': ['est*'],                             # 0.64%
            #'eleison': ['e', 'le', 'i', 'son'],               # 0.59%
            'linguis': ['lin', 'guis'],                   # 0.59%
            'linguae': ['lin', 'guae'],                   # 0.47%
            'sequuntur': ['se', 'quun', 'tur'],           # 0.42%
            'sanguinis': ['san', 'gui', 'nis'],           # 0.40%
            #'euge': ['e', 'u', 'ge'],                         # 0.29%
            'eleemosynam': ['e', 'lee', 'mo', 'sy', 'nam'],# 0.27%
            'iniquum': ['in', 'i', 'quum'],               # 0.23%
            'sunt*': ['sunt*'],                           # 0.23%
            'unguenti': ['un', 'guen', 'ti'],             # 0.21%
            'persequuntur': ['per', 'se', 'quun', 'tur'], # 0.20%
            'unguentum': ['un', 'guen', 'tum'],           # 0.20%
            'unguentorum': ['un', 'guen', 'to', 'rum'],   # 0.16%
            'urbs': ['urbs'],                             # 0.16%
            'equuleo': ['e', 'quu', 'le', 'o'],           # 0.15%
            #'perpetuum': ['per', 'pe', 'tu', 'um'],           # 0.14%
            #'antiquus': ['an', 'ti', 'qu', 'us'],             # 0.14%
            'sanguinibus': ['san', 'gui', 'ni', 'bus'],   # 0.13%
            'eleemosyna': ['e', 'lee', 'mo', 'sy', 'na'], # 0.13%
            'linguam': ['lin', 'guam'],                   # 0.13%
            'stirps': ['stirps'],                         # 0.11%
            #'ait': ['a', 'it'],                               # 0.11%
            'languores': ['lan', 'guo', 'res'],           # 0.11%
            #'jerusalem': ['je', 'ru', 'sa', 'lem'],           # 0.10%
            'loquuntur': ['lo', 'quun', 'tur'],           # 0.09%
            # 'tuum': ['tu', 'um'],                             # 0.09%
            # 'ideoque': ['i', 'de', 'o', 'que'],               # 0.09%
            'annuntiaverunt*': ['an', 'nun', 'ti', 'a', 've', 'runt*'],# 0.09%
            'linguarum': ['lin', 'gua', 'rum'],           # 0.09%
            'in*': ['in*'],                               # 0.09%
            'unguento': ['un', 'guen', 'to'],             # 0.09%
            'urguentes': ['ur', 'guen', 'tes'],           # 0.09%
            'langueo': ['lan', 'gue', 'o'],               # 0.08%
            'sanguinum': ['san', 'gui', 'num'],           # 0.08%
            'ihesum': ['ihe', 'sum'],                     # 0.08%
            'languoribus': ['lan', 'guo', 'ri', 'bus'],   # 0.07%
            'probaverunt': ['pro', 'ba', 've', 'runt'],       # 0.07%
            'faciam': ['fa', 'ci', 'am'],                     # 0.07%
            #'equum': ['e', 'qu', 'um'],                       # 0.07%
            #'jerusalem*': ['je', 'ru', 'sa', 'lem*'],         # 0.07%
            'moyses': ['moy', 'ses'],                     # 0.07%
            'pinguedine': ['pin', 'gue', 'di', 'ne'],     # 0.07%
            'linguas': ['lin', 'guas'],                   # 0.06%
            #'erue': ['e', 'ru', 'e'],                         # 0.06%
            'galaaditim': ['ga', 'laa', 'di', 'tim'],     # 0.06%
            'languentium': ['lan', 'guen', 'ti', 'um'],   # 0.05%
            'mansuetudinem': ['man', 'sue', 'tu', 'di', 'nem'],# 0.05%
            #'iniquus': ['in', 'i', 'quus'],               # 0.05%
            #'filiis': ['fi', 'li', 'is'],                     # 0.05%
            'gloria*': ['glo', 'ri', 'a*'],                   # 0.05%
            'leyson': ['ley', 'son'],                     # 0.05%
            'moysi': ['moy', 'si'],                       # 0.05%
            #'suavitatis': ['su', 'a', 'vi', 'ta', 'tis'],     # 0.05%
            'accipite': ['ac', 'ci', 'pi', 'te'],             # 0.05%
            'exsurgens*': ['ex', 'sur', 'gens*'],         # 0.05%
        }

        js_cantus_exceptions = {
            # Exceptions from the alignment algorithm used on the
            # Cantus website
            #'euouae': ['e', 'u', 'o', 'u', 'a', 'e'],
            #'seuouae': ['se', 'u', 'o', 'u', 'a', 'e'],
            #'alleluya': ['al', 'le', 'lu', 'ya'],
            'hierusalem': ['hie', 'ru', 'sa', 'lem'],
            'hiesum': ['hie', 'sum'],
            'kyrieleison': ['ky', 'ri', 'e', 'lei', 'son'],
            'xpisteleison': ['xpi', 'ste', 'lei', 'son'],
            'eleison': ['e', 'lei', 'son'],
        }

        exceptions = dict(LATIN['exceptions'], 
            **undersegmented, 
            **oversegmented,
            **js_cantus_exceptions)
        return exceptions
 
    def syllabify(self, text):
        """
        Syllabifies the (lowercased) text

        Lowercased since since otherwise CLTK doesn't work well
        """
        return self.syllabifier.syllabify(text.lower())
