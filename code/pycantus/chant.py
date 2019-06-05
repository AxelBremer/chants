from pandas import isnull

from .parser import parse_chant

class Chant(object):

    def __init__(self, chant_id, Cantus):
        self.Cantus = Cantus
        self.data = Cantus.chants.loc[chant_id, :]
        
        if self.volpiano is not None:
            self.parsed_chant = parse_chant(self.data)
        else:
            self.parsed_chant = None 

    def __str__(self):
        return self.incipit

    def __repr__(self):
        return f'Chant({self.incipit})'
    
    def _get(self, key, default=None):
        if isnull(self.data[key]):
            return default
        else:
            return self.data[key]

    @property
    def id(self):
        return self.data.name

    @property
    def incipit(self):
        return self._get('incipit')

    @property
    def volpiano(self):
        return self._get('volpiano')

    @property
    def mode(self):
        return self._get('mode')

    @property
    def full_text(self):
        return self._get('full_text')

    @property
    def full_text_manuscript(self):
        return self._get('full_text_manuscript')

    @property
    def position(self):
        return self._get('position')

    @property
    def cantus_id(self):
        return self._get('cantus_id')

    @property
    def office(self):
        return self._get('office')
    
    @property
    def feast(self):
        return self._get('feast')

    @property
    def genre(self):
        return self._get('genre')

    def show(self, 
        details=True, 
        volpiano=False,
        text=False, 
        full_text=False,
        full_text_manuscript=False,
        parse_params={},
        **kwargs):
        """
        """
        from IPython.core.display import display, HTML

        html = """
        <style type="text/css">
            .chant {
                border: 1px solid #ccc;
                border-radius:5px;
                padding: 1em;
                position: relative;
            }
            .details {
                margin-bottom: 1em;
            }
            .value {
                margin-right: 1em;
            }
            .prop {
                color: #999;
            }
            .incipit {
                font-weight: bold;
                margin-bottom: .5em;
            }
            .chant-id {
                font-size: .8em;
                color: #666;
                margin-top: .5em;
            }
        </style>
        """
        html += '<div class="chant">'

        if details:

            properties = []
            keys = ['cantus_id', 'feast', 'office', 'genre', 'position', 'mode']
            values = [getattr(self, key) for key in keys]
            properties.extend(zip(keys, values))
            
            props_html = ''
            for prop, value in properties:
                props_html += f'<span class="prop">{prop}</span> <span class="value">{value}</span>'

            html += f"""
                <div class="details">
                    <p class="incipit">{self.incipit}</p>
                    {props_html}
                    <a href="{self.data['drupal_path']}" target="_blank">View online</a>
                </div>"""
        
        if full_text:
            html += f'<p><strong>Full text:</strong> {self.full_text}</p>'
        if full_text_manuscript:
            html += f'<p><strong>Manuscript:</strong> {self.full_text_manuscript}</p>' 
        if volpiano:
            if self.volpiano is not None:
                html += f'<pre>{self.volpiano}</pre>'
            else:
                html += f'<pre>No volpiano</pre>'
        if self.parsed_chant is not None:
            html += self.parsed_chant.html(**kwargs)
        html += f'<p class="chant-id">{self.id}</p>'

        html += '</div>'
        display(HTML(html))

    def play(self):
        import music21
        if self.volpiano is not None:
            part = music21.volpiano.toPart(self.volpiano)
            return part.show('midi')
        else:
            return False
