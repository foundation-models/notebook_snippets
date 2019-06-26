from generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
import base64
from IPython.core.display import display, HTML

def facets_display(data):
  HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/0.7.24/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          document.querySelector("#elem").data = data;
        </script>"""
  jsonstr = data.to_json(orient='records')
  html = HTML_TEMPLATE.format(jsonstr=jsonstr)
  print('Displaying factes plots inside Notebook. You may need different browser to view it corectly')
  src = HTML(html)
  display(src)
  return src
  
