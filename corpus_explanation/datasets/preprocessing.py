from torchtext.data import Pipeline
import re

def remove_br_tag(token):
    return re.sub(r"br|(/><.*)|(</?(.*)/?>)|(<?(.*)/?>)|(<?/?(.*?)/?>?)", "", token)

remove_br_html_tags = Pipeline(remove_br_tag)
