"""
Copy from sentence-transformers repo
"""
from typing import Union, List

# same as InputExample
class PointInputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<PointInputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

class GroupInputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', 
                 center: str = None, 
                 texts: List[str] = None, 
                 labels: List[float] = None,):

        self.guid = guid
        self.center = center # it can be either a document-centeric or a query-centric.
        self.texts = texts
        self.labels = labels

    def __str__(self):
        return "<PairInputExample> texts_c: {}, texts: {}".format(
                self.center, "; ".join(self.texts)
        )

