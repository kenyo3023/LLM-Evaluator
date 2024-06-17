from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class EstimationHypothesis:
    ent:str = lambda x: x
    con:str = lambda x: 1-x # based on complementary event/probability

class SummaCZS:
    def __init__(self,
                 estimation:Literal['mean', 'min', 'max']='max',
                 aggregation:Literal['mean', 'min', 'max']='mean'):
        self.estimation = getattr(np, estimation)
        self.aggregation = getattr(np, aggregation)

    def score(self, term:np.ndarray, hypothesis:Literal['ent', 'con']='ent', return_details:bool=False):

        # Calculate term estimate
        term_score = self.estimation(term, axis=0)
        term_score = getattr(EstimationHypothesis, hypothesis)(term_score)

        # Calculate finalize estimate
        aggregation_score = self.aggregation(term_score, axis=0)

        if return_details:
            return {
                'term_score': term_score,
                'finalize_score': aggregation_score,
            }
        return aggregation_score


if __name__ == "__main__":

    from nli.summac.base import SummaCImagerForNLI

    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
    One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
    The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
    Arcadia Planitia is in Mars' northern lowlands."""
    summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
    summary2 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers."

    model_name_or_path="tals/albert-xlarge-vitaminc-mnli"
    imager = SummaCImagerForNLI(model_name_or_path)
    probs, image = imager.build_image(document, summary, batch_size=8, return_dict_or_matrix='dict')
    print(f'image shape: {image.shape}')
    print(f'image: {image}')

    term = image['entailment']
    summaczs = SummaCZS(estimation='max')
    score = summaczs.score(term=term, return_details=True)
    print(f'score: {score}')