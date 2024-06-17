import re
from tqdm import tqdm
from dataclasses import dataclass

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ChunkSplitter:

    @staticmethod
    def split_to_sentence(text:str, at_least_length:int=10):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > at_least_length]
        return sentences

    @staticmethod
    def split_to_paragraph(text:str, at_least_length:int=10):
        paragraphs = re.split(r"\n\n|\n", text)
        paragraphs = [p for p in paragraphs if len(p) > at_least_length]
        return paragraphs


@dataclass
class SplitGranularity:
    paragraph :str = ChunkSplitter.split_to_paragraph
    sentence  :str = ChunkSplitter.split_to_sentence


class SummaCImager:
    def __init__(self,
                 model_name_or_path:str="tals/albert-xlarge-vitaminc-mnli",
                 document_granularity:SplitGranularity="sentence",
                 summary_granularity:SplitGranularity="sentence",
                 use_cache=True,
                 max_doc_sents=100,
                 device="cpu",
                 **kwargs):

        self.device = device
        self.model_name_or_path = model_name_or_path
        self.load_nli()

        self.entailment_idx = 0
        self.contradiction_idx = 1
        self.neutral_idx = 2
        self.channel = sum(x is not None for x in [self.entailment_idx, self.contradiction_idx, self.neutral_idx])

        self.document_granularity = document_granularity
        self.document_splitter = getattr(SplitGranularity, self.document_granularity)

        self.summary_granularity = summary_granularity
        self.summary_splitter = getattr(SplitGranularity, summary_granularity)

        self.max_doc_sents = max_doc_sents
        self.max_input_length = 500


    def load_nli(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).eval()
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()


    def create_pair_dataset(self, document_chunks:list[str], summary_chunks:list[str]):

        def count_generator(i=0):
            while True:
                yield i
                i += 1
        counter = count_generator(0)

        return [
            {
                'document': document_chunks[i],
                'summary': summary_chunks[j],
                'document_idx': i,
                'summary_idx': j,
                'pair_idx': next(counter),
            }
            for i in range(len(document_chunks))
            for j in range(len(summary_chunks))
        ]


    def build_image(self, document:str, summary:str, batch_size=20, return_dict_or_matrix:str='dict'):
        document_chunks = self.document_splitter(document)
        summary_chunks = self.summary_splitter(summary)
        pair_dataset = self.create_pair_dataset(document_chunks, summary_chunks)

        def batch_generator(dataset:list[dict], batch_size:str=20):

            dataset_size = len(dataset)
            chunk_size = (dataset_size // batch_size) + bool((dataset_size % batch_size))

            for i in range(chunk_size):
                batch = dataset[batch_size*i:batch_size*(i+1)]
                pair_data = [(data['document'], data['summary']) for data in batch]
                yield pair_data

        for i, batch in enumerate(tqdm(batch_generator(pair_dataset, batch_size=batch_size), desc='Processing')):
            batch_tokens = self.tokenizer.batch_encode_plus(batch,
                                                            padding=True,
                                                            truncation=True,
                                                            max_length=self.max_input_length,
                                                            return_tensors="pt",
                                                            truncation_strategy="only_first")
            with torch.no_grad():
                model_outputs = self.model(**batch_tokens)

            if i:
                probs = torch.cat((probs, torch.nn.functional.softmax(model_outputs["logits"], dim=-1)))
            else:
                probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)

        if return_dict_or_matrix == 'matrix':
            _shape = (self.channel, document_chunks.__len__(), summary_chunks.__len__())
            image = probs[:, [self.entailment_idx, self.contradiction_idx, self.neutral_idx]].numpy().T.reshape(_shape)
        elif return_dict_or_matrix == 'dict':
            _shape = (document_chunks.__len__(), summary_chunks.__len__())
            image = {
                'entailment': probs[:, self.entailment_idx].numpy().T.reshape(_shape),
                'contradiction': probs[:, self.contradiction_idx].numpy().T.reshape(_shape),
                'neutral': probs[:, self.neutral_idx].numpy().T.reshape(_shape),
            }
        else:
            raise ValueError(f'Invalid value \'return_dict_or_matrix\'={return_dict_or_matrix}')

        return probs, image


if __name__ == "__main__":

    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
    One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
    The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
    Arcadia Planitia is in Mars' northern lowlands."""
    summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
    summary2 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers."

    model_name_or_path="tals/albert-xlarge-vitaminc-mnli"
    imager = SummaCImager(model_name_or_path)
    probs, image = imager.build_image(document, summary, batch_size=8, return_dict_or_matrix='dict')
    print(f'image shape: {image.shape}')
    print(f'image: {image}')