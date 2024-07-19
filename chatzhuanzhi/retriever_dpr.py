import pytorch_lightning as pl
from transformers import DPRQuestionEncoder, DPRContextEncoder

class RetrieverDPR(pl.LightningModule):
    """
    Class of retriever model
    """
    def __init__(self):
        super().__init__()
        models_path = "/home/zzu_zxw/zjl_data/Retrieval-Augmented-Visual-Question-Answering/src/models/facebook/dpr-question_encoder-single-nq-base"
        self.query_encoder = DPRQuestionEncoder.from_pretrained(models_path)
        models_path = "/home/zzu_zxw/zjl_data/Retrieval-Augmented-Visual-Question-Answering/src/models/facebook/dpr-ctx_encoder-single-nq-base"
        self.item_encoder = DPRContextEncoder.from_pretrained(models_path)

    def resize_token_embeddings(self, dim, decoder_dim=None):
        self.query_encoder.resize_token_embeddings(dim)
        self.item_encoder.resize_token_embeddings(decoder_dim)

    def generate_query_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        query_outputs = self.query_encoder(input_ids=input_ids,attention_mask=attention_mask)
        query_last_hidden_states = query_outputs.pooler_output
        query_embeddings = query_last_hidden_states
        return query_embeddings

    def generate_item_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        item_outputs = self.item_encoder(input_ids=input_ids,attention_mask=attention_mask)
        item_last_hidden_states = item_outputs.pooler_output
        item_embeddings = item_last_hidden_states
        return item_embeddings