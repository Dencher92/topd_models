
# Service logic, search endpoint:


def connect_db(creds):
    return {}


def connect_cache(creds):
    return {}


class SearchService:
    def __init__(
            self, image_embedder, text_embedder, adjective_texts, adjective_embs,
            doc_ids, doc_embs_index, photo_db_creds, cache_creds, n_adjectives
    ):
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder
        self.adjective_texts = adjective_texts
        self.adjective_embs = adjective_embs
        self.doc_ids = doc_ids
        self.doc_embs_index = doc_embs_index
        self.photo_db_connection = connect_db(photo_db_creds)
        self.cache_connection = connect_cache(cache_creds)
        self.n_adjectives = n_adjectives

    @staticmethod
    def _get_top_similar(query_embs, doc_embs, topk=100):
        similarity = (100.0 * query_embs @ doc_embs.T).softmax(dim=-1)
        values, indices = similarity.topk(topk)
        return values, indices

    @staticmethod
    def _get_adjectives_prompt(adjective_indices, adjective_texts, adjective_embs):
        prompt = ' '.join([adjective_texts[x] for x in adjective_indices])
        return prompt

    def _search_doc_index(self, query_emb, topk=100):
        _, top_doc_indices = self._get_top_similar(query_emb, self.doc_embs_index, topk)
        return top_doc_indices

    # endpoint
    def search(self, photo_id, *args, **kwargs):
        response_cache = self.cache_connection.get(photo_id)
        if response_cache:
            return response_cache

        photo_bytes = self.photo_db_connection.get(photo_id)
        photo_emb = self.image_embedder(photo_bytes)

        _, top_adjective_indices = self._get_top_similar(
            photo_emb, self.adjective_embs, self.n_adjectives
        )

        query_text_prompt = self._get_adjectives_prompt(
            top_adjective_indices, self.adjective_texts, self.adjective_embs
        )

        query_prompt_emb = self.text_embedder(query_text_prompt)

        top_doc_indices = self._search_doc_index(query_prompt_emb, kwargs['topk'])

        return [self.doc_ids[x] for x in top_doc_indices]


def fibonacci(n):
    f = [0, 1]

    for i in range(2, n + 1):
        f.append(f[i - 1] + f[i - 2])
    return f[n]


if __name__ == '__main__':
    import datetime
    for n in [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 100, 1000, 10000, 100000]:
        t = datetime.datetime.now()
        fibonacci(n)
        print(n, (datetime.datetime.now() - t))
