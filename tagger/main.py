import pandas as pd
from sentence_transformers import SentenceTransformer
import json
from tqdm.autonotebook import tqdm
import numpy as np
import faiss


def title_tag(title_text: str) -> str:
    """Принимает название видео и подбирает по нему наиболее подходящий тег

    Args:
        title_text (str): название видео

    Returns:
        str: тег
    """
    taxonomy = pd.read_csv("baseline/IAB_tags.csv")

    model = "cointegrated/rubert-tiny2"
    dim = 768  # размер вектора эмбеддинга

    title_text = title_text.apply(
        lambda l: model.encode(l, convert_to_tensor=True).cpu().numpy()
    )

    def get_tags():
        tags = {}
        for i, row in tqdm(taxonomy.iterrows()):
            if isinstance(row["Уровень 1 (iab)"], str):
                tags[row["Уровень 1 (iab)"]] = (
                    model.encode(row["Уровень 1 (iab)"], convert_to_tensor=True)
                    .cpu()
                    .numpy()
                )  # .tolist()
            if isinstance(row["Уровень 2 (iab)"], str):
                tags[row["Уровень 1 (iab)"] + ": " + row["Уровень 2 (iab)"]] = (
                    model.encode(
                        row["Уровень 1 (iab)"] + ": " + row["Уровень 2 (iab)"],
                        convert_to_tensor=True,
                    )
                    .cpu()
                    .numpy()
                )  # .tolist()
            if isinstance(row["Уровень 3 (iab)"], str):
                tags[
                    row["Уровень 1 (iab)"]
                    + ": "
                    + row["Уровень 2 (iab)"]
                    + ": "
                    + row["Уровень 3 (iab)"]
                ] = (
                    model.encode(
                        row["Уровень 1 (iab)"]
                        + ": "
                        + row["Уровень 2 (iab)"]
                        + ": "
                        + row["Уровень 3 (iab)"],
                        convert_to_tensor=True,
                    )
                    .cpu()
                    .numpy()
                )  # .tolist()
        return tags

    tags = get_tags()
    tags_list = list(tags.keys())
    vectors = np.array(list(tags.values()))

    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)

    return tags_list[index.search(np.array([title_text]), topn)[1][0][0]]
