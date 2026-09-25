from kazrag import data


def test_corpus_filters_table_debris():
    docids = [p.docid for p in data.iter_corpus()]
    assert docids == ["1_1_1", "2_1_1", "3_1_1", "4_1_1", "5_1_1"]


def test_passage_content_prepends_title():
    p = next(data.iter_corpus())
    assert p.content.startswith("Астана. Астана —")


def test_qrels_keep_only_relevant_grades():
    qrels = data.load_qrels("test")
    assert qrels["q1"] == {"1_1_1"}
    assert all("5_1_1" not in docs for docs in qrels.values())


def test_topics_and_questions_align():
    topics = data.load_topics("test")
    questions = {q.qid: q for q in data.load_questions("test")}
    assert set(topics) == set(questions)
    assert questions["q3"].answers == ["Ұлтабарға"]
    assert questions["q3"].docids == ["4_1_1"]


def test_rc_passages_are_unique():
    passages = data.reading_comprehension_passages()
    assert sorted(p.docid for p in passages) == ["1_1_1", "3_1_1", "4_1_1"]
