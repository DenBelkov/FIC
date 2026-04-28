import pytest
import pandas as pd
import torch

from preprocessing.vectorizing import cosine_distance, process_skills


@pytest.fixture
def dummy_data():
    return pd.DataFrame(
        {
            "position": ["аналитик данных", "программист", "тестировщик"],
            "key_skills": [
                "SQL, Python, Excel",
                "Java, Git, Docker",
                "Selenium, Java",
            ],
        }
    )


@pytest.fixture
def simple_data():
    return pd.DataFrame(
        {
            "position": ["программист"],
            "key_skills": ["SQL, Python"],
        }
    )


def test_cosine_distance_smoke():
    # просто проверим, что функция не падает и возвращает число в [‑1, 1]
    text1 = "SQL"
    text2 = "аналитик данных"
    score = cosine_distance(text1, text2)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_cosine_distance_same_texts():
    text = "Python"
    score = cosine_distance(text, text)
    # для одинаковых текстов косинусное расстояние обычно близко к 1.0
    assert score >= 0.95  # допускаем небольшой шум


def test_process_skills_empty_skills(dummy_data):
    # искусственный ряд без навыков
    row = pd.Series({"position": "аналитик", "key_skills": ""})
    res = process_skills(row)

    assert res["mean_distance"] == 0.0
    assert res["count_above_05"] == 0
    assert res["min_distance"] == 0.0
    assert res["max_distance"] == 0.0
    assert res["std_distance"] == 0.0


def test_process_skills_basic(dummy_data):
    row = dummy_data.iloc[0]  # первый пример: "аналитик данных"
    res = process_skills(row)

    assert isinstance(res["mean_distance"], float)
    assert res["count_above_05"] >= 0
    assert res["min_distance"] <= res["max_distance"]

    # если навыков нет, behaved как в тесте выше
    row_empty = pd.Series({"position": row["position"], "key_skills": ""})
    res_empty = process_skills(row_empty)
    assert res_empty["mean_distance"] == 0.0


def test_process_skills_integration(simple_data):
    # применяем функцию ко всему небольшому DF
    result = simple_data.apply(
        lambda row: process_skills(row), axis=1
    )

    assert len(result) == len(simple_data)
    assert "mean_distance" in result.columns
    assert "count_above_05" in result.columns
    assert "min_distance" in result.columns
    assert "max_distance" in result.columns
    assert "std_distance" in result.columns

    # проверим, что все метрики числовые
    for col in result.columns:
        assert pd.api.types.is_numeric_dtype(result[col])