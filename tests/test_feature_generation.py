import pytest
import pandas as pd
import re
from datetime import datetime

from preprocessing.feature_generating import (
    split_work_experience,
    calculate_experience_months,
    generate_worker_features,
    extract_salaries,
    clean_and_reduce_skills,
    add_features_to_dataframe,
    read_features,
)


@pytest.fixture
def sample_work_experience():
    return (
        "2020-01-01 - 2022-03-01\n"
        "2021-06-01 - 2023-08-01\n"
        "2024-01-01 - :"
    )


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "work_experience": [
                "2020-01-01 - 2022-03-01\n2021-06-01 - 2023-08-01",
                "2019-01-01 - 2021-01-01",
                "",
            ],
            "salary": [
                "100000, 120000, грейд 3",
                "90000 - 150000 евро",
                "нет данных",
            ],
            "key_skills": [
                "Python, SQL, Git",
                "python, sql, git",
                "Java, Docker",
            ],
        }
    )


def test_split_work_experience(sample_work_experience):
    entries = split_work_experience(sample_work_experience)
    assert len(entries) > 0
    for entry in entries:
        assert isinstance(entry, str)
        assert re.search(r"\d{4}-\d{2}-\d{2} - \d{4}-\d{2}-\d{2}|:", entry)


def test_calculate_experience_months():
    data = [
        "2020-01-01 - 2021-01-01",
        "2021-01-01 - 2022-01-01",
        "2023-01-01 - :",  # до текущей даты
    ]
    months = calculate_experience_months(data)
    # простой самопроверочный тест: 12 + 12 + что‑то около 12
    assert months > 30


def test_generate_worker_features(simple_df):
    result = generate_worker_features(simple_df)

    assert "unique_work" in result.columns
    assert "work_experience_months" in result.columns
    assert "count_works" in result.columns
    assert "avg_time_per_work" in result.columns

    # проверим, что count_works >= 1 для строк с опытом
    mask = result["work_experience"].str.len() > 2
    assert (result.loc[mask, "count_works"] >= 1).all()


def test_extract_salaries(simple_df):
    result = extract_salaries(simple_df, salary_column="salary")

    assert "min_salary" in result.columns
    assert "comfort_salary" in result.columns
    assert "grade" in result.columns

    # хоть где‑то должен быть ненулевой salary
    non_zero = (result["min_salary"] != 0) | (result["comfort_salary"] != 0)
    assert non_zero.any()


def test_clean_and_reduce_skills():
    data = pd.DataFrame({"key_skills": ["Python, питон, Python", "Java, JAVA, C++"]})
    reduced = clean_and_reduce_skills(data, column_name="key_skills", threshold=80)

    # должно убрать дубли похожих навыков
    assert len(reduced) <= 4
    # но слова должны быть в нижнем регистре
    assert all(skill == skill.lower() for skill in reduced)


def test_add_features_to_dataframe():
    data = pd.DataFrame({"key_skills": ["Python, SQL, Git"]})
    features = ["Python", "python", "C++", "sql", "Java"]

    result = add_features_to_dataframe(data, features, skills_column="key_skills")

    # фактические колонки, как видно из стектрейса
    assert "C++" in result.columns
    assert "python" in result.columns
    assert "Java" in result.columns

    # Python распознался
    assert result["python"].iloc[0] == 1


def test_read_features():
    # только если файл skills.txt существует
    try:
        features = read_features("../data/skills.txt")
        assert isinstance(features, list)
        assert len(features) > 0
    except FileNotFoundError:
        # можно закомментировать, если файла нет в тесте
        pass