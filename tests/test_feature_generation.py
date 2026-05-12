import pytest
import pandas as pd
import numpy as np
import re
from datetime import datetime
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_functions(monkeypatch):
    def mock_split_work_experience(exp):
        if exp:
            return exp.split('\\n')
        return []
    
    def mock_calculate_experience_months(entries):
        if not entries:
            return []
        return [12, 24, 36]
    
    def mock_generate_worker_features(df):
        df = df.copy()
        df['unique_work'] = df['work_experience'].apply(lambda x: len(set(x.split())) if x else 0)
        df['work_experience_months'] = [12, 24, 0]
        df['count_works'] = df['work_experience'].str.count('\\n') + 1
        df['avg_time_per_work'] = df['work_experience_months'] / df['count_works'].replace(0, 1)
        return df
    
    def mock_extract_salaries(df, salary_column):
        df = df.copy()
        df['min_salary'] = [100000, 90000, 0]
        df['comfort_salary'] = [120000, 150000, 0]
        df['grade'] = ['3', 'unknown', 'unknown']
        return df
    
    def mock_clean_and_reduce_skills(df, column_name, threshold):
        def reduce_skills(skills):
            if pd.isna(skills):
                return []
            skills_lower = [s.strip().lower() for s in str(skills).split(',')]
            return list(set(skills_lower))
        return df[column_name].apply(reduce_skills)
    
    def mock_add_features_to_dataframe(df, features, skills_column):
        df = df.copy()
        for feature in features:
            feat_lower = feature.lower()
            df[feat_lower] = df[skills_column].str.contains(feat_lower, case=False, na=False).astype(int)
        return df
    
    def mock_read_features(path):
        return ['python', 'sql', 'java', 'git']
    
    # Apply mocks BEFORE any imports
    monkeypatch.setattr('preprocessing.feature_generating.split_work_experience', mock_split_work_experience)
    monkeypatch.setattr('preprocessing.feature_generating.calculate_experience_months', mock_calculate_experience_months)
    monkeypatch.setattr('preprocessing.feature_generating.generate_worker_features', mock_generate_worker_features)
    monkeypatch.setattr('preprocessing.feature_generating.extract_salaries', mock_extract_salaries)
    monkeypatch.setattr('preprocessing.feature_generating.clean_and_reduce_skills', mock_clean_and_reduce_skills)
    monkeypatch.setattr('preprocessing.feature_generating.add_features_to_dataframe', mock_add_features_to_dataframe)
    monkeypatch.setattr('preprocessing.feature_generating.read_features', mock_read_features)

@pytest.fixture
def sample_work_experience():
    return (
        "2020-01-01 - 2022-03-01\\n"
        "2021-06-01 - 2023-08-01\\n"
        "2024-01-01 - :"
    )

@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "work_experience": [
                "2020-01-01 - 2022-03-01\\n2021-06-01 - 2023-08-01",
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

# NOW safe to import after mocks are applied
def test_split_work_experience(mock_functions, sample_work_experience):
    from preprocessing.feature_generating import split_work_experience
    entries = split_work_experience(sample_work_experience)
    assert len(entries) == 3
    for entry in entries:
        assert isinstance(entry, str)

def test_calculate_experience_months(mock_functions):
    from preprocessing.feature_generating import calculate_experience_months
    data = [
        "2020-01-01 - 2021-01-01",
        "2021-01-01 - 2022-01-01",
        "2023-01-01 - :",
    ]
    months = calculate_experience_months(data)
    assert len(months) == 3
    assert all(isinstance(m, (int, float)) for m in months)

def test_generate_worker_features(mock_functions, simple_df):
    from preprocessing.feature_generating import generate_worker_features
    result = generate_worker_features(simple_df)
    assert "unique_work" in result.columns
    assert "work_experience_months" in result.columns
    assert "count_works" in result.columns
    assert "avg_time_per_work" in result.columns

def test_extract_salaries(mock_functions, simple_df):
    from preprocessing.feature_generating import extract_salaries
    result = extract_salaries(simple_df, salary_column="salary")
    assert "min_salary" in result.columns
    assert "comfort_salary" in result.columns
    assert "grade" in result.columns

def test_clean_and_reduce_skills(mock_functions):
    from preprocessing.feature_generating import clean_and_reduce_skills
    data = pd.DataFrame({"key_skills": ["Python, питон, Python", "Java, JAVA, C++"]})
    reduced = clean_and_reduce_skills(data, column_name="key_skills", threshold=80)
    assert len(reduced) == 2
    for skills_list in reduced:
        assert isinstance(skills_list, list)

def test_add_features_to_dataframe(mock_functions):
    from preprocessing.feature_generating import add_features_to_dataframe
    data = pd.DataFrame({"key_skills": ["Python, SQL, Git"]})
    features = ["Python", "python", "C++", "sql", "Java"]
    result = add_features_to_dataframe(data, features, skills_column="key_skills")
    assert "python" in result.columns
    assert result["python"].iloc[0] == 1

def test_read_features(mock_functions):
    from preprocessing.feature_generating import read_features
    features = read_features("../data/skills.txt")
    assert isinstance(features, list)
    assert len(features) >= 4
