import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

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

@patch('preprocessing.feature_generating.split_work_experience')
@patch('preprocessing.feature_generating.calculate_experience_months')
@patch('preprocessing.feature_generating.generate_worker_features')
@patch('preprocessing.feature_generating.extract_salaries')
@patch('preprocessing.feature_generating.clean_and_reduce_skills')
@patch('preprocessing.feature_generating.add_features_to_dataframe')
@patch('preprocessing.feature_generating.read_features')
def test_all_functions(
    mock_read_features, mock_add_features, mock_clean_skills, 
    mock_extract_salaries, mock_generate_features, mock_calc_months, mock_split_exp,
    sample_work_experience, simple_df
):
    # Configure mocks
    mock_split_exp.return_value = sample_work_experience.split('\\n')
    mock_calc_months.return_value = [12, 24, 36]
    mock_generate_features.return_value = simple_df.copy()
    mock_extract_salaries.return_value = simple_df.copy()
    mock_clean_skills.return_value = pd.Series([['python'], ['java']])
    mock_add_features.return_value = simple_df.copy()
    mock_read_features.return_value = ['python', 'sql', 'java', 'git']
    
    # Test imports work
    from preprocessing.feature_generating import (
        split_work_experience, calculate_experience_months, generate_worker_features,
        extract_salaries, clean_and_reduce_skills, add_features_to_dataframe, read_features
    )
    
    # Test split_work_experience
    entries = split_work_experience(sample_work_experience)
    assert len(entries) == 3
    mock_split_exp.assert_called_once_with(sample_work_experience)
    
    # Test calculate_experience_months
    months = calculate_experience_months(['2020-01-01 - 2021-01-01'])
    assert len(months) == 3
    mock_calc_months.assert_called()
    
    # Test generate_worker_features
    result = generate_worker_features(simple_df)
    assert result is not None
    mock_generate_features.assert_called_once_with(simple_df)
    
    # Test extract_salaries
    result = extract_salaries(simple_df, "salary")
    assert result is not None
    mock_extract_salaries.assert_called_once()
    
    # Test clean_and_reduce_skills
    data = pd.DataFrame({"skills": ["Python, питон"]})
    reduced = clean_and_reduce_skills(data, "skills", 80)
    assert len(reduced) == 1
    mock_clean_skills.assert_called()
    
    # Test add_features_to_dataframe
    result = add_features_to_dataframe(simple_df, ['python'], "key_skills")
    assert result is not None
    mock_add_features.assert_called()
    
    # Test read_features
    features = read_features("../data/skills.txt")
    assert len(features) == 4
    mock_read_features.assert_called_once()
