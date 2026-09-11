import pandas as pd
import pytest

from xai_core.utils import (
    InputValidationError, detect_target_column, read_evaluation_csv, resolve_target_column,
)


@pytest.mark.parametrize('columns', [['labels', 'text'], ['text', 'labels']])
def test_labels_text_order_is_not_target_order(columns):
    frame = pd.DataFrame({'labels': [1, 3], 'text': ['short', 'words']})[columns]
    assert resolve_target_column(frame) == 'labels'
    assert detect_target_column(frame) == 'labels'


def test_model_target_authoritative_and_explicit_target_validated():
    frame = pd.DataFrame({'target': [1], 'labels': [3], 'text': ['hello']})
    assert resolve_target_column(frame, model_label='labels') == 'labels'
    assert resolve_target_column(frame, 'labels', 'labels') == 'labels'
    assert resolve_target_column(frame, 'text') == 'text'
    with pytest.raises(InputValidationError, match='conflicts'):
        resolve_target_column(frame, 'text', 'labels')
    with pytest.raises(InputValidationError, match='not found'):
        resolve_target_column(frame, model_label='missing')
    with pytest.raises(InputValidationError, match='not found'):
        resolve_target_column(frame, explicit_target='missing')
    with pytest.raises(InputValidationError, match='ambiguous'):
        resolve_target_column(frame)


@pytest.mark.parametrize('frame,explicit', [
    (pd.DataFrame({'text': ['one']}), None),
    (pd.DataFrame({'labels': [], 'text': []}), None),
    (pd.DataFrame([[1, 2]], columns=['labels', 'labels']), None),
    (pd.DataFrame({'labels': [None], 'text': ['one']}), None),
    (pd.DataFrame({'labels': [' '], 'text': ['one']}), None),
    (pd.DataFrame({'labels': [1]}), ''),
    (pd.DataFrame({'labels': [1]}), '   '),
])
def test_invalid_target_data(frame, explicit):
    with pytest.raises(InputValidationError):
        resolve_target_column(frame, explicit)


def test_unlabeled_eda_does_not_guess_last_column():
    frame = pd.DataFrame({'a': [1], 'b': [2]})
    assert detect_target_column(frame) is None
    assert resolve_target_column(frame, required=False) is None


@pytest.mark.parametrize('csv', [b'labels,text,text\n1,hi,bye', b'', b',text\n1,hi'])
def test_csv_rejects_duplicate_or_empty_headers_before_pandas(csv):
    with pytest.raises(InputValidationError):
        read_evaluation_csv(csv)
