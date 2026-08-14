'''
Exact-value tests for the evaluation practitioners, built on a hand-crafted
confusion matrix so every metric can be checked against arithmetic.
'''
import os

import numpy as np
import pandas as pd
import pytest

from project_team.io_project import Pytorch_Manager, io_traindeploy_config
from project_team.ml_project.Practitioners.ClassificationEval_Practitioner \
    import (ClassificationEval_Practitioner,
            ClassificationEval_Practitioner_config)
from project_team.ml_project.Practitioners.ROCAnalysis_Practitioner import \
    ROCAnalysis_Practitioner
from project_team.ml_project.Practitioners.Ordinal_Correlation_Practitioner \
    import (Ordinal_Correlation_Practitioner, OrdinalCor_Practitioner_config)


# Confusion matrix for class 1 as positive: TP=4, FN=1, FP=2, TN=3
CONFUSION_DF = pd.DataFrame({
    'y':      [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'pred_y': [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
})


def make_manager(tmp_path):
    cfg = io_traindeploy_config(project_folder=str(tmp_path),
                                experiment_name='exp')
    return Pytorch_Manager(cfg)


def test_specificity_formula_against_known_confusion_matrix():
    # regression 6.3: 'Spec.' used to compute precision TP/(TP+FP) = 4/6.
    # True specificity is TN/(TN+FP) = 3/5 = 0.6
    ev = ClassificationEval_Practitioner(
        ClassificationEval_Practitioner_config(classes=[0, 1]))
    ev.evaluate(CONFUSION_DF)
    res = ev.eval_results
    assert res['Spec._1'].iloc[0] == pytest.approx(0.6, abs=1e-6)
    assert res['Sens._1'].iloc[0] == pytest.approx(0.8, abs=1e-6)
    assert res['Spec._0'].iloc[0] == pytest.approx(0.8, abs=1e-6)
    assert res['Sens._0'].iloc[0] == pytest.approx(0.6, abs=1e-6)
    assert res['F1_1'].iloc[0] == pytest.approx(8 / 11, abs=1e-6)
    assert res['Acc._Overall'].iloc[0] == pytest.approx(0.7, abs=1e-6)


def test_evaluate_metric_noncontiguous_classes():
    # regression 6.4: label maps were built from range(0..max) and indexed
    # by position, so classes=[1, 2] reported class 0's numbers under 1
    shifted = CONFUSION_DF.replace({0: 1, 1: 2})
    ev = ClassificationEval_Practitioner(
        ClassificationEval_Practitioner_config(classes=[1, 2]))
    ev.evaluate(shifted)
    res = ev.eval_results
    assert res['Sens._2'].iloc[0] == pytest.approx(0.8, abs=1e-6)
    assert res['Spec._2'].iloc[0] == pytest.approx(0.6, abs=1e-6)


def test_classification_eval_applies_preprocessors():
    # regression 6.11a: pred_preprocess/gt_preprocess were accepted and
    # silently ignored
    flipped = CONFUSION_DF.copy()
    flipped['pred_y'] = 1 - flipped['pred_y']
    ev = ClassificationEval_Practitioner(
        ClassificationEval_Practitioner_config(classes=[0, 1]),
        pred_preprocess=lambda v: 1 - v)
    ev.evaluate(flipped)
    assert ev.eval_results['Acc._Overall'].iloc[0] == pytest.approx(0.7)


def test_classification_eval_input_types(tmp_path):
    ev = ClassificationEval_Practitioner(
        ClassificationEval_Practitioner_config(classes=[0, 1]))
    ev.evaluate(CONFUSION_DF.to_dict('records'))  # list input
    csv_path = tmp_path / 'results.csv'
    CONFUSION_DF.to_csv(csv_path, index=False)
    ev.evaluate(str(csv_path))  # csv path input
    with pytest.raises(TypeError, match='classification evaluator'):
        ev.evaluate(42)


def test_roc_analysis_saves_stats_and_plots(tmp_path):
    # regressions 6.8/6.9/6.7: AUC csv now saved, figure closed,
    # positive_label configurable
    manager = make_manager(tmp_path)
    df = pd.DataFrame({
        'score': [0.1, 0.2, 0.3, 0.8, 0.9, 0.95],
        'truth': ['no', 'no', 'no', 'yes', 'yes', 'yes'],
    })
    roc = ROCAnalysis_Practitioner('score', 'truth', manager,
                                   positive_label='yes')
    roc.evaluate(df)
    assert roc.roc_test['AUC'].iloc[0] == pytest.approx(1.0)
    for suffix in ('_ROCAUCStats.csv', '_ROCAUCCurve.png',
                   '_ROCAUCCurve.pdf'):
        assert os.path.isfile(os.path.join(
            manager.root, 'score_truth' + suffix)), suffix
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()  # no leaked figures


def test_roc_negative_positive_auto_detection(tmp_path):
    manager = make_manager(tmp_path)
    df = pd.DataFrame({
        'score': [0.1, 0.9, 0.2, 0.8],
        'truth': ['Negative', 'Positive', 'Negative', 'Positive'],
    })
    roc = ROCAnalysis_Practitioner('score', 'truth', manager)
    roc.evaluate(df)
    assert roc.roc_test['AUC'].iloc[0] == pytest.approx(1.0)


def test_ordinal_correlation_monotonic(tmp_path):
    manager = make_manager(tmp_path)
    cfg = OrdinalCor_Practitioner_config(
        exogenous='grade', endogenous='value', ordinal_label=['a', 'b', 'c'],
        histogram=False, boxplot=False)
    prac = Ordinal_Correlation_Practitioner(cfg, dt_processor=None,
                                            io_manager=manager)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'grade': ['a'] * 10 + ['b'] * 10 + ['c'] * 10,
        'value': np.concatenate([rng.normal(m, 0.3, 10)
                                 for m in (1.0, 2.0, 3.0)]),
    })
    prac.evaluate(df)
    stats = pd.read_csv(os.path.join(manager.root, 'Stats_grade.csv'))
    kendall = stats[stats['Test'] == 'Kendall Tau']['Coefficient'].iloc[0]
    assert kendall > 0.7
    # regression NEW-5: a perfect correlation must not divide by zero
    df_perfect = pd.DataFrame({'grade': ['a', 'b', 'c'],
                               'value': [1.0, 2.0, 3.0]})
    prac.evaluate(df_perfect)  # no ZeroDivisionError / no crash
