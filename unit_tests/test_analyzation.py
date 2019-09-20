from evaluation.analyzation import ExperimentAnalyzer


def test_init():
    analyzer = ExperimentAnalyzer('temp_unit_test')

    assert len(analyzer.hyperparams) == 4
    assert len(analyzer.log[0]) == 2
    assert len(analyzer.param_value_index_mapping[('optimizer_args', 'lr')]) == 2
    assert len(analyzer.param_value_index_mapping[('optimizer_args', 'lr')][0]) == 2
    assert analyzer.param_value_index_mapping[('optimizer_args', 'lr')][1][0] == [0, 1]
    assert analyzer.per_metric_results['original'].shape == (4, 4, 2, 1)
    probe_shape = analyzer.per_param_value_results['original'][('optimizer_args', 'lr')].shape
    assert probe_shape == (2, 4, 2, 2, 1)


def test_best_models():
    analyzer = ExperimentAnalyzer('temp_unit_test')
    analyzer.print_best_model_summary()
