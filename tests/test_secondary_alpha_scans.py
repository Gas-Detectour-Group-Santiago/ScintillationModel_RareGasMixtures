from pathlib import Path
import pandas as pd


def test_secondary_scan_configuration_alpha_only():
    path = Path(__file__).resolve().parents[1] / 'config' / 'plots' / 'secondary.csv'
    df = pd.read_csv(path, keep_default_na=False)
    scans = df.loc[df['plot_type'] == 'scan']
    assert list(scans['plot_id']) == ['alpha_vs_E', 'alpha_vs_E_over_p']
    assert set(scans['output']) == {'secondary/alpha_studies/Ar_99_CF4_1'}
    assert set(scans['series']) == {'pressure'}
    assert set(scans['y']) == {'alpha_eff_cm_inv'}
