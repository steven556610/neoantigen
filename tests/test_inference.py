import pandas as pd
import pytest
from cnneopp.inference import run_cnneopp_pipeline

def test_inference_pipeline_runs_without_models():
    """
    Test that the pipeline can process a dataframe gracefully
    even if the model weights (.pth) are missing (should print skips).
    """
    df = pd.DataFrame({
        'hla': ['A*02:01'],
        'peptide': ['SLYNTVATL']
    })
    
    # We pass a nonexistent dir to guarantee models aren't found,
    # expecting it to return the original DF seamlessly without crash
    result_df = run_cnneopp_pipeline(df, model_dir='non_existent_models_dir')
    
    # Basic assertions
    assert 'hla' in result_df.columns
    assert 'peptide' in result_df.columns
    assert len(result_df) == 1
    # Check that temporary columns from preprocessing are properly cleaned up
    assert 'trans' not in result_df.columns
