"""
Unit tests for audit modules.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMIAModule:
    """Tests for MIA (Membership Inference Attack) module."""
    
    def test_import(self):
        """Test that MIA module can be imported."""
        from audit import mia
        assert mia is not None
    
    def test_mia_functions_exist(self):
        """Test MIA module has required functions."""
        from audit import mia
        assert hasattr(mia, 'membership_signal')
        assert hasattr(mia, 'loss_based_mia')
        assert hasattr(mia, 'compare_stages')


class TestExtractionModule:
    """Tests for Canary Extraction module."""
    
    def test_import(self):
        """Test that extraction module can be imported."""
        from audit import extraction
        assert extraction is not None
    
    def test_extraction_functions_exist(self):
        """Test extraction module has required functions."""
        from audit import extraction
        assert hasattr(extraction, 'logprob_of_sequence')
        assert hasattr(extraction, 'canary_extraction_test')
        assert hasattr(extraction, 'compare_extraction')


class TestInternalSignalsModule:
    """Tests for Internal Signals module."""
    
    def test_import(self):
        """Test that internal_signals module can be imported."""
        from audit import internal_signals
        assert internal_signals is not None
    
    def test_internal_signals_functions_exist(self):
        """Test internal_signals module has required functions."""
        from audit import internal_signals
        assert hasattr(internal_signals, 'compute_perplexity')
        assert hasattr(internal_signals, 'analyze_internal_signals')


class TestStressTestModule:
    """Tests for Stress Test module."""
    
    def test_import(self):
        """Test that stress_test module can be imported."""
        from audit import stress_test
        assert stress_test is not None
    
    def test_stress_test_functions_exist(self):
        """Test stress_test module has required functions."""
        from audit import stress_test
        assert hasattr(stress_test, 'run_stress_test')
        assert hasattr(stress_test, 'compare_stress_test')


class TestAuditPackage:
    """Tests for audit package initialization."""
    
    def test_package_import(self):
        """Test that audit package can be imported."""
        from src import audit
        assert audit is not None
    
    def test_submodules_accessible(self):
        """Test all audit submodules are accessible."""
        from src.audit import mia, extraction, internal_signals, stress_test
        assert mia is not None
        assert extraction is not None
        assert internal_signals is not None
        assert stress_test is not None
