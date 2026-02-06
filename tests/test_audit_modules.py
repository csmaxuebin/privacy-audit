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
        assert hasattr(mia, 'MembershipInferenceAttack')
    
    def test_mia_class_exists(self):
        """Test MIA class has required methods."""
        from audit.mia import MembershipInferenceAttack
        assert hasattr(MembershipInferenceAttack, 'compute_loss')
        assert hasattr(MembershipInferenceAttack, 'run_attack')


class TestExtractionModule:
    """Tests for Canary Extraction module."""
    
    def test_import(self):
        """Test that extraction module can be imported."""
        from audit import extraction
        assert hasattr(extraction, 'CanaryExtractor')
    
    def test_extractor_class_exists(self):
        """Test CanaryExtractor class has required methods."""
        from audit.extraction import CanaryExtractor
        assert hasattr(CanaryExtractor, 'extract_canaries')
        assert hasattr(CanaryExtractor, 'calculate_exposure')


class TestInternalSignalsModule:
    """Tests for Internal Signals module."""
    
    def test_import(self):
        """Test that internal_signals module can be imported."""
        from audit import internal_signals
        assert hasattr(internal_signals, 'InternalSignalAnalyzer')


class TestStressTestModule:
    """Tests for Stress Test module."""
    
    def test_import(self):
        """Test that stress_test module can be imported."""
        from audit import stress_test
        assert hasattr(stress_test, 'StressTester')


class TestAuditPackage:
    """Tests for audit package initialization."""
    
    def test_package_import(self):
        """Test that audit package can be imported."""
        from src import audit
        assert audit is not None
    
    def test_all_modules_accessible(self):
        """Test all audit modules are accessible from package."""
        from src.audit import (
            MembershipInferenceAttack,
            CanaryExtractor,
            InternalSignalAnalyzer,
            StressTester
        )
        assert MembershipInferenceAttack is not None
        assert CanaryExtractor is not None
        assert InternalSignalAnalyzer is not None
        assert StressTester is not None
