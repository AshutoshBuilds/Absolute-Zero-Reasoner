#!/usr/bin/env python3
"""
Simple test to verify CurriculumLearningManager.current_level fix
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_curriculum_manager():
    """Test that CurriculumLearningManager has current_level property"""
    from hf_curriculum_learning import CurriculumLearningManager, create_curriculum_config
    config = create_curriculum_config()
    manager = CurriculumLearningManager(config)

    print("✅ CurriculumLearningManager initialized successfully")
    print(f"✅ current_level = '{manager.current_level}'")
    print(f"✅ current_difficulty = {manager.current_difficulty}")
    print(f"✅ current_difficulty_numeric() = {manager.current_difficulty_numeric()}")

    assert manager.current_level is not None
    assert manager.current_difficulty is not None
    assert isinstance(manager.current_difficulty_numeric(), (float, int))

    # Test that we can access the property without error
    level_name = manager.current_level
    print(f"✅ Successfully accessed current_level: '{level_name}'")

if __name__ == "__main__":
    print("🔧 Testing Curriculum Manager Fix...")
    print("=" * 50)

    if test_curriculum_manager():
        print("\n✅ SUCCESS: Curriculum manager fix is working!")
        print("The 'current_level' attribute error has been resolved.")
    else:
        print("\n❌ FAILED: Curriculum manager still has issues.")

    print("=" * 50)
