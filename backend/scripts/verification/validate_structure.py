#!/usr/bin/env python3
"""
🔍 SwaggyStacks Structure Validation
===================================
Validates the new cleaned-up project structure
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def validate_directory_structure() -> Dict[str, any]:
    """Validate the expected directory structure exists"""
    results = {
        "status": "PASSED",
        "issues": [],
        "structure": {}
    }

    project_root = Path(__file__).parent.parent.parent.parent

    # Expected structure
    expected_structure = {
        # Root level - should be clean of Python files
        "root_python_files": {
            "path": project_root,
            "expected_count": 0,
            "description": "No Python files in project root"
        },

        # Backend structure
        "backend_python_files": {
            "path": project_root / "backend",
            "expected_files": ["run_production.py"],
            "description": "Only production entry point in backend root"
        },

        # Legacy organization
        "legacy_crypto_files": {
            "path": project_root / "backend/scripts/legacy/crypto",
            "min_files": 5,
            "description": "Crypto trading files archived"
        },

        "legacy_agents_files": {
            "path": project_root / "backend/scripts/legacy/agents",
            "min_files": 5,
            "description": "Agent system files archived"
        },

        # Test organization
        "test_integration_files": {
            "path": project_root / "backend/tests/integration",
            "min_files": 3,
            "description": "Test files properly organized"
        },

        # TaskMaster structure
        "single_taskmaster": {
            "path": project_root / ".taskmaster",
            "expected": True,
            "description": "Single TaskMaster directory in root"
        },

        "no_duplicate_taskmaster": {
            "path": project_root / "backend/.taskmaster",
            "expected": False,
            "description": "No duplicate TaskMaster in backend"
        },

        # Environment files
        "env_files": {
            "paths": [
                project_root / ".env",
                project_root / ".env.example",
                project_root / ".env.production"
            ],
            "description": "Clean environment configuration"
        }
    }

    # Validate each structure requirement
    for check_name, config in expected_structure.items():
        if check_name == "root_python_files":
            py_files = list(config["path"].glob("*.py"))
            if len(py_files) != config["expected_count"]:
                results["issues"].append(f"Found {len(py_files)} Python files in root, expected {config['expected_count']}")
                results["status"] = "FAILED"
            results["structure"][check_name] = f"✅ {len(py_files)} files"

        elif check_name == "backend_python_files":
            py_files = list(config["path"].glob("*.py"))
            actual_files = [f.name for f in py_files]
            if not all(f in actual_files for f in config["expected_files"]):
                results["issues"].append(f"Missing expected files in backend: {config['expected_files']}")
                results["status"] = "FAILED"
            results["structure"][check_name] = f"✅ {len(py_files)} files ({', '.join(actual_files)})"

        elif "min_files" in config:
            if config["path"].exists():
                py_files = list(config["path"].glob("*.py"))
                if len(py_files) < config["min_files"]:
                    results["issues"].append(f"{check_name}: Expected at least {config['min_files']} files, found {len(py_files)}")
                    results["status"] = "FAILED"
                results["structure"][check_name] = f"✅ {len(py_files)} files"
            else:
                results["issues"].append(f"{check_name}: Directory {config['path']} does not exist")
                results["status"] = "FAILED"
                results["structure"][check_name] = "❌ MISSING"

        elif check_name in ["single_taskmaster", "no_duplicate_taskmaster"]:
            exists = config["path"].exists()
            if exists != config["expected"]:
                results["issues"].append(f"{check_name}: Expected {config['expected']}, found {exists}")
                results["status"] = "FAILED"
            status = "✅" if exists == config["expected"] else "❌"
            results["structure"][check_name] = f"{status} {'EXISTS' if exists else 'NOT EXISTS'}"

        elif check_name == "env_files":
            existing = [p for p in config["paths"] if p.exists()]
            if len(existing) != len(config["paths"]):
                results["issues"].append(f"Missing environment files. Expected {len(config['paths'])}, found {len(existing)}")
                results["status"] = "FAILED"
            results["structure"][check_name] = f"✅ {len(existing)}/{len(config['paths'])} files"

    return results

def validate_import_paths() -> Dict[str, any]:
    """Check that critical files have correct import paths"""
    results = {
        "status": "PASSED",
        "issues": [],
        "imports": {}
    }

    project_root = Path(__file__).parent.parent.parent.parent

    # Check production file
    prod_file = project_root / "backend/run_production.py"
    if prod_file.exists():
        content = prod_file.read_text()
        if "backend/backend" in content:
            results["issues"].append("Production file still references backend/backend")
            results["status"] = "FAILED"
            results["imports"]["production"] = "❌ BAD PATHS"
        else:
            results["imports"]["production"] = "✅ CLEAN PATHS"
    else:
        results["issues"].append("Production file not found")
        results["status"] = "FAILED"
        results["imports"]["production"] = "❌ MISSING"

    return results

def validate_no_duplicates() -> Dict[str, any]:
    """Ensure no duplicate directories exist"""
    results = {
        "status": "PASSED",
        "issues": [],
        "duplicates": {}
    }

    project_root = Path(__file__).parent.parent.parent.parent

    # Check for nested backend
    nested_backend = project_root / "backend/backend"
    if nested_backend.exists():
        results["issues"].append("Nested backend/backend directory still exists")
        results["status"] = "FAILED"
        results["duplicates"]["nested_backend"] = "❌ FOUND"
    else:
        results["duplicates"]["nested_backend"] = "✅ REMOVED"

    # Check for old backup
    old_backup = project_root / "backend/backup_before_cleanup"
    if old_backup.exists():
        results["issues"].append("Old backup directory still exists")
        results["status"] = "FAILED"
        results["duplicates"]["old_backup"] = "❌ FOUND"
    else:
        results["duplicates"]["old_backup"] = "✅ REMOVED"

    return results

def main():
    """Run all validation checks"""
    print("🔍 SwaggyStacks Structure Validation")
    print("=" * 50)

    # Run all validation checks
    checks = [
        ("Directory Structure", validate_directory_structure),
        ("Import Paths", validate_import_paths),
        ("No Duplicates", validate_no_duplicates)
    ]

    all_passed = True
    all_results = {}

    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 30)

        results = check_func()
        all_results[check_name] = results

        if results["status"] == "PASSED":
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            all_passed = False

        # Show details
        for category, items in results.items():
            if category in ["structure", "imports", "duplicates"]:
                for item, status in items.items():
                    print(f"  {item}: {status}")

        # Show issues
        if results["issues"]:
            print("  Issues:")
            for issue in results["issues"]:
                print(f"  - {issue}")

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("✅ Project structure is clean and organized")
        print("🚀 Ready for production deployment")
        return 0
    else:
        print("⚠️  VALIDATION ISSUES FOUND")
        print("❌ Please fix issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())