#!/usr/bin/env python3
"""
Test runner script for Swaggy Stacks backend test suite
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def run_test_suite(test_type="all", coverage=True, verbose=False, fail_fast=False):
    """Run the specified test suite"""
    
    # Base pytest command
    cmd_parts = ["pytest"]
    
    # Add test path based on type
    if test_type == "unit":
        cmd_parts.append("tests/unit/")
    elif test_type == "integration":
        cmd_parts.append("tests/integration/")
    elif test_type == "trading":
        cmd_parts.append("tests/trading/")
    elif test_type == "mcp":
        cmd_parts.append("tests/mcp/")
    elif test_type == "all":
        cmd_parts.append("tests/")
    else:
        cmd_parts.append(f"tests/{test_type}/")
    
    # Add options
    if verbose:
        cmd_parts.append("-v")
    
    if fail_fast:
        cmd_parts.append("--maxfail=1")
    
    if coverage and test_type in ["all", "unit"]:
        cmd_parts.extend([
            "--cov=app",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=html:htmlcov"
        ])
    
    # Additional useful flags
    cmd_parts.extend([
        "--tb=short",  # Shorter traceback format
        "-ra",  # Show short test summary for all except passed
        "--strict-markers",  # Strict marker validation
    ])
    
    cmd = " ".join(cmd_parts)
    return run_command(cmd, f"Running {test_type} tests")


def run_linting():
    """Run code linting"""
    commands = [
        ("black --check app/", "Code formatting check"),
        ("isort --check-only app/", "Import sorting check"),
        ("flake8 app/", "Linting check"),
        ("mypy app/", "Type checking")
    ]
    
    all_passed = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            all_passed = False
    
    return all_passed


def run_security_tests():
    """Run security-focused tests"""
    cmd = "pytest tests/ -m security -v"
    return run_command(cmd, "Security tests")


def run_performance_tests():
    """Run performance tests"""
    cmd = "pytest tests/ -m performance -v --tb=line"
    return run_command(cmd, "Performance tests")


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n📊 Generating test coverage report...")
    
    # Generate HTML coverage report
    if os.path.exists("htmlcov/index.html"):
        print("✅ HTML coverage report generated: htmlcov/index.html")
    
    # Generate XML coverage report for CI
    if os.path.exists("coverage.xml"):
        print("✅ XML coverage report generated: coverage.xml")
    
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Swaggy Stacks Test Runner")
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "trading", "mcp"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--no-coverage", "--nc",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--lint-only", "-l",
        action="store_true",
        help="Run only linting checks"
    )
    
    parser.add_argument(
        "--security", "-s",
        action="store_true",
        help="Run security tests only"
    )
    
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Run performance tests only"
    )
    
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full test suite including linting and security"
    )
    
    args = parser.parse_args()
    
    print("🧪 Swaggy Stacks Backend Test Suite")
    print("=" * 60)
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    success = True
    
    if args.lint_only:
        success = run_linting()
    elif args.security:
        success = run_security_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.full:
        # Run complete test suite
        success &= run_linting()
        success &= run_test_suite(
            test_type=args.type,
            coverage=not args.no_coverage,
            verbose=args.verbose,
            fail_fast=args.fail_fast
        )
        success &= run_security_tests()
        success &= generate_test_report()
    else:
        # Run specified tests
        success = run_test_suite(
            test_type=args.type,
            coverage=not args.no_coverage,
            verbose=args.verbose,
            fail_fast=args.fail_fast
        )
        
        if success and not args.no_coverage:
            generate_test_report()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    
    if success:
        print("🎉 All tests passed successfully!")
        print("\n💡 Next steps:")
        if args.type == "all":
            print("- Review coverage report in htmlcov/index.html")
            print("- Check for any TODO items in test files")
            print("- Consider adding more edge case tests")
        
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        print("\n🔧 Debugging tips:")
        print("- Run with --verbose for detailed output")
        print("- Run specific test categories to isolate issues")
        print("- Check logs for error details")
        
        sys.exit(1)


if __name__ == "__main__":
    main()