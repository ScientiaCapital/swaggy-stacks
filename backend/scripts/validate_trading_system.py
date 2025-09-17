#!/usr/bin/env python3
"""
Standalone Pre-Market Validation Script
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.validation.pre_market_validator import PreMarketValidator, ValidationResult
from app.core.config import settings

async def main():
    """Main validation runner"""
    parser = argparse.ArgumentParser(description='Pre-Market Trading System Validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--alerts', action='store_true', help='Send alerts on failures')
    args = parser.parse_args()

    print("🚀 SwaggyStacks Pre-Market Validation System")
    print("=" * 60)

    validator = PreMarketValidator()

    try:
        # Run validation
        summary = await validator.run_full_validation()

        if args.json:
            import json
            # Output JSON results
            result_data = {
                'overall_status': summary.overall_status.value,
                'market_ready': summary.market_ready,
                'total_checks': summary.total_checks,
                'passed': summary.passed,
                'warnings': summary.warnings,
                'failures': summary.failures,
                'errors': summary.errors,
                'critical_failures': summary.critical_failures,
                'total_duration_ms': summary.total_duration_ms,
                'timestamp': summary.timestamp.isoformat(),
                'checks': [
                    {
                        'name': check.name,
                        'status': check.status.value,
                        'message': check.message,
                        'critical': check.critical,
                        'duration_ms': check.duration_ms,
                        'details': check.details
                    }
                    for check in validator.checks
                ]
            }
            print(json.dumps(result_data, indent=2))
        else:
            # Human-readable output
            print(f"\n📊 VALIDATION SUMMARY")
            print(f"Overall Status: {_get_status_emoji(summary.overall_status)} {summary.overall_status.value}")
            print(f"Market Ready: {'✅ YES' if summary.market_ready else '❌ NO'}")
            print(f"Duration: {summary.total_duration_ms:.1f}ms")
            print(f"")
            print(f"Results: {summary.passed} passed, {summary.warnings} warnings, "
                  f"{summary.failures} failures, {summary.errors} errors")

            if summary.critical_failures > 0:
                print(f"⚠️  Critical Failures: {summary.critical_failures}")

            print(f"\n📋 DETAILED RESULTS")
            print("-" * 60)

            for check in validator.checks:
                status_emoji = _get_status_emoji(check.status)
                critical_marker = " [CRITICAL]" if check.critical else ""
                print(f"{status_emoji} {check.name}{critical_marker}")

                if args.verbose or check.status != ValidationResult.PASS:
                    print(f"   Message: {check.message}")
                    print(f"   Duration: {check.duration_ms:.1f}ms")

                    if args.verbose and check.details:
                        print(f"   Details: {check.details}")
                print()

            print("=" * 60)

            if summary.market_ready:
                print("🎉 SYSTEM READY FOR TRADING!")
                print("✅ All critical systems operational")
                if summary.warnings > 0:
                    print(f"⚠️  {summary.warnings} warnings detected - monitor closely")
                print("🚀 Run: cd backend && python3 run_production.py")
            else:
                print("🛑 SYSTEM NOT READY FOR TRADING")
                print("❌ Critical issues must be resolved before trading")
                if summary.critical_failures > 0:
                    print(f"🚨 {summary.critical_failures} critical failures detected")

        # Exit with appropriate code
        if summary.market_ready:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"❌ VALIDATION SYSTEM ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)

def _get_status_emoji(status: ValidationResult) -> str:
    """Get emoji for validation status"""
    return {
        ValidationResult.PASS: "✅",
        ValidationResult.WARN: "⚠️",
        ValidationResult.FAIL: "❌",
        ValidationResult.ERROR: "💥"
    }.get(status, "❓")

if __name__ == "__main__":
    asyncio.run(main())