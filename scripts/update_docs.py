#!/usr/bin/env python3
"""
RAGLib Documentation Update Runner

Simple script to run the documentation auto-updater with common configurations.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the documentation updater."""
    scripts_dir = Path(__file__).parent
    updater_script = scripts_dir / "auto_update_docs.py"
    
    if not updater_script.exists():
        print("❌ Auto-updater script not found!")
        return 1
    
    print("🚀 Running RAGLib Documentation Auto-Updater")
    print("=" * 50)
    
    # Run with verbose output
    try:
        result = subprocess.run([
            sys.executable, 
            str(updater_script),
            "--verbose"
        ], cwd=scripts_dir.parent)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Update cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Failed to run updater: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())