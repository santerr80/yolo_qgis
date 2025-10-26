# -*- coding: utf-8 -*-
"""
Utility module to fix stderr issues in QGIS environment
This module should be imported early to prevent NumPy stderr errors
"""

import sys
import io


def fix_stderr():
    """
    Fix stderr issue in QGIS environment where sys.stderr might be None.
    This prevents NumPy and other libraries from crashing when trying to write to stderr.
    """
    if sys.stderr is None:
        sys.stderr = io.StringIO()
        return True
    return False


def restore_stderr():
    """
    Restore stderr to None if it was originally None.
    This is mainly for cleanup purposes.
    """
    if hasattr(sys.stderr, 'getvalue'):
        # If stderr is a StringIO object, we can restore it to None
        sys.stderr = None
        return True
    return False


# Automatically apply the fix when this module is imported
fix_stderr()

